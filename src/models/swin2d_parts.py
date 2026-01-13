import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)        #训练时随机“丢掉”一些神经元，防止模型太自信（过拟合

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)        #交换维度顺序
    return windows                                                                                  #(B, H, W, C)---->(总窗口数, ws, ws, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 创建相对位置偏置表。尺寸为(2*window_size-1)², num_heads，为所有可能的相对位置对预先分配可学习参数。
        # 这是Swin的核心创新——从绝对位置编码转为相对位置编码，让模型学习"像素A在像素B的左上角"这样的关系，而不是"像素A在第3行第4列"。
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        #生成窗口内所有像素的坐标。meshgrid创建坐标网格，flatten将坐标展平为2×M²矩阵（M = window_size），第一行是行坐标，第二行是列坐标。
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        #将相对坐标映射到一维索引。
        # 骚操作三连：1) 加window_size-1将范围[-(M-1), M-1]偏移到[0, 2M-2]；
        # 2) 对行坐标乘2M-1，制造唯一ID；
        # 3) 行列相加得到唯一索引。这波哈希操作很秀，用数学保证了每个相对位置有唯一索引。
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)

        #注册为buffer而非parameter，表明这是不学习的常数。这样保存模型时会被序列化，但不会在优化器中更新。不这么干的话，每次推理都要重新计算相对位置索引，浪费时间。
        self.register_buffer("relative_position_index", relative_position_index)

        #标准Transformer配置。注意qkv一次投影出Q、K、V三个矩阵，比分开投影节省两次矩阵乘法。trunc_normal_用截断正态分布初始化位置偏置表，防止初始化过大导致训练不稳定。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        #计算Q、K、V。一波维度操作猛如虎：1) 线性投影得到(B_, N, 3*C)；
        # 2) reshape拆出3个头维度；
        # 3) permute重排为(3, B_, num_heads, N, head_dim)。
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #计算注意力分数。标准点积注意力，先缩放Q再计算QK^T
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        #注入相对位置偏置。这是Swin的灵魂操作：
        # 1) 用预计算的索引表从偏置表中查找对应位置的偏置；
        # 2) reshape成(M², M², num_heads)；
        # 3) 加到注意力分数上。unsqueeze(0)给batch维度留位置。对比ViT的绝对位置编码，这种相对编码能更好处理可变分辨率。
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock2D(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        #标准Transformer配置加上随机深度。DropPath是StochasticDepth，随网络加深增加丢弃概率，防止过拟合。MLP扩展比为4，这是Transformer的标准配置。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) does not match H*W ({H}*{W}). Check input image size."

        #残差连接准备和空间重建。先保存输入用于残差连接，然后做LayerNorm，最后reshape回空间格式(B,H,W,C)。Pre-norm比Post-norm训练更稳定，梯度更好传播。
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        #窗口注意力计算。调用WindowAttention模块，内部处理QKV投影、相对位置编码、注意力掩码等。如果是SW-MSA，WindowAttention内部会根据mask参数屏蔽跨窗口的注意力。
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        #逆循环移位。如果是SW-MSA，需要将移位后的特征图移回原始位置。(shift_size, shift_size)是向右下移动，正好抵消前面的(-shift_size, -shift_size)。不这么做的话，特征图位置就错乱了。
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        #重新序列化。从空间格式(B,H,W,C)变回序列格式(B,H*W,C)，准备输入FFN。这种序列↔空间的来回转换是Vision Transformer的基操。
        x = x.view(B, H * W, C)

        # FFN
        #残差连接和FFN。标准Transformer块结构，但加了DropPath。
        # 注意这里有两个残差连接：注意力输出和原始输入，FFN输出和注意力输出。DropPath在训练时随机丢弃整个路径，测试时是恒等映射，相当于深度网络的集成学习。
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed2D(nn.Module):
    def __init__(self, img_size=512, patch_size=2, in_chans=64, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        #flatten(2):把空间维度拉直  (B, 96, 256, 256)-->(B, 96, 256*256)
        #transpose(1, 2)交换顺序
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x
