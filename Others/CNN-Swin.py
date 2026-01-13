import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ===============================================================================
# Part 1: 核心模块 - 2D Swin Transformer Components
# (无需修改，直接作为底层支持)
# ===============================================================================

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


# ===============================================================================
# Part 2: 混合融合模型 (Hybrid Fusion Net)
# ===============================================================================

class HybridFusionNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, img_size=512):
        super(HybridFusionNet, self).__init__()

        # 1. CNN Encoder (提取局部细节)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 2. Swin Transformer Bridge (提取全局上下文)
        self.embed_dim = 96
        self.patch_size = 2  # 保持较高分辨率

        # 输入通道为 64*2=128
        #把 CNN 特征图变成 token 序列
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=self.patch_size,
            in_chans=base_channels * 2, embed_dim=self.embed_dim
        )

        # 两层 Transformer Block (Window Size=8)
        #token 通道数
        self.swin_layers = nn.ModuleList([
            SwinTransformerBlock2D(dim=self.embed_dim, num_heads=4, window_size=8, shift_size=0),
            SwinTransformerBlock2D(dim=self.embed_dim, num_heads=4, window_size=8, shift_size=4)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # 3. Decoder & Mask Generator
        #up_conv：把 256×256 的全局特征上采样回 512×512
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, base_channels, kernel_size=2, stride=2),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )

        # 决策图生成器
        self.mask_generator = nn.Sequential(
            #把三路信息融合压缩到 64 通道
            nn.Conv2d(base_channels * 3, base_channels, 3, padding=1),  # 128(CNN) + 64(Trans)
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),  # 输出单通道 Mask   mask_logits: (B,1,512,512)
        )
        # 添加以下初始化逻辑
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, img1, img2):
        # 1. 局部特征提取
        feat1 = self.cnn_encoder(img1)
        feat2 = self.cnn_encoder(img2)

        # 2. 特征拼接与全局建模
        concat_feat = torch.cat([feat1, feat2], dim=1)  # (B, 128, H, W)

        swin_feat = self.patch_embed(concat_feat)  # (B, L, C)
        H, W = self.patch_embed.patches_resolution

        #经过两层 Swin block
        for blk in self.swin_layers:
            swin_feat = blk(swin_feat, H, W)
        swin_feat = self.norm(swin_feat)

        # 恢复图像维度
        swin_feat = swin_feat.transpose(1, 2).view(-1, self.embed_dim, H, W)
        global_context = self.up_conv(swin_feat)  # (B, 64, H, W)

        # 3. 生成决策图
        combined_feat = torch.cat([feat1, feat2, global_context], dim=1)

        mask_logits = self.mask_generator(combined_feat)  # 未归一化的 Logits (-inf, +inf)
        mask = torch.sigmoid(mask_logits)
        mask = mask.clamp(1e-4, 1.0 - 1e-4)

        # 4. 基于Mask的融合 (物理意义明确)
        # Mask=1 -> 取img1, Mask=0 -> 取img2
        fused_image = mask * img1 + (1 - mask) * img2

        return fused_image, mask, mask_logits


# ===============================================================================
# Part 3: 梯度引导的无监督损失函数
# ===============================================================================
class StableFusionLoss(nn.Module):
    def __init__(self, w_grad=10.0, w_intensity=1.0, w_mask=0.1):
        super().__init__()
        self.w_grad = w_grad
        self.w_intensity = w_intensity
        self.w_mask = w_mask

        # Sobel 核（固定，不参与训练）
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = sobel_x.t()

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def sobel_grad(self, x):
        b, c, h, w = x.shape
        sobel_x = self.sobel_x.repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(c, 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=c)

        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, img1, img2, fused, mask):
        # 1️⃣ 梯度最大值保留
        g1 = self.sobel_grad(img1)
        g2 = self.sobel_grad(img2)
        gf = self.sobel_grad(fused)
        grad_max = torch.max(g1, g2)
        loss_grad = F.l1_loss(gf, grad_max)

        # 2️⃣ 像素强度一致性
        intensity_target = torch.max(img1, img2)
        loss_intensity = F.l1_loss(fused, intensity_target)

        # 3️⃣ Mask 正则（防止全 0 / 全 1）
        loss_mask = torch.mean(mask * (1 - mask))

        total_loss = (
            self.w_grad * loss_grad +
            self.w_intensity * loss_intensity +
            self.w_mask * loss_mask
        )

        return total_loss, loss_grad, loss_intensity, loss_mask
# ===============================================================================
# Part 4: 数据集与工具函数
# ===============================================================================

class MultiFocusDataset(Dataset):
    def __init__(self, root_dir, img_size=512):  # 修改默认尺寸为512以适配Swin
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_pairs = []

        print(f"Loading dataset from: {root_dir}")
        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset directory {root_dir} does not exist!")

        for pair_dir in os.listdir(root_dir):
            pair_path = os.path.join(root_dir, pair_dir)
            if os.path.isdir(pair_path):
                files = os.listdir(pair_path)
                a_files = [f for f in files if 'a' in f.lower()]
                b_files = [f for f in files if 'b' in f.lower()]

                for a_file in a_files:
                    for b_file in b_files:
                        img_a_path = os.path.join(pair_path, a_file)
                        img_b_path = os.path.join(pair_path, b_file)
                        self.image_pairs.append((img_a_path, img_b_path))

        # 全局配对回退策略
        if len(self.image_pairs) == 0:
            print("⚠️ Switching to global sequential pairing...")
            all_images = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                          if f.lower().endswith(('.jpg', '.png', '.bmp'))]
            all_images.sort()
            for i in range(0, len(all_images) - 1, 2):
                self.image_pairs.append((all_images[i], all_images[i + 1]))

        print(f"Total {len(self.image_pairs)} image pairs found")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_a_path, img_b_path = self.image_pairs[idx]
        try:
            img_a = cv2.imread(img_a_path)
            img_b = cv2.imread(img_b_path)

            # 强制转为RGB
            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
            img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

            # Resize到512 (适应Swin Transformer Window操作)
            img_a = cv2.resize(img_a, (self.img_size, self.img_size))
            img_b = cv2.resize(img_b, (self.img_size, self.img_size))

            img_a = img_a.astype(np.float32) / 255.0
            img_b = img_b.astype(np.float32) / 255.0

            img_a = torch.from_numpy(img_a).permute(2, 0, 1)
            img_b = torch.from_numpy(img_b).permute(2, 0, 1)
            return img_a, img_b
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.image_pairs))


def train_model(model, train_loader, val_loader, criterion,
                optimizer, device, num_epochs=10):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for img1, img2 in pbar:
            img1 = img1.to(device)
            img2 = img2.to(device)

            optimizer.zero_grad()

            fused, mask, _ = model(img1, img2)

            loss, l_grad, l_int, l_mask = criterion(
                img1, img2, fused, mask
            )

            if torch.isnan(loss):
                raise RuntimeError("NaN detected in loss")

            loss.backward()

            # 梯度裁剪（Transformer 必须）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Grad": f"{l_grad.item():.4f}",
                "Int": f"{l_int.item():.4f}",
                "Mask": f"{l_mask.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for img1, img2 in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                fused, mask, _ = model(img1, img2)
                loss, _, _, _ = criterion(img1, img2, fused, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

    return model

def test_model(model, test_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for i, (img1, img2) in enumerate(test_loader):
            img1, img2 = img1.to(device), img2.to(device)
            fused_img, mask, _ = model(img1, img2)

            # 转 Numpy
            res_list = [img1, img2, fused_img]
            np_imgs = []
            for t in res_list:
                img = t.squeeze().cpu().permute(1, 2, 0).numpy()
                np_imgs.append((img * 255).astype(np.uint8))

            # 处理 Mask (单通道)
            mask_np = mask.squeeze().cpu().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)
            mask_colormap = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)  # 热力图可视化

            results.append((np_imgs[0], np_imgs[1], np_imgs[2], mask_colormap))

            if i < 5:
                cv2.imwrite(f'result_fused_10_epoch{i}.png', cv2.cvtColor(np_imgs[2], cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'result_mask_10_epoch{i}.png', mask_colormap)
    return results


def visualize_results(results):
    for i, (img1, img2, fused, mask) in enumerate(results[:3]):
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1);
        plt.imshow(img1);
        plt.title('Input 1');
        plt.axis('off')
        plt.subplot(1, 4, 2);
        plt.imshow(img2);
        plt.title('Input 2');
        plt.axis('off')
        plt.subplot(1, 4, 3);
        plt.imshow(fused);
        plt.title('Fused Result');
        plt.axis('off')
        plt.subplot(1, 4, 4);
        plt.imshow(mask);
        plt.title('Decision Mask (Attention)');
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'vis_swin_result_{i}.png')
        plt.show()


# ===============================================================================
# Part 5: 主函数
# ===============================================================================

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")  # 强制指定为 CPU
    print(f"Using device: {device}")

    # 数据集路径 (请修改为你实际的路径)
    train_dir = "E:\\Multi-fusion_cnn-swinv2\\dataset\\train"
    val_dir = "E:\\Multi-fusion_cnn-swinv2\\dataset\\val"
    test_dir = "E:\\Multi-fusion_cnn-swinv2\\dataset\\test"

    # 注意：这里将 img_size 强制设为 512，适配 Swin Transformer 的 Patch/Window 计算
    try:
        train_dataset = MultiFocusDataset(train_dir, img_size=512)
        val_dataset = MultiFocusDataset(val_dir, img_size=512)
        test_dataset = MultiFocusDataset(test_dir, img_size=512)
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)  # 显存不够可减小 batch_size
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Samples - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化混合模型
    model = HybridFusionNet(img_size=512).to(device)

    # 打印参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Swin-Fusion Model Parameters: {params / 1e6:.2f}M")

    # 使用新的无监督损失函数
    criterion = StableFusionLoss(
        w_grad=10.0,
        w_intensity=1.0,
        w_mask=0.1
    ).to(device)
    # 使用 AdamW 优化器，对 Transformer 更友好
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-5,  # 比你原来更安全
        weight_decay=1e-4
    )

    print("Starting training with Swin Transformer...")
    # 训练 100 个 Epoch 足够看到效果 (Demo用)，实际科研可跑更多
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

    torch.save(trained_model.state_dict(), "swin_fusion_final.pth")
    print("Model saved.")

    print("Testing model...")
    results = test_model(trained_model, test_loader, device)
    visualize_results(results)


if __name__ == "__main__":
    main()