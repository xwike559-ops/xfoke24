import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def window_partition(x, window_size):
    # x: (B,H,W,C) -> windows: (B*nW, ws, ws, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    # windows: (B*nW, ws, ws, C) -> x: (B,H,W,C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention with relative position bias.
    + Support semantic prior M (1-channel) injected as logits bias BEFORE softmax.
    """
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 prior_beta=1.0, learnable_prior=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # semantic prior strength
        if learnable_prior:
            self.prior_beta = nn.Parameter(torch.tensor(float(prior_beta)))
        else:
            self.register_buffer("prior_beta", torch.tensor(float(prior_beta)), persistent=False)

        # relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _semantic_bias(self, M_win, eps=1e-6):
        """
        M_win: (B_, N) or (B_, N, 1) in [0,1]
        return bias: (B_, 1, N, N)
        """
        if M_win is None:
            return None
        if M_win.dim() == 3:
            M_win = M_win.squeeze(-1)
        M_win = M_win.clamp(0.0, 1.0)

        # bias_ij = 0.5*(m_i + m_j)
        bi = M_win.unsqueeze(2)  # (B_,N,1)
        bj = M_win.unsqueeze(1)  # (B_,1,N)
        bias = 0.5 * (bi + bj)   # (B_,N,N)

        # 可选：让 bias 更尖锐一点（对差异区域更聚焦）
        # bias = torch.log(bias + eps)

        return bias.unsqueeze(1)  # (B_,1,N,N)

    def forward(self, x, mask=None, M_win=None):
        """
        x: (B_, N, C)
        mask: (nW, N, N) or None
        M_win: (B_, N) or (B_, N, 1) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, heads, N, N)

        # relative position bias
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        rpb = rpb.permute(2, 0, 1).contiguous()  # (heads, N, N)
        attn = attn + rpb.unsqueeze(0)

        # ---- semantic prior bias (logits-level) ----
        sb = self._semantic_bias(M_win)
        if sb is not None:
            attn = attn + self.prior_beta * sb  # broadcast to heads

        # ---- SW-MSA mask (prevent cross-window) ----
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock2D(nn.Module):
    """
    Support passing semantic prior map M_map aligned to token grid:
      M_map: (B, H, W, 1) or (B,1,H,W)
    """
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 prior_beta=1.0, learnable_prior=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            prior_beta=prior_beta, learnable_prior=learnable_prior
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def _build_attn_mask(self, H, W, device):
        """
        Standard Swin SW-MSA attention mask:
        returns (nW, N, N) with 0 or -100.
        """
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)  # (1,H,W,1)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask  # (nW, N, N)

    def forward(self, x, H, W, M_map=None):
        """
        x: (B, L, C) where L=H*W
        M_map optional: (B,H,W,1) or (B,1,H,W)
        """
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) != H*W ({H}*{W})"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # prepare M_map to (B,H,W,1)
        if M_map is not None:
            if M_map.dim() == 4 and M_map.shape[1] == 1:  # (B,1,H,W)
                M_map = M_map.permute(0, 2, 3, 1).contiguous()
            elif M_map.dim() == 4 and M_map.shape[-1] == 1:
                pass
            else:
                raise ValueError("M_map must be (B,1,H,W) or (B,H,W,1)")

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_M = torch.roll(M_map, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if M_map is not None else None
        else:
            shifted_x = x
            shifted_M = M_map

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (B*nW, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (B_, N, C)

        if shifted_M is not None:
            m_windows = window_partition(shifted_M, self.window_size)  # (B*nW, ws, ws, 1)
            m_windows = m_windows.view(-1, self.window_size * self.window_size, 1)  # (B_, N, 1)
        else:
            m_windows = None

        # attention mask for SW-MSA
        attn_mask = self._build_attn_mask(H, W, device=x.device)

        # W-MSA/SW-MSA with semantic prior
        attn_windows = self.attn(x_windows, mask=attn_mask, M_win=m_windows)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN + residual
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
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, E, H', W') -> (B, L, E)
        if self.norm is not None:
            x = self.norm(x)
        return x
