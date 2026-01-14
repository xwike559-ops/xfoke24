import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin2d_parts import (
    PatchEmbed2D,
    SwinTransformerBlock2D
)

class DenseConvBlock(nn.Module):
    """
    Dense-ish CNN block for high-frequency (texture/edges).
    light but deeper than plain 2-conv.
    """
    def __init__(self, in_ch, growth=32, layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(ch, growth, 3, padding=1, bias=False),
                nn.BatchNorm2d(growth),
                nn.ReLU(inplace=True),
            ))
            ch += growth
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            y = layer(torch.cat(feats, dim=1))
            feats.append(y)
        out = self.fuse(torch.cat(feats, dim=1))
        return out


class SpatialEnhanceModule(nn.Module):
    """
    EM: spatial attention enhancement (low-freq guides high-freq).
    outputs reweighted features.
    """
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, 1, 1, bias=True),
        )

    def forward(self, x):
        # x: (B,C,H,W) -> weight: (B,1,H,W)
        w = torch.sigmoid(self.proj(x))
        return x * w + x


class PatchTokenMixer(nn.Module):
    """
    Convert (B,C,H,W) -> tokens by PatchEmbed2D, apply Swin blocks, back to (B,C,H,W)
    """
    def __init__(self, img_size, in_ch, embed_dim, patch_size=2, num_heads=4, window_size=8):
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dim
        )
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 2 Swin blocks (W-MSA + SW-MSA) like your current style but per-stage
        self.blocks = nn.ModuleList([
            SwinTransformerBlock2D(dim=embed_dim, num_heads=num_heads,
                                   window_size=window_size, shift_size=0),
            SwinTransformerBlock2D(dim=embed_dim, num_heads=num_heads,
                                   window_size=window_size, shift_size=window_size // 2),
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # project back to in_ch
        self.back = nn.Conv2d(embed_dim, in_ch, 1)

    def forward(self, x):
        # x: (B,C,H,W)
        tokens = self.patch_embed(x)  # (B, L, E)
        H, W = self.patch_embed.patches_resolution
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        tokens = self.norm(tokens)
        feat = tokens.transpose(1, 2).contiguous().view(-1, self.embed_dim, H, W)  # (B,E,H',W')
        feat = self.back(feat)  # (B,C,H',W')
        # NOTE: spatial size is down by patch_size
        return feat


class CrossAttentionFuse(nn.Module):
    """
    Cross-attention between img1/img2 tokens at bottleneck.
    (No semantic prior yet; can be extended with mask M later.)
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, t1, t2):
        # t1,t2: (B,L,C)
        # t1 queries t2, and t2 queries t1
        a12, _ = self.mha(query=t1, key=t2, value=t2, need_weights=False)
        a21, _ = self.mha(query=t2, key=t1, value=t1, need_weights=False)
        z = a12 + a21
        z = self.ln(z + self.ffn(z))
        return z

class HybridFusionNet(nn.Module):
    """
    Deeper multi-scale hybrid encoder + cross-attn fusion + UNet decoder mask.
    Drop-in replacement of your current HybridFusionNet.
    """
    def __init__(self, in_channels=3, base_channels=64, img_size=512):
        super().__init__()
        self.img_size = img_size
        C = base_channels

        # stem (per-image)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # 4-stage CNN dense blocks (shared for img1/img2)
        self.cnn_stage1 = nn.Sequential(DenseConvBlock(C, growth=32, layers=3), DenseConvBlock(C, growth=32, layers=3))
        self.cnn_stage2 = nn.Sequential(DenseConvBlock(C, growth=32, layers=3), DenseConvBlock(C, growth=32, layers=3))
        self.cnn_stage3 = nn.Sequential(DenseConvBlock(C, growth=32, layers=3), DenseConvBlock(C, growth=32, layers=3))
        self.cnn_stage4 = nn.Sequential(DenseConvBlock(C, growth=32, layers=3), DenseConvBlock(C, growth=32, layers=3))

        # downsample between stages
        self.down = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # per-stage Transformer mixers (run on concatenated features of img1/img2)
        # stage1: 512 -> patch2 => 256
        self.tr1 = PatchTokenMixer(img_size=img_size,     in_ch=C * 2, embed_dim=96,  patch_size=2, num_heads=4, window_size=8)
        # stage2: 256 -> patch2 => 128
        self.tr2 = PatchTokenMixer(img_size=img_size//2,  in_ch=C * 2, embed_dim=128, patch_size=2, num_heads=4, window_size=8)
        # stage3: 128 -> patch2 => 64
        self.tr3 = PatchTokenMixer(img_size=img_size//4,  in_ch=C * 2, embed_dim=160, patch_size=2, num_heads=5, window_size=8)
        # stage4: 64  -> patch2 => 32
        self.tr4 = PatchTokenMixer(img_size=img_size//8,  in_ch=C * 2, embed_dim=192, patch_size=2, num_heads=6, window_size=8)

        # project transformer outputs back to C and upsample to match CNN feature sizes per stage
        self.tr_proj1 = nn.Conv2d(C * 2, C, 1)
        self.tr_proj2 = nn.Conv2d(C * 2, C, 1)
        self.tr_proj3 = nn.Conv2d(C * 2, C, 1)
        self.tr_proj4 = nn.Conv2d(C * 2, C, 1)

        # EM modules (spatial attention enhance) per stage
        self.em1 = SpatialEnhanceModule(C)
        self.em2 = SpatialEnhanceModule(C)
        self.em3 = SpatialEnhanceModule(C)
        self.em4 = SpatialEnhanceModule(C)

        # bottleneck cross-attention fusion (token-level)
        self.bottleneck_embed = nn.Conv2d(C * 2, 192, 1)
        self.cross_fuse = CrossAttentionFuse(dim=192, num_heads=6)
        self.bottleneck_back = nn.Conv2d(192, C, 1)

        # decoder (UNet-ish) to produce mask
        self.up3 = nn.ConvTranspose2d(C, C, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(C, C, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(C, C, 2, stride=2)
        self.up0 = nn.ConvTranspose2d(C, C, 2, stride=2)

        self.dec3 = nn.Sequential(nn.Conv2d(C * 2, C, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(C * 2, C, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(C * 2, C, 3, padding=1), nn.ReLU(inplace=True))
        self.dec0 = nn.Sequential(nn.Conv2d(C * 2, C, 3, padding=1), nn.ReLU(inplace=True))

        # multi-scale mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 1, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _stage(self, f1, f2, cnn_block, tr_mixer, tr_proj, em):
        """
        One stage:
        - CNN dense per-image
        - Transformer on concat (global low-freq)
        - bring transformer feature to CNN resolution and enhance
        """
        f1 = cnn_block(f1)
        f2 = cnn_block(f2)

        cat = torch.cat([f1, f2], dim=1)              # (B,2C,H,W)
        tr = tr_mixer(cat)                            # (B,2C,H/2,W/2) because patch_size=2 and back->2C
        tr = tr_proj(tr)                              # (B,C,H/2,W/2)
        tr = F.interpolate(tr, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        # EM: use (f1+f2+tr) as a spatial guide
        guide = (f1 + f2 + tr) / 3.0
        guide = em(guide)

        # inject guide into both branches (light residual)
        f1 = f1 + guide
        f2 = f2 + guide
        return f1, f2, guide

    def forward(self, img1, img2):
        # ---- stem ----
        f1 = self.stem(img1)
        f2 = self.stem(img2)

        # ---- stage1 (512) ----
        s1_f1, s1_f2, s1_g = self._stage(f1, f2, self.cnn_stage1, self.tr1, self.tr_proj1, self.em1)

        # ---- down -> stage2 (256) ----
        d1_f1, d1_f2 = self.down(s1_f1), self.down(s1_f2)
        s2_f1, s2_f2, s2_g = self._stage(d1_f1, d1_f2, self.cnn_stage2, self.tr2, self.tr_proj2, self.em2)

        # ---- down -> stage3 (128) ----
        d2_f1, d2_f2 = self.down(s2_f1), self.down(s2_f2)
        s3_f1, s3_f2, s3_g = self._stage(d2_f1, d2_f2, self.cnn_stage3, self.tr3, self.tr_proj3, self.em3)

        # ---- down -> stage4 (64) ----
        d3_f1, d3_f2 = self.down(s3_f1), self.down(s3_f2)
        s4_f1, s4_f2, s4_g = self._stage(d3_f1, d3_f2, self.cnn_stage4, self.tr4, self.tr_proj4, self.em4)

        # ---- bottleneck cross-attention fusion (token) ----
        # tokens at 64x64 scale
        cat4 = torch.cat([s4_f1, s4_f2], dim=1)     # (B,2C,64,64)
        b = self.bottleneck_embed(cat4)             # (B,192,64,64)
        B, Cb, H, W = b.shape
        t = b.flatten(2).transpose(1, 2)            # (B,L,C)
        # split into two "domains" by simple channel split proxy
        # (Alternatively use separate proj from s4_f1/s4_f2; keep it simple & stable)
        t1 = t
        t2 = torch.flip(t, dims=[1])                # cheap asymmetry to avoid collapse
        z = self.cross_fuse(t1, t2)                 # (B,L,C)
        z = z.transpose(1, 2).contiguous().view(B, Cb, H, W)
        z = self.bottleneck_back(z)                 # (B,C,64,64)

        # ---- decoder with skip (use guide features) ----
        x = z  # 64
        x = self.up3(x)                             # 128
        x = self.dec3(torch.cat([x, s3_g], dim=1))

        x = self.up2(x)                             # 256
        x = self.dec2(torch.cat([x, s2_g], dim=1))

        x = self.up1(x)                             # 512
        x = self.dec1(torch.cat([x, s1_g], dim=1))

        # keep one more refinement at full-res
        x = self.dec0(torch.cat([x, (self.stem(img1) + self.stem(img2)) / 2.0], dim=1))

        mask_logits = self.mask_head(x)             # (B,1,512,512)
        mask = torch.sigmoid(mask_logits).clamp(1e-4, 1.0 - 1e-4)

        fused_image = mask * img1 + (1 - mask) * img2
        return fused_image, mask, mask_logits