import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin2d_parts import (
    PatchEmbed2D,
    SwinTransformerBlock2D
)

# ----------------------------
# Semantic Prior M (DeepLabv3 + focus difference)
# ----------------------------
class SemanticPriorM(nn.Module):
    """
    Build 1-channel semantic-guided focus-difference prior M in [0,1]:
      D = |LoG/Laplacian(gray(img1)) - LoG/Laplacian(gray(img2))|
      S_conf = top1 softmax prob from DeepLabv3
      S_edge = |∇S_conf| (Sobel)
      M = Normalize( D * (1 + lam*S_edge) ) ^ gamma

    Inputs img1,img2 are [0,1] float tensors: (B,3,H,W)
    """
    def __init__(self, diff_mode="lap", semantic_mode="edge", lam=2.0, gamma=1.5,
                 use_deeplab=True):
        super().__init__()
        self.diff_mode = diff_mode
        self.semantic_mode = semantic_mode
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.use_deeplab = bool(use_deeplab)

        if self.use_deeplab:
            try:
                from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
            except Exception as e:
                raise ImportError("torchvision segmentation is required for DeepLabv3 prior.") from e

            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.seg = deeplabv3_resnet50(weights=weights).eval()
            for p in self.seg.parameters():
                p.requires_grad_(False)

            # DeepLabv3 normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("dl_mean", mean, persistent=False)
            self.register_buffer("dl_std", std, persistent=False)
        else:
            self.seg = None
            self.register_buffer("dl_mean", torch.zeros(1,3,1,1), persistent=False)
            self.register_buffer("dl_std", torch.ones(1,3,1,1), persistent=False)

    @staticmethod
    def _to_gray(x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    @staticmethod
    def _normalize01(m):
        m = m - m.amin(dim=(-2, -1), keepdim=True)
        m = m / (m.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        return m

    @staticmethod
    def _laplacian_mag(gray):
        k = torch.tensor([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        y = F.conv2d(gray, k, padding=1)
        return y.abs()

    @staticmethod
    def _grad_mag(x):
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    @torch.no_grad()
    def _semantic_conf(self, img):
        """
        img: (B,3,H,W) in [0,1]
        return S_conf: (B,1,H,W) in [0,1]
        """
        if self.seg is None:
            # no semantic model: return ones (neutral)
            return torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), device=img.device, dtype=img.dtype)

        self.seg.eval()

        x = (img - self.dl_mean.to(img.device, img.dtype)) / self.dl_std.to(img.device, img.dtype)
        logits = self.seg(x)["out"]  # (B,C,h,w)
        prob = logits.softmax(dim=1)
        prob = F.interpolate(prob, size=img.shape[-2:], mode="bilinear", align_corners=False)
        conf, _ = prob.max(dim=1, keepdim=True)  # (B,1,H,W)
        return conf

    @torch.no_grad()
    def forward(self, img1, img2):
        """
        img1,img2: (B,3,H,W) in [0,1]
        return M: (B,1,H,W) in [0,1]
        """
        g1 = self._to_gray(img1)
        g2 = self._to_gray(img2)

        if self.diff_mode == "lap":
            D = (self._laplacian_mag(g1) - self._laplacian_mag(g2)).abs()
        elif self.diff_mode == "grad":
            D = (self._grad_mag(g1) - self._grad_mag(g2)).abs()
        else:
            raise ValueError("diff_mode must be 'lap' or 'grad'")
        D = self._normalize01(D)

        S_conf = self._semantic_conf(img1)  # 用 img1 跑语义即可（也可换成 (img1+img2)/2）

        if self.semantic_mode == "edge":
            S_edge = self._grad_mag(S_conf)
            S_edge = self._normalize01(S_edge)
            M = D * (1.0 + self.lam * S_edge)
        elif self.semantic_mode == "gate":
            M = D * S_conf
        else:
            raise ValueError("semantic_mode must be 'edge' or 'gate'")

        M = self._normalize01(M)
        if self.gamma != 1.0:
            M = M.clamp(0, 1).pow(self.gamma)
        return M


# ----------------------------
# Original blocks
# ----------------------------
class DenseConvBlock(nn.Module):
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
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, 1, 1, bias=True),
        )

    def forward(self, x):
        w = torch.sigmoid(self.proj(x))
        return x * w + x


class PatchTokenMixer(nn.Module):
    """
    Convert (B,C,H,W) -> tokens by PatchEmbed2D, apply Swin blocks, back to (B,C,H',W')
    + Accept semantic prior M at same spatial size as input x, then resize to token grid and pass down.
    """
    def __init__(self, img_size, in_ch, embed_dim, patch_size=2, num_heads=4, window_size=8,
                 prior_beta=1.0, learnable_prior=False):
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dim
        )
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.blocks = nn.ModuleList([
            SwinTransformerBlock2D(dim=embed_dim, num_heads=num_heads,
                                   window_size=window_size, shift_size=0,
                                   prior_beta=prior_beta, learnable_prior=learnable_prior),
            SwinTransformerBlock2D(dim=embed_dim, num_heads=num_heads,
                                   window_size=window_size, shift_size=window_size // 2,
                                   prior_beta=prior_beta, learnable_prior=learnable_prior),
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.back = nn.Conv2d(embed_dim, in_ch, 1)

    def forward(self, x, M=None):
        """
        x: (B,C,H,W)
        M: (B,1,H,W) aligned to x resolution (same H,W). If None -> no prior injection.
        """
        tokens = self.patch_embed(x)  # (B, L, E)
        H, W = self.patch_embed.patches_resolution

        # prepare M_map aligned to token grid size (H,W)
        if M is not None:
            M_tok = F.interpolate(M, size=(H, W), mode="bilinear", align_corners=False)  # (B,1,H,W)
            M_map = M_tok.permute(0, 2, 3, 1).contiguous()  # (B,H,W,1)
        else:
            M_map = None

        for blk in self.blocks:
            tokens = blk(tokens, H, W, M_map=M_map)

        tokens = self.norm(tokens)
        feat = tokens.transpose(1, 2).contiguous().view(-1, self.embed_dim, H, W)  # (B,E,H,W)
        feat = self.back(feat)  # (B,C,H,W)
        return feat


class CrossAttentionFuse(nn.Module):
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
        a12, _ = self.mha(query=t1, key=t2, value=t2, need_weights=False)
        a21, _ = self.mha(query=t2, key=t1, value=t1, need_weights=False)
        z = a12 + a21
        z = self.ln(z + self.ffn(z))
        return z


class HybridFusionNet(nn.Module):
    """
    Hybrid encoder + Swin token mixer + decoder mask
    + SemanticPriorM injected into Swin window attention at all stages.
    """
    def __init__(self, in_channels=3, base_channels=64, img_size=512,
                 use_semantic_prior=True,
                 prior_diff_mode="lap", prior_semantic_mode="edge",
                 prior_lam=2.0, prior_gamma=1.5,
                 prior_beta=1.0, learnable_prior=False,
                 use_deeplab=True):
        super().__init__()
        self.img_size = img_size
        C = base_channels

        # semantic prior module
        self.use_semantic_prior = bool(use_semantic_prior)
        if self.use_semantic_prior:
            self.semantic_prior = SemanticPriorM(
                diff_mode=prior_diff_mode,
                semantic_mode=prior_semantic_mode,
                lam=prior_lam,
                gamma=prior_gamma,
                use_deeplab=use_deeplab
            )
        else:
            self.semantic_prior = None

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # CNN stages
        self.cnn_stage1 = nn.Sequential(DenseConvBlock(C, 32, 3), DenseConvBlock(C, 32, 3))
        self.cnn_stage2 = nn.Sequential(DenseConvBlock(C, 32, 3), DenseConvBlock(C, 32, 3))
        self.cnn_stage3 = nn.Sequential(DenseConvBlock(C, 32, 3), DenseConvBlock(C, 32, 3))
        self.cnn_stage4 = nn.Sequential(DenseConvBlock(C, 32, 3), DenseConvBlock(C, 32, 3))

        self.down = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Transformer mixers (concat features)
        self.tr1 = PatchTokenMixer(img_size=img_size,     in_ch=C * 2, embed_dim=96,
                                   patch_size=2, num_heads=4, window_size=8,
                                   prior_beta=prior_beta, learnable_prior=learnable_prior)
        self.tr2 = PatchTokenMixer(img_size=img_size//2,  in_ch=C * 2, embed_dim=128,
                                   patch_size=2, num_heads=4, window_size=8,
                                   prior_beta=prior_beta, learnable_prior=learnable_prior)
        self.tr3 = PatchTokenMixer(img_size=img_size//4,  in_ch=C * 2, embed_dim=160,
                                   patch_size=2, num_heads=5, window_size=8,
                                   prior_beta=prior_beta, learnable_prior=learnable_prior)
        self.tr4 = PatchTokenMixer(img_size=img_size//8,  in_ch=C * 2, embed_dim=192,
                                   patch_size=2, num_heads=6, window_size=8,
                                   prior_beta=prior_beta, learnable_prior=learnable_prior)

        self.tr_proj1 = nn.Conv2d(C * 2, C, 1)
        self.tr_proj2 = nn.Conv2d(C * 2, C, 1)
        self.tr_proj3 = nn.Conv2d(C * 2, C, 1)
        self.tr_proj4 = nn.Conv2d(C * 2, C, 1)

        self.em1 = SpatialEnhanceModule(C)
        self.em2 = SpatialEnhanceModule(C)
        self.em3 = SpatialEnhanceModule(C)
        self.em4 = SpatialEnhanceModule(C)

        self.bottleneck_embed = nn.Conv2d(C * 2, 192, 1)
        self.cross_fuse = CrossAttentionFuse(dim=192, num_heads=6)
        self.bottleneck_back = nn.Conv2d(192, C, 1)

        self.up3 = nn.ConvTranspose2d(C, C, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(C, C, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(C, C, 2, stride=2)
        self.up0 = nn.ConvTranspose2d(C, C, 2, stride=2)

        self.dec3 = nn.Sequential(nn.Conv2d(C * 2, C, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(C * 2, C, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(C * 2, C, 3, padding=1), nn.ReLU(inplace=True))
        self.dec0 = nn.Sequential(nn.Conv2d(C * 2, C, 3, padding=1), nn.ReLU(inplace=True))

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

    def _stage(self, f1, f2, cnn_block, tr_mixer, tr_proj, em, M_stage=None):
        f1 = cnn_block(f1)
        f2 = cnn_block(f2)

        cat = torch.cat([f1, f2], dim=1)  # (B,2C,H,W)

        # Transformer on concat with semantic prior M_stage
        tr = tr_mixer(cat, M=M_stage)     # (B,2C,H/2,W/2)
        tr = tr_proj(tr)                  # (B,C,H/2,W/2)
        tr = F.interpolate(tr, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        guide = (f1 + f2 + tr) / 3.0
        guide = em(guide)

        f1 = f1 + guide
        f2 = f2 + guide
        return f1, f2, guide

    def forward(self, img1, img2):
        # ---- semantic prior (compute once at full-res) ----
        if self.use_semantic_prior and self.semantic_prior is not None:
            # ensure seg model on same device
            self.semantic_prior.seg = self.semantic_prior.seg.to(img1.device) if self.semantic_prior.seg is not None else None
            M_full = self.semantic_prior(img1, img2)  # (B,1,512,512)
        else:
            M_full = None

        # ---- stem ----
        f1 = self.stem(img1)
        f2 = self.stem(img2)

        # stage1 (512)
        M1 = M_full
        s1_f1, s1_f2, s1_g = self._stage(f1, f2, self.cnn_stage1, self.tr1, self.tr_proj1, self.em1, M_stage=M1)

        # stage2 (256)
        d1_f1, d1_f2 = self.down(s1_f1), self.down(s1_f2)
        M2 = F.interpolate(M_full, size=d1_f1.shape[-2:], mode="bilinear", align_corners=False) if M_full is not None else None
        s2_f1, s2_f2, s2_g = self._stage(d1_f1, d1_f2, self.cnn_stage2, self.tr2, self.tr_proj2, self.em2, M_stage=M2)

        # stage3 (128)
        d2_f1, d2_f2 = self.down(s2_f1), self.down(s2_f2)
        M3 = F.interpolate(M_full, size=d2_f1.shape[-2:], mode="bilinear", align_corners=False) if M_full is not None else None
        s3_f1, s3_f2, s3_g = self._stage(d2_f1, d2_f2, self.cnn_stage3, self.tr3, self.tr_proj3, self.em3, M_stage=M3)

        # stage4 (64)
        d3_f1, d3_f2 = self.down(s3_f1), self.down(s3_f2)
        M4 = F.interpolate(M_full, size=d3_f1.shape[-2:], mode="bilinear", align_corners=False) if M_full is not None else None
        s4_f1, s4_f2, s4_g = self._stage(d3_f1, d3_f2, self.cnn_stage4, self.tr4, self.tr_proj4, self.em4, M_stage=M4)

        # ---- bottleneck cross-attn ----
        cat4 = torch.cat([s4_f1, s4_f2], dim=1)
        b = self.bottleneck_embed(cat4)             # (B,192,64,64)
        B, Cb, H, W = b.shape
        t = b.flatten(2).transpose(1, 2)            # (B,L,C)

        t1 = t
        t2 = torch.flip(t, dims=[1])                # cheap asymmetry
        z = self.cross_fuse(t1, t2)
        z = z.transpose(1, 2).contiguous().view(B, Cb, H, W)
        z = self.bottleneck_back(z)                 # (B,C,64,64)

        # ---- decoder ----
        x = z
        x = self.up3(x)                             # 128
        x = self.dec3(torch.cat([x, s3_g], dim=1))

        x = self.up2(x)                             # 256
        x = self.dec2(torch.cat([x, s2_g], dim=1))

        x = self.up1(x)                             # 512
        x = self.dec1(torch.cat([x, s1_g], dim=1))

        x = self.dec0(torch.cat([x, (self.stem(img1) + self.stem(img2)) / 2.0], dim=1))

        mask_logits = self.mask_head(x)             # (B,1,512,512)
        mask = torch.sigmoid(mask_logits).clamp(1e-4, 1.0 - 1e-4)

        fused_image = mask * img1 + (1 - mask) * img2
        return fused_image, mask, mask_logits
