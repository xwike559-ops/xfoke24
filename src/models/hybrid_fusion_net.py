import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .swin2d_parts import (
    PatchEmbed2D,
    SwinTransformerBlock2D
)

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
