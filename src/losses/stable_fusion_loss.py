import torch
import torch.nn as nn
import torch.nn.functional as F

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