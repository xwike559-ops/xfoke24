import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedFusionLoss(nn.Module):
    """
    可直接替换你当前 StableFusionLoss 的改进版（适合 Lytro）：

    L = λd * L_detail(LoG) + λs * (1-SSIM(fused, target_soft))
        + λb * mean(mask*(1-mask)) + λtv * TV(mask)

    兼容你的训练调用方式：
        loss, l_detail, l_ssim, l_bin, l_tv = criterion(img1, img2, fused, mask)

    说明：
    - LoG：用固定高斯平滑 + 拉普拉斯实现（可微、无参数）
    - target_soft：用清晰度权重（|LoG|）在像素级构建软目标
    - SSIM：实现了一个稳定版本（默认 11x11 高斯窗）
    - TV(mask)：让 mask 空间更连续，减少碎片/毛刺
    """

    def __init__(
        self,
        w_detail: float = 1.0,
        w_ssim: float = 0.5,
        w_mask_bin: float = 0.05,
        w_mask_tv: float = 0.1,
        # LoG相关
        gaussian_ksize: int = 5,
        gaussian_sigma: float = 1.0,
        # SSIM相关
        ssim_ksize: int = 11,
        ssim_sigma: float = 1.5,
        data_range: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.w_detail = w_detail
        self.w_ssim = w_ssim
        self.w_mask_bin = w_mask_bin
        self.w_mask_tv = w_mask_tv
        self.data_range = data_range
        self.eps = eps

        # 固定高斯核（LoG用）
        gk = self._build_gaussian_kernel2d(gaussian_ksize, gaussian_sigma)  # (1,1,k,k)
        self.register_buffer("gaussian_kernel", gk)

        # 固定拉普拉斯核（3x3）
        lap = torch.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("laplacian_kernel", lap.view(1, 1, 3, 3))

        # SSIM高斯窗
        ssim_w = self._build_gaussian_kernel2d(ssim_ksize, ssim_sigma)
        self.register_buffer("ssim_window", ssim_w)

        # SSIM常数（按经典定义）
        # C1=(K1*L)^2, C2=(K2*L)^2, 其中L=data_range
        K1, K2 = 0.01, 0.03
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2

    @staticmethod
    def _ensure_odd(k: int) -> int:
        k = int(k)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        return k

    def _build_gaussian_kernel2d(self, ksize: int, sigma: float) -> torch.Tensor:
        ksize = self._ensure_odd(ksize)
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / (kernel.sum() + 1e-12)
        return kernel.view(1, 1, ksize, ksize)

    def _gaussian_blur(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        kernel: (1,1,k,k) -> 每通道独立卷积
        """
        B, C, H, W = x.shape
        k = kernel.shape[-1]
        pad = k // 2
        weight = kernel.repeat(C, 1, 1, 1)  # (C,1,k,k)
        return F.conv2d(x, weight, padding=pad, groups=C)

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) -> 每通道独立拉普拉斯
        """
        B, C, H, W = x.shape
        weight = self.laplacian_kernel.repeat(C, 1, 1, 1)  # (C,1,3,3)
        return F.conv2d(x, weight, padding=1, groups=C)

    def _log_response(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoG近似：GaussianBlur -> Laplacian
        """
        x_s = self._gaussian_blur(x, self.gaussian_kernel)
        return self._laplacian(x_s)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算 SSIM，返回每张图的平均 SSIM（标量）。
        x,y: (B,C,H,W), assumed in [0, data_range]
        """
        # 用高斯窗计算局部均值与方差（逐通道独立）
        mu_x = self._gaussian_blur(x, self.ssim_window)
        mu_y = self._gaussian_blur(y, self.ssim_window)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = self._gaussian_blur(x * x, self.ssim_window) - mu_x2
        sigma_y2 = self._gaussian_blur(y * y, self.ssim_window) - mu_y2
        sigma_xy = self._gaussian_blur(x * y, self.ssim_window) - mu_xy

        # SSIM map
        num = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        ssim_map = num / (den + self.eps)

        # 先对通道、空间求均值，再对 batch 求均值
        return ssim_map.mean(dim=(1, 2, 3)).mean()

    def _tv_loss(self, m: torch.Tensor) -> torch.Tensor:
        """
        Total Variation: encouraging spatial smoothness.
        m: (B,1,H,W) or (B,C,H,W)
        """
        dh = torch.abs(m[:, :, 1:, :] - m[:, :, :-1, :]).mean()
        dw = torch.abs(m[:, :, :, 1:] - m[:, :, :, :-1]).mean()
        return dh + dw

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, fused: torch.Tensor, mask: torch.Tensor):
        """
        img1,img2,fused: (B,3,H,W) in [0,1]
        mask: (B,1,H,W) in (0,1)
        """

        # ---------- 1) LoG detail loss ----------
        log1 = self._log_response(img1)
        log2 = self._log_response(img2)
        logf = self._log_response(fused)

        # max(LoG) 用“幅值”更合理（虚焦时幅值明显变小）
        abs1 = torch.abs(log1)
        abs2 = torch.abs(log2)
        absf = torch.abs(logf)
        abs_max = torch.max(abs1, abs2)

        loss_detail = F.l1_loss(absf, abs_max)

        # ---------- 2) target_soft + SSIM loss ----------
        # 清晰度权重（像素级、逐通道）：w = abs1/(abs1+abs2)
        w = abs1 / (abs1 + abs2 + self.eps)
        target_soft = w * img1 + (1.0 - w) * img2

        ssim_val = self._ssim(fused, target_soft)  # 越大越好
        loss_ssim = 1.0 - ssim_val

        # ---------- 3) mask binarization regularizer ----------
        loss_mask_bin = torch.mean(mask * (1.0 - mask))

        # ---------- 4) mask TV smoothness ----------
        loss_mask_tv = self._tv_loss(mask)

        total_loss = (
            self.w_detail * loss_detail
            + self.w_ssim * loss_ssim
            + self.w_mask_bin * loss_mask_bin
            + self.w_mask_tv * loss_mask_tv
        )

        return total_loss, loss_detail, loss_ssim, loss_mask_bin, loss_mask_tv
