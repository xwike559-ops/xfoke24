
"""
Semantic-guided Difference Prior (M) for Multi-focus Fusion
----------------------------------------------------------
目标：生成 1 通道权重图 M，使其在“近焦/远焦显著差异区域”更亮，并用语义分割先验抑制噪声差异、强调结构/边界。

输出可视化：
1) Near / Far 原图
2) 差异图 D（基于 Laplacian 或 Gradient）
3) 语义先验图 S_conf（来自 DeepLabv3 的 top1 概率）
4) 语义边界图 S_edge（可选）
5) 最终 M 热力图 + 与 Near/Far 的叠加图
6) M 直方图（分布）

用法：
- 修改 near_path / far_path
- 运行脚本，会在同目录保存 M_visualization.png
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


# -------------------------
# 设备
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# DeepLabv3 预训练模型
# -------------------------
weights = DeepLabV3_ResNet50_Weights.DEFAULT
seg_model = deeplabv3_resnet50(weights=weights).to(device).eval()
preprocess = weights.transforms()  # 推荐预处理（包含 resize/normalize 等）


# -------------------------
# 工具函数：读图 -> tensor（0~1）
# -------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor in [0,1], shape [1,3,H,W]"""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: [1,3,H,W] -> [1,1,H,W]"""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def normalize01(m: torch.Tensor) -> torch.Tensor:
    """Normalize per-image to [0,1], m: [1,1,H,W]"""
    m = m - m.amin(dim=(-2, -1), keepdim=True)
    m = m / (m.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return m


def laplacian_mag(gray: torch.Tensor) -> torch.Tensor:
    """Laplacian magnitude, gray: [1,1,H,W] -> [1,1,H,W]"""
    k = torch.tensor([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    y = F.conv2d(gray, k, padding=1)
    return y.abs()


def grad_mag(gray: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude, gray: [1,1,H,W] -> [1,1,H,W]"""
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


# -------------------------
# 语义先验：跑 DeepLabv3 得到 prob
# -------------------------
@torch.no_grad()
def get_semantic_prob_from_pil(pil_img: Image.Image, out_hw=None) -> torch.Tensor:
    """
    返回 prob: [1,C,H,W]（softmax 概率图）
    out_hw: (H,W) 可选，强制插值到某分辨率
    """
    x = preprocess(pil_img).unsqueeze(0).to(device)  # [1,3,h,w] 已 normalize
    logits = seg_model(x)["out"]                      # [1,C,h,w]
    prob = logits.softmax(dim=1)                      # [1,C,h,w]
    if out_hw is not None:
        prob = F.interpolate(prob, size=out_hw, mode="bilinear", align_corners=False)
    return prob


# -------------------------
# 核心：构造“语义引导差异” M
# -------------------------
@torch.no_grad()
def build_semantic_guided_diff_M(
        near_rgb01: torch.Tensor,
        far_rgb01: torch.Tensor,
        prob_semantic: torch.Tensor,
        diff_mode="lap",          # "lap" or "grad"
        semantic_mode="edge",     # "edge" or "gate"
        lam=2.0,                  # 语义边界强化强度
        gamma=1.0                 # 对最终 M 做幂次增强（>1 更聚焦差异区）
):
    """
    near_rgb01/far_rgb01: [1,3,H,W] in [0,1]
    prob_semantic:        [1,C,H,W] (与 H,W 对齐)
    返回:
      D      [1,1,H,W] 差异图
      S_conf [1,1,H,W] 语义置信图(top1 prob)
      S_edge [1,1,H,W] 语义边界图(可选，若 semantic_mode="edge" 才有效)
      M      [1,1,H,W] 最终语义引导差异权重
    """
    H, W = near_rgb01.shape[-2:]

    # 1) 差异图 D：在“近/远焦明显不同”的地方亮
    gn = to_gray(near_rgb01.to(device))
    gf = to_gray(far_rgb01.to(device))

    if diff_mode == "lap":
        D = (laplacian_mag(gn) - laplacian_mag(gf)).abs()
    elif diff_mode == "grad":
        D = (grad_mag(gn) - grad_mag(gf)).abs()
    else:
        raise ValueError("diff_mode must be 'lap' or 'grad'")
    D = normalize01(D)

    # 2) 语义置信图 S_conf：不是“前景”，而是“语义结构的可靠性”
    #    这里用于抑制无意义噪声差异 & 提示结构区域
    S_conf, _ = prob_semantic.max(dim=1, keepdim=True)  # [1,1,H,W] in [0,1]

    # 3) 语义引导：两种常用方式
    S_edge = torch.zeros_like(S_conf)
    if semantic_mode == "edge":
        # 边界强化：更贴合“差异区域往往出现在结构边界/细节切换处”
        S_edge = grad_mag(S_conf)
        S_edge = normalize01(S_edge)
        M = D * (1.0 + lam * S_edge)
    elif semantic_mode == "gate":
        # 门控：只让语义更可靠的地方的差异更重要
        M = D * S_conf
    else:
        raise ValueError("semantic_mode must be 'edge' or 'gate'")

    M = normalize01(M)

    # 4) 可选：幂次增强，让差异区域更聚焦
    if gamma != 1.0:
        M = M.clamp(0, 1).pow(gamma)

    return D, S_conf, S_edge, M


# -------------------------
# 可视化：热力图 + 叠加 + 直方图
# -------------------------
def _tensor_to_hw(t: torch.Tensor) -> np.ndarray:
    """[1,1,H,W] -> [H,W] numpy"""
    a = t.detach().cpu().numpy()
    return a[0, 0]


def _pil_to_np(pil_img: Image.Image, out_hw=None) -> np.ndarray:
    img = pil_img
    if out_hw is not None:
        img = img.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    return np.array(img)


def visualize_all(
        near_pil: Image.Image,
        far_pil: Image.Image,
        D: torch.Tensor,
        S_conf: torch.Tensor,
        S_edge: torch.Tensor,
        M: torch.Tensor,
        save_path=None,
        overlay_alpha=0.45,
        bins=80
):
    H, W = D.shape[-2:]
    near_np = _pil_to_np(near_pil, (H, W))
    far_np = _pil_to_np(far_pil, (H, W))

    D_np = _tensor_to_hw(D)
    Sconf_np = _tensor_to_hw(S_conf)
    Sedge_np = _tensor_to_hw(S_edge)
    M_np = _tensor_to_hw(M)

    fig = plt.figure(figsize=(16, 12))

    # 1) Near / Far
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(near_np)
    ax1.set_title("Near-focus (resized)")
    ax1.axis("off")

    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(far_np)
    ax2.set_title("Far-focus (resized)")
    ax2.axis("off")

    # 2) D
    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.imshow(D_np, cmap="jet", vmin=0, vmax=1)
    ax3.set_title("D: Focus Difference (0~1)")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 3) S_conf
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(Sconf_np, cmap="jet", vmin=0, vmax=1)
    ax4.set_title("S_conf: Semantic Confidence (top1 prob)")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 4) S_edge
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(Sedge_np, cmap="jet", vmin=0, vmax=1)
    ax5.set_title("S_edge: Semantic Boundary Strength")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # 5) M
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.imshow(M_np, cmap="jet", vmin=0, vmax=1)
    ax6.set_title(f"M: Semantic-guided Difference Prior\nmin={M.min().item():.4f}, max={M.max().item():.4f}, mean={M.mean().item():.4f}")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # 6) Overlay on Near
    ax7 = plt.subplot(3, 3, 7)
    ax7.imshow(near_np)
    ax7.imshow(M_np, cmap="jet", vmin=0, vmax=1, alpha=overlay_alpha)
    ax7.set_title(f"Overlay on Near (alpha={overlay_alpha})")
    ax7.axis("off")

    # 7) Overlay on Far
    ax8 = plt.subplot(3, 3, 8)
    ax8.imshow(far_np)
    ax8.imshow(M_np, cmap="jet", vmin=0, vmax=1, alpha=overlay_alpha)
    ax8.set_title(f"Overlay on Far (alpha={overlay_alpha})")
    ax8.axis("off")

    # 8) Histogram
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(M_np.flatten(), bins=bins)
    ax9.set_title(f"Histogram of M (bins={bins})")
    ax9.set_xlabel("M value (0~1)")
    ax9.set_ylabel("Pixel count")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    plt.show()


# -------------------------
# 主程序：请改成你的路径
# -------------------------
if __name__ == "__main__":
    # 你需要提供一对近焦/远焦图像
    near_path = r"E:\桌面\Scientific Study Road\Z-STACK\LytroDataset\LytroDataset\LytroDataset\lytro-02-A.jpg"
    far_path  = r"E:\桌面\Scientific Study Road\Z-STACK\LytroDataset\LytroDataset\LytroDataset\lytro-02-B.jpg"

    out_hw = (512, 512)           # 统一分辨率，便于可视化/后续窗口划分
    diff_mode = "lap"             # "lap" 更像“清晰度差”，一般更适合多焦距
    semantic_mode = "edge"        # "edge"（更聚焦边界差异） or "gate"（语义门控）
    lam = 2.0                     # 边界强化强度，建议 1~4
    gamma = 1.5                   # M 幂次增强，建议 1~3（先从 1.5 试）

    near_pil = Image.open(near_path).convert("RGB")
    far_pil = Image.open(far_path).convert("RGB")

    # 1) 构造一个“语义更稳定”的代理图来跑分割（推荐：用 near 或 far 的任意一张也可以）
    #    简化起见，这里直接对 near 跑语义；你也可以改成对 proxy/两张取mean/max
    prob = get_semantic_prob_from_pil(near_pil, out_hw=out_hw)  # [1,C,H,W]

    # 2) 近/远图 -> 0~1 tensor，并 resize 到 out_hw（保持一致）
    near_resized = near_pil.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    far_resized = far_pil.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    near_t = pil_to_tensor01(near_resized).to(device)
    far_t = pil_to_tensor01(far_resized).to(device)

    # 3) 构造 D, S_conf, S_edge, M
    D, S_conf, S_edge, M = build_semantic_guided_diff_M(
        near_rgb01=near_t,
        far_rgb01=far_t,
        prob_semantic=prob,
        diff_mode=diff_mode,
        semantic_mode=semantic_mode,
        lam=lam,
        gamma=gamma
    )

    print("M stats:", M.shape,
          "min=", M.min().item(),
          "max=", M.max().item(),
          "mean=", M.mean().item(),
          "std=", M.std().item())

    # 4) 可视化并保存
    save_path = os.path.join(os.path.dirname(near_path), "M_visualization.png")
    visualize_all(
        near_pil=near_pil,
        far_pil=far_pil,
        D=D,
        S_conf=S_conf,
        S_edge=S_edge,
        M=M,
        save_path=save_path,
        overlay_alpha=0.45,
        bins=80
    )
