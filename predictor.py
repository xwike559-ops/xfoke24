import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.hybrid_fusion_net import HybridFusionNet


# -----------------------------
# Config
# -----------------------------
@dataclass
class PredictConfig:
    img_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True                   # 推理自动混合精度（GPU时更快更省显存）
    return_mask: bool = True
    clamp_output: bool = True          # 输出 clamp 到 [0,1]
    strict_load: bool = True           # load_state_dict 严格匹配
    in_channels: int = 3               # 默认RGB


# -----------------------------
# Utils
# -----------------------------
def _to_3ch_pil(img: Image.Image) -> Image.Image:
    """确保输入是 3 通道 RGB。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _save_tensor_image(t: torch.Tensor, path: str):
    """
    t: (1,C,H,W) or (C,H,W), range [0,1]
    """
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().clamp(0, 1)
    to_pil = transforms.ToPILImage()
    img = to_pil(t)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    img.save(path)


# -----------------------------
# Predictor
# -----------------------------
class SwinFusionPredictor:
    def __init__(self, weight_path: str, cfg: Optional[PredictConfig] = None):
        self.cfg = cfg or PredictConfig()
        self.device = torch.device(self.cfg.device)

        # 1) build model
        self.model = HybridFusionNet(img_size=self.cfg.img_size).to(self.device)
        self.model.eval()

        # 2) load weights
        ckpt = torch.load(weight_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        missing, unexpected = self.model.load_state_dict(ckpt, strict=self.cfg.strict_load)
        if (missing or unexpected) and self.cfg.strict_load is False:
            print(f"[Warn] Missing keys: {missing}")
            print(f"[Warn] Unexpected keys: {unexpected}")

        # 3) preprocess
        self.tf = transforms.Compose([
            transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
            transforms.ToTensor(),  # PIL -> [0,1], (C,H,W)
        ])

    @torch.no_grad()
    def predict_tensor(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        img1/img2: (B,C,H,W) float in [0,1]
        returns: fused, mask, mask_logits
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # 若尺寸不是 cfg.img_size，可自动插值（保持拓展性）
        if img1.shape[-2:] != (self.cfg.img_size, self.cfg.img_size):
            img1 = F.interpolate(img1, size=(self.cfg.img_size, self.cfg.img_size),
                                 mode="bilinear", align_corners=False)
        if img2.shape[-2:] != (self.cfg.img_size, self.cfg.img_size):
            img2 = F.interpolate(img2, size=(self.cfg.img_size, self.cfg.img_size),
                                 mode="bilinear", align_corners=False)

        use_amp = self.cfg.amp and (self.device.type == "cuda")
        autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast

        with autocast_ctx(enabled=use_amp):
            out = self.model(img1, img2)

        # 兼容：有的模型只返回 fused, mask
        if isinstance(out, (tuple, list)):
            if len(out) == 3:
                fused, mask, mask_logits = out
            elif len(out) == 2:
                fused, mask = out
                mask_logits = None
            else:
                fused = out[0]
                mask, mask_logits = None, None
        else:
            fused, mask, mask_logits = out, None, None

        if self.cfg.clamp_output:
            fused = fused.clamp(0, 1)

        if not self.cfg.return_mask:
            mask, mask_logits = None, None

        return fused, mask, mask_logits

    @torch.no_grad()
    def predict_pil(
        self,
        img1_pil: Image.Image,
        img2_pil: Image.Image,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        img1_pil = _to_3ch_pil(img1_pil)
        img2_pil = _to_3ch_pil(img2_pil)

        t1 = self.tf(img1_pil).unsqueeze(0)  # (1,C,H,W)
        t2 = self.tf(img2_pil).unsqueeze(0)
        return self.predict_tensor(t1, t2)

    def predict_files(
        self,
        img1_path: str,
        img2_path: str,
        save_fused_path: Optional[str] = None,
        save_mask_path: Optional[str] = None,
    ):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        fused, mask, _ = self.predict_pil(img1, img2)

        if save_fused_path:
            _save_tensor_image(fused, save_fused_path)

        if save_mask_path and mask is not None:
            # mask: (1,1,H,W) -> 保存成单通道灰度图
            m = mask.detach().cpu().clamp(0, 1)
            if m.dim() == 4:
                m = m[0]  # (1,H,W)
            # 转成 3 通道也可以，这里保留单通道
            os.makedirs(os.path.dirname(save_mask_path), exist_ok=True) if os.path.dirname(save_mask_path) else None
            transforms.ToPILImage()(m).save(save_mask_path)

        return fused, mask

    def export_onnx(
        self,
        onnx_path: str,
        opset: int = 17,
        dynamic: bool = True,
    ):
        """
        导出 ONNX（先给一版可用的，后续再做更深的 onnx 兼容优化）
        注意：Swin/WindowAttention 之类模块可能需要额外处理，先试导出，
        不行的话再针对报错点逐个改写。
        """
        self.model.eval()

        dummy = torch.randn(1, self.cfg.in_channels, self.cfg.img_size, self.cfg.img_size, device=self.device)
        input_names = ["img1", "img2"]
        output_names = ["fused", "mask", "mask_logits"]

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "img1": {0: "batch", 2: "height", 3: "width"},
                "img2": {0: "batch", 2: "height", 3: "width"},
                "fused": {0: "batch", 2: "height", 3: "width"},
            }

        torch.onnx.export(
            self.model,
            (dummy, dummy),
            onnx_path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        print(f"[OK] ONNX exported to: {onnx_path}")


# -----------------------------
# CLI demo
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="swin_fusion_final.pth")
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--out_fused", type=str, default="out/fused.png")
    parser.add_argument("--out_mask", type=str, default="out/mask.png")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = PredictConfig()
    if args.device is not None:
        cfg.device = args.device

    predictor = SwinFusionPredictor(args.weights, cfg)
    predictor.predict_files(args.img1, args.img2, args.out_fused, args.out_mask)
    print("Done.")
