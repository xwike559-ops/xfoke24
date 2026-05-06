import torch
import cv2
import numpy as np
from pathlib import Path


def test_model(model, test_loader, device, output_dir=None, max_save=5):
    model.eval()
    results = []
    samples_dir = Path(output_dir) if output_dir is not None else None
    if samples_dir is not None:
        samples_dir.mkdir(parents=True, exist_ok=True)

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

            if samples_dir is not None and i < max_save:
                cv2.imwrite(
                    str(samples_dir / f"pair_{i:03d}_fused.png"),
                    cv2.cvtColor(np_imgs[2], cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(str(samples_dir / f"pair_{i:03d}_mask.png"), mask_colormap)
    return results


def visualize_results(results, output_dir=None, max_items=3, show=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save overview figures.") from exc

    figures_dir = Path(output_dir) if output_dir is not None else None
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)

    for i, (img1, img2, fused, mask) in enumerate(results[:max_items]):
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
        if figures_dir is not None:
            plt.savefig(figures_dir / f"pair_{i:03d}_overview.png")
        if show:
            plt.show()
        plt.close()
