import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
                cv2.imwrite(f'result_100_losschanged_v2_epoch{i}.png', cv2.cvtColor(np_imgs[2], cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'resultmask_100_losschanged_v2_epoch{i}.png', mask_colormap)
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
