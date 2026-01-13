import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import torch.optim as optim
from src.models.hybrid_fusion_net import HybridFusionNet
from src.losses.stable_fusion_loss import StableFusionLoss
from src.losses.ImprovedFusionLoss import ImprovedFusionLoss
from src.data.multifocus_dataset import MultiFocusDataset
from src.engine.trainer import train_model
from src.engine.evaluator import test_model
from src.engine.evaluator import visualize_results

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")  # 强制指定为 CPU
    print(f"Using device: {device}")

    # 数据集路径 (请修改为你实际的路径)
    train_dir = "E:\\Multi-fusion_cnn-swinv2\\dataset\\train"
    val_dir = "E:\\Multi-fusion_cnn-swinv2\\dataset\\val"
    test_dir = "E:\\Multi-fusion_cnn-swinv2\\dataset\\test"

    # 注意：这里将 img_size 强制设为 512，适配 Swin Transformer 的 Patch/Window 计算
    try:
        train_dataset = MultiFocusDataset(train_dir, img_size=512)
        val_dataset = MultiFocusDataset(val_dir, img_size=512)
        test_dataset = MultiFocusDataset(test_dir, img_size=512)
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)  # 显存不够可减小 batch_size
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Samples - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化混合模型
    model = HybridFusionNet(img_size=512).to(device)

    # 打印参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Swin-Fusion Model Parameters: {params / 1e6:.2f}M")

    # 使用新的无监督损失函数
    # criterion = StableFusionLoss(
    #     w_grad=10.0,
    #     w_intensity=1.0,
    #     w_mask=0.1
    # ).to(device)

    criterion = ImprovedFusionLoss(
        w_detail=1.0,
        w_ssim=0.5,
        w_mask_bin=0.05,
        w_mask_tv=0.1,
        gaussian_ksize=5,
        gaussian_sigma=1.0,
        ssim_ksize=11,
        ssim_sigma=1.5,
    ).to(device)


    # 使用 AdamW 优化器，对 Transformer 更友好
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-5,  # 比你原来更安全
        weight_decay=1e-4
    )

    print("Starting training with Swin Transformer...")
    # 训练 100 个 Epoch 足够看到效果 (Demo用)，实际科研可跑更多
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100)

    torch.save(trained_model.state_dict(), "swin_fusion_final.pth")
    print("Model saved.")

    print("Testing model...")
    results = test_model(trained_model, test_loader, device)
    visualize_results(results)