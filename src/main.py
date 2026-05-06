import argparse
import random
import sys
from pathlib import Path

# 允许两种运行方式都能找到项目包：
# 1) python start.py
# 2) python src/main.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args():
    """集中管理训练参数，方便每次调模型时直接从命令行覆盖。"""
    parser = argparse.ArgumentParser(description="Train and evaluate Swin fusion experiments.")

    # 实验命名与输出根目录。
    # 每次运行会自动创建 experiments/runs/<时间戳-实验名>/，不用再手动改图片名区分版本。
    parser.add_argument("--experiment-name", type=str, default="swin_fusion")
    parser.add_argument("--output-root", type=str, default="experiments/runs")

    # 数据集目录，目录结构默认是 dataset/train、dataset/val、dataset/test。
    # 如果换数据集，只需要在命令行传入新的路径。
    parser.add_argument("--train-dir", type=str, default="dataset/train")
    parser.add_argument("--val-dir", type=str, default="dataset/val")
    parser.add_argument("--test-dir", type=str, default="dataset/test")

    # 基础训练参数。
    # img-size 需要和 Swin 窗口/patch 设置兼容；batch-size 受显存影响最大。
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # 优化器参数。当前使用 AdamW，调参时最常改 lr 和 weight-decay。
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # device 留空时自动优先用 cuda；也可以手动传 --device cpu 或 --device cuda:0。
    parser.add_argument("--device", type=str, default=None)

    # 损失函数与权重。
    # improved 是当前主线版本；stable 保留为对照实验。
    parser.add_argument("--loss", choices=("improved", "stable"), default="improved")
    # ImprovedFusionLoss 参数：细节、结构相似性、mask 二值化、mask 平滑。
    parser.add_argument("--w-detail", type=float, default=2.0)
    parser.add_argument("--w-ssim", type=float, default=0.5)
    parser.add_argument("--w-mask-bin", type=float, default=0.05)
    parser.add_argument("--w-mask-tv", type=float, default=0.02)
    # StableFusionLoss 参数：梯度、强度、mask 正则。
    parser.add_argument("--w-grad", type=float, default=10.0)
    parser.add_argument("--w-intensity", type=float, default=1.0)
    parser.add_argument("--w-mask", type=float, default=0.1)

    # 测试阶段保存多少张单独结果图和总览图。
    # fused/mask 会进入 samples/，四宫格预览会进入 figures/。
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-figures", type=int, default=3)
    # 服务器或远程训练时通常不需要弹窗，默认只保存图片。
    parser.add_argument("--show-figures", action="store_true")

    return parser.parse_args()


def build_criterion(args, device):
    """根据 --loss 选择损失函数，并把命令行权重填进去。"""
    # 延迟导入是为了让 python start.py --help 不依赖完整训练环境。
    from src.losses.ImprovedFusionLoss import ImprovedFusionLoss
    from src.losses.stable_fusion_loss import StableFusionLoss

    if args.loss == "stable":
        return StableFusionLoss(
            w_grad=args.w_grad,
            w_intensity=args.w_intensity,
            w_mask=args.w_mask,
        ).to(device)

    return ImprovedFusionLoss(
        w_detail=args.w_detail,
        w_ssim=args.w_ssim,
        w_mask_bin=args.w_mask_bin,
        w_mask_tv=args.w_mask_tv,
        gaussian_ksize=5,
        gaussian_sigma=1.0,
        ssim_ksize=11,
        ssim_sigma=1.5,
    ).to(device)


def main():
    args = parse_args()

    # 训练相关依赖放在 main 内部导入：
    # 看 --help 时不需要加载 torch/timm，也方便在轻量环境中检查参数说明。
    import numpy as np
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from src.data.multifocus_dataset import MultiFocusDataset
    from src.engine.evaluator import test_model, visualize_results
    from src.engine.trainer import train_model
    from src.models.hybrid_fusion_net import HybridFusionNet
    from src.utils.experiment import make_run_dir, module_hparams, write_run_metadata

    # 固定随机种子，方便同一组参数下复现实验。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 自动选择训练设备；命令行传 --device 时优先使用手动指定值。
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 创建本次实验目录：
    # run_meta.json      保存参数记录
    # checkpoints/       保存模型权重
    # samples/           保存 fused/mask 单图
    # figures/           保存输入-输出-mask 总览图
    run_dir = make_run_dir(args.output_root, args.experiment_name)
    print(f"Experiment run: {run_dir}")

    # 读取 train/val/test 数据；路径错误时直接提示并退出。
    try:
        train_dataset = MultiFocusDataset(args.train_dir, img_size=args.img_size)
        val_dataset = MultiFocusDataset(args.val_dir, img_size=args.img_size)
        test_dataset = MultiFocusDataset(args.test_dir, img_size=args.img_size)
    except Exception as exc:
        print(f"Dataset Error: {exc}")
        return

    # DataLoader 负责把图片对组成 batch。
    # 测试集 batch 固定为 1，便于逐对保存结果图。
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Samples - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化融合模型，并记录可训练参数量，后续比较模型版本时很有用。
    model = HybridFusionNet(img_size=args.img_size).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Swin-Fusion Model Parameters: {params / 1e6:.2f}M")

    # 构建损失函数和优化器。
    criterion = build_criterion(args, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    checkpoint_path = Path(run_dir) / "checkpoints" / "swin_fusion_final.pth"

    # 在训练开始前先写入本次实验的完整配置。
    # 这样即使训练中断，也能知道这次 run 用了哪些参数。
    write_run_metadata(
        run_dir,
        {
            "experiment_name": args.experiment_name,
            "seed": args.seed,
            "device": str(device),
            "datasets": {
                "train": args.train_dir,
                "val": args.val_dir,
                "test": args.test_dir,
            },
            "training": {
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "num_workers": args.num_workers,
            },
            "model": {
                "name": model.__class__.__name__,
                "trainable_params": params,
            },
            "loss": {
                "name": criterion.__class__.__name__,
                "hparams": module_hparams(criterion),
            },
            "outputs": {
                "checkpoint": str(checkpoint_path),
                "samples_dir": str(Path(run_dir) / "samples"),
                "figures_dir": str(Path(run_dir) / "figures"),
            },
        },
    )

    # 训练阶段：train_model 内部负责每个 epoch 的训练与验证 loss 打印。
    print("Starting training with Swin Transformer...")
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=args.epochs,
    )

    # 模型权重保存在本次 run 的 checkpoints/，避免多个实验互相覆盖。
    torch.save(trained_model.state_dict(), checkpoint_path)
    print(f"Model saved: {checkpoint_path}")

    # 测试阶段：保存单独 fused/mask 图，以及更方便人工观察的四宫格总览图。
    print("Testing model...")
    results = test_model(
        trained_model,
        test_loader,
        device,
        output_dir=Path(run_dir) / "samples",
        max_save=args.max_samples,
    )
    visualize_results(
        results,
        output_dir=Path(run_dir) / "figures",
        max_items=args.max_figures,
        show=args.show_figures,
    )
    print(f"Results saved under: {run_dir}")


if __name__ == "__main__":
    main()
