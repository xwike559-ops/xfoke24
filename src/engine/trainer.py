import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion,
                optimizer, device, num_epochs=10):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for img1, img2 in pbar:
            img1 = img1.to(device)
            img2 = img2.to(device)

            optimizer.zero_grad()

            fused, mask, _ = model(img1, img2)

            loss, l_detail, l_ssim, l_bin, l_tv = criterion(img1, img2, fused, mask)

            if torch.isnan(loss):
                raise RuntimeError("NaN detected in loss")

            loss.backward()

            # 梯度裁剪（Transformer 必须）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "detail": f"{l_detail.item():.4f}",
                "ssim": f"{l_ssim.item():.4f}",
                "bin": f"{l_bin.item():.4f}",
                "tv": f"{l_tv.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for img1, img2 in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                fused, mask, _ = model(img1, img2)
                loss, _, _, _,_ = criterion(img1, img2, fused, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

    return model