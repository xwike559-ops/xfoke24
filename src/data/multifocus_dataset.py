import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class MultiFocusDataset(Dataset):
    def __init__(self, root_dir, img_size=512):  # 修改默认尺寸为512以适配Swin
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_pairs = []

        print(f"Loading dataset from: {root_dir}")
        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset directory {root_dir} does not exist!")

        for pair_dir in os.listdir(root_dir):
            pair_path = os.path.join(root_dir, pair_dir)
            if os.path.isdir(pair_path):
                files = os.listdir(pair_path)
                a_files = [f for f in files if 'a' in f.lower()]
                b_files = [f for f in files if 'b' in f.lower()]

                for a_file in a_files:
                    for b_file in b_files:
                        img_a_path = os.path.join(pair_path, a_file)
                        img_b_path = os.path.join(pair_path, b_file)
                        self.image_pairs.append((img_a_path, img_b_path))

        # 全局配对回退策略
        if len(self.image_pairs) == 0:
            print("⚠️ Switching to global sequential pairing...")
            all_images = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                          if f.lower().endswith(('.jpg', '.png', '.bmp'))]
            all_images.sort()
            for i in range(0, len(all_images) - 1, 2):
                self.image_pairs.append((all_images[i], all_images[i + 1]))

        print(f"Total {len(self.image_pairs)} image pairs found")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_a_path, img_b_path = self.image_pairs[idx]
        try:
            img_a = cv2.imread(img_a_path)
            img_b = cv2.imread(img_b_path)

            # 强制转为RGB
            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
            img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

            # Resize到512 (适应Swin Transformer Window操作)
            img_a = cv2.resize(img_a, (self.img_size, self.img_size))
            img_b = cv2.resize(img_b, (self.img_size, self.img_size))

            img_a = img_a.astype(np.float32) / 255.0
            img_b = img_b.astype(np.float32) / 255.0

            img_a = torch.from_numpy(img_a).permute(2, 0, 1)
            img_b = torch.from_numpy(img_b).permute(2, 0, 1)
            return img_a, img_b
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.image_pairs))



