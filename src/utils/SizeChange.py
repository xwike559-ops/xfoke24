import os
from PIL import Image
from pathlib import Path


def resize_images_in_folders(root_path, target_size=(520, 520)):
    """
    递归遍历路径下的所有文件夹，将图片调整为指定大小

    Args:
        root_path: 根目录路径
        target_size: 目标图片大小，默认为(512, 512)
    """
    root_path = Path(root_path)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

    # 统计信息
    processed_count = 0
    failed_count = 0

    # 递归遍历所有文件夹
    for folder_path, _, files in os.walk(root_path):
        print(f"处理文件夹: {folder_path}")

        for filename in files:
            file_path = Path(folder_path) / filename
            file_ext = file_path.suffix.lower()

            # 检查是否为图片文件
            if file_ext in image_extensions:
                try:
                    # 打开图片
                    with Image.open(file_path) as img:
                        # 检查图片是否需要调整
                        if img.size != target_size:
                            print(f"  调整: {filename} ({img.size} -> {target_size})")

                            # 调整图片大小
                            # 使用LANCZOS重采样滤波器（高质量）
                            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

                            # 保持原始图片的模式（如RGB、RGBA等）
                            if img.mode != resized_img.mode:
                                resized_img = resized_img.convert(img.mode)

                            # 保存图片（覆盖原文件）
                            resized_img.save(file_path, quality=95)
                            processed_count += 1
                        else:
                            print(f"  跳过: {filename} (已是目标大小)")

                except Exception as e:
                    print(f"  错误处理 {filename}: {e}")
                    failed_count += 1

    print("\n" + "=" * 50)
    print(f"处理完成!")
    print(f"成功处理: {processed_count} 张图片")
    print(f"处理失败: {failed_count} 张图片")

def main():
    # 设置你的根目录路径
    root_directory = r"C:\Users\14875\Desktop\imgs\1"

    # 目标图片大小
    target_size = (520, 520)

    # 确认操作
    print(f"将要处理路径下的所有图片: {root_directory}")
    print(f"目标大小: {target_size}")
    print("此操作将覆盖原始图片文件，请确保已备份！")

    confirm = input("是否继续? (输入 'yes' 继续): ")

    if confirm.lower() == 'yes':
        # 检查路径是否存在
        if not os.path.exists(root_directory):
            print(f"错误: 路径不存在 - {root_directory}")
            return

        # 处理图片
        resize_images_in_folders(root_directory, target_size)
    else:
        print("操作已取消")

if __name__ == "__main__":
    main()