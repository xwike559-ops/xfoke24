"""
Create two blurred versions (A/B) for each COCO image by applying Gaussian blur
to randomly selected regions.

Outputs:
  output_dir/
    000001-A.jpg
    000001-B.jpg
    000002-A.jpg
    000002-B.jpg
    ...

Features:
- Blur region can be random rectangles (optionally multiple per image).
- Two modes for A/B masks:
  1) independent: A and B regions are generated independently (can overlap).
  2) complementary: B region is generated from the non-A area (disjoint by design),
     and can be full complement or a random subset of the complement.
- Regions do NOT need to cover half the image.

Usage example:
  python make_multifocus_from_coco.py \
      --input_dir "E:/coco/train2017" \
      --output_dir "E:/coco_blur_pairs/train" \
      --mode complementary \
      --start_index 1 \
      --max_images 2000 \
      --blur_ksize 31 \
      --sigma 8.0

Requires: opencv-python, numpy
"""
import os
import cv2
import glob
import math
import argparse
import numpy as np
from typing import Tuple, List


def list_images(input_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    paths = sorted(paths)
    return paths


def ensure_odd(k: int) -> int:
    k = int(k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k


def make_random_rect_mask(
    h: int,
    w: int,
    num_rects: int,
    area_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a binary mask with several random rectangles.
    area_range: fraction of image area per rectangle, e.g. (0.05, 0.25)
    aspect_range: rectangle aspect ratio range (w/h), e.g. (0.5, 2.0)
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    img_area = h * w

    for _ in range(num_rects):
        target_area = rng.uniform(area_range[0], area_range[1]) * img_area
        aspect = rng.uniform(aspect_range[0], aspect_range[1])

        rect_w = int(round(math.sqrt(target_area * aspect)))
        rect_h = int(round(math.sqrt(target_area / aspect)))

        rect_w = max(8, min(rect_w, w))
        rect_h = max(8, min(rect_h, h))

        x1 = int(rng.integers(0, max(1, w - rect_w + 1)))
        y1 = int(rng.integers(0, max(1, h - rect_h + 1)))
        x2 = x1 + rect_w
        y2 = y1 + rect_h

        mask[y1:y2, x1:x2] = 1

    return mask


def apply_blur_on_mask(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    ksize: int,
    sigma: float,
) -> np.ndarray:
    """
    Blur only masked region. Outside mask stays original.
    mask01: 0/1 uint8 array (H,W)
    """
    ksize = ensure_odd(ksize)
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    mask = mask01.astype(np.float32)[..., None]  # (H,W,1)
    out = img_bgr.astype(np.float32) * (1.0 - mask) + blurred.astype(np.float32) * mask
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def subset_of_mask(mask01: np.ndarray, keep_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly keep only a portion of a binary mask. keep_ratio in (0,1].
    """
    if keep_ratio >= 0.999:
        return mask01.copy()

    ys, xs = np.where(mask01 > 0)
    if len(ys) == 0:
        return mask01.copy()

    n_keep = max(1, int(round(len(ys) * keep_ratio)))
    idx = rng.permutation(len(ys))[:n_keep]
    out = np.zeros_like(mask01)
    out[ys[idx], xs[idx]] = 1
    return out


def build_masks(
    h: int,
    w: int,
    mode: str,
    num_rects_a: int,
    num_rects_b: int,
    area_range_a: Tuple[float, float],
    area_range_b: Tuple[float, float],
    aspect_range: Tuple[float, float],
    complement_subset_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns maskA, maskB (0/1 uint8).
    mode:
      - independent: A and B are independently generated (can overlap).
      - complementary: B is generated from non-A area (disjoint).
                       If complement_subset_ratio<1, B will be a random subset of complement.
    """
    maskA = make_random_rect_mask(h, w, num_rects_a, area_range_a, aspect_range, rng)

    if mode == "independent":
        maskB = make_random_rect_mask(h, w, num_rects_b, area_range_b, aspect_range, rng)
        return maskA, maskB

    if mode == "complementary":
        complement = (1 - maskA).astype(np.uint8)
        # Option 1: full complement (blur everything outside A) -> set complement_subset_ratio=1.0
        # Option 2: subset of complement (not necessarily half / not necessarily full)
        maskB = subset_of_mask(complement, complement_subset_ratio, rng)

        # Make B "more region-like": optionally expand kept pixels a bit using dilation
        # (helps avoid speckle if subset_ratio is small)
        if complement_subset_ratio < 0.95:
            k = 3
            kernel = np.ones((k, k), np.uint8)
            maskB = cv2.dilate(maskB, kernel, iterations=1)
            maskB = (maskB > 0).astype(np.uint8)

        # Ensure disjointness:
        maskB = (maskB * (1 - maskA)).astype(np.uint8)
        return maskA, maskB

    raise ValueError(f"Unknown mode: {mode}. Use 'independent' or 'complementary'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="COCO images folder, e.g. train2017/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for numbered A/B images")
    parser.add_argument("--mode", type=str, default="complementary",
                        choices=["independent", "complementary"],
                        help="How to generate A/B blur regions")
    parser.add_argument("--start_index", type=int, default=1, help="Start numbering from this index")
    parser.add_argument("--max_images", type=int, default=0, help="0 means process all images")
    parser.add_argument("--seed", type=int, default=42)

    # Blur settings
    parser.add_argument("--blur_ksize", type=int, default=31, help="Gaussian kernel size (odd)")
    parser.add_argument("--sigma", type=float, default=8.0, help="Gaussian sigma")

    # Region settings
    parser.add_argument("--num_rects_a", type=int, default=1, help="Number of rectangles for mask A")
    parser.add_argument("--num_rects_b", type=int, default=1, help="Number of rectangles for mask B (independent mode)")
    parser.add_argument("--area_min_a", type=float, default=0.08, help="Min rectangle area fraction for A")
    parser.add_argument("--area_max_a", type=float, default=0.30, help="Max rectangle area fraction for A")
    parser.add_argument("--area_min_b", type=float, default=0.08, help="Min rectangle area fraction for B")
    parser.add_argument("--area_max_b", type=float, default=0.30, help="Max rectangle area fraction for B")
    parser.add_argument("--aspect_min", type=float, default=0.5, help="Min aspect ratio (w/h)")
    parser.add_argument("--aspect_max", type=float, default=2.0, help="Max aspect ratio (w/h)")

    # Complementary mode control
    parser.add_argument("--complement_subset_ratio", type=float, default=0.60,
                        help="In complementary mode, blur only this fraction of the complement area. "
                             "1.0 means blur the full complement.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    img_paths = list_images(args.input_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    if args.max_images and args.max_images > 0:
        img_paths = img_paths[:args.max_images]

    idx = args.start_index

    for p in img_paths:
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[Skip] failed to read: {p}")
            continue

        h, w = img_bgr.shape[:2]

        maskA, maskB = build_masks(
            h=h, w=w,
            mode=args.mode,
            num_rects_a=args.num_rects_a,
            num_rects_b=args.num_rects_b,
            area_range_a=(args.area_min_a, args.area_max_a),
            area_range_b=(args.area_min_b, args.area_max_b),
            aspect_range=(args.aspect_min, args.aspect_max),
            complement_subset_ratio=args.complement_subset_ratio,
            rng=rng,
        )

        imgA = apply_blur_on_mask(img_bgr, maskA, args.blur_ksize, args.sigma)
        imgB = apply_blur_on_mask(img_bgr, maskB, args.blur_ksize, args.sigma)

        pair_dir = os.path.join(args.output_dir, f"pair{idx}")
        os.makedirs(pair_dir, exist_ok=True)

        outA = os.path.join(pair_dir, "A.jpg")
        outB = os.path.join(pair_dir, "B.jpg")

        cv2.imwrite(outA, imgA)
        cv2.imwrite(outB, imgB)

        idx += 1

    print(f"Done. Wrote pairs: {idx - args.start_index} to {args.output_dir}")


if __name__ == "__main__":
    import sys

    sys.argv = [
        "add_blur.py",
        "--input_dir", r"E:\test",
        "--output_dir", r"E:\output",
        "--mode", "complementary"
    ]
    main()


#改进：
#让模糊区域由 COCO 标注（人/物体）控制，而不是随机矩形