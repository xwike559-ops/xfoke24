"""
Create two blurred versions (A/B) for each image by applying Gaussian blur
to randomly selected regions.

Outputs:
  output_dir/
    pair1/A.jpg
    pair1/B.jpg
    pair2/A.jpg
    pair2/B.jpg
    ...

Constraints (updated):
- Each image generates ONE blur region for A, ONE blur region for B.
- A/B blur regions should overlap as little as possible (near-disjoint).
- Blur region area fraction is in [1/5, 1/2] of image area (0.20~0.50).
- Region shape is natural/irregular (not strict rectangles) to avoid shortcut learning.
- Soft mask blending to avoid artificial hard edges.

Requires: opencv-python, numpy
"""
import os
import cv2
import glob
import math
import argparse
import numpy as np
from typing import Tuple, List


# ------------------------
# Basic helpers
# ------------------------
def list_images(input_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return sorted(paths)


def ensure_odd(k: int) -> int:
    k = int(k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def mask_area_frac(mask01: np.ndarray) -> float:
    return float(mask01.mean())


def overlap_ratio(maskA: np.ndarray, maskB: np.ndarray) -> float:
    """Overlap measured as intersection / area(A) (you can swap to IoU if you prefer)."""
    inter = float((maskA & maskB).sum())
    a = float(maskA.sum() + 1e-6)
    return inter / a


def iou(maskA: np.ndarray, maskB: np.ndarray) -> float:
    inter = float((maskA & maskB).sum())
    union = float((maskA | maskB).sum() + 1e-6)
    return inter / union


# ------------------------
# Natural single-blob mask
# ------------------------
def make_random_blob_mask(
    h: int,
    w: int,
    area_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create ONE natural/irregular blob mask:
    - ellipse / polygon / stroke (chosen randomly)
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    img_area = h * w
    a_min, a_max = area_range

    # sample target area
    target_area = rng.uniform(a_min, a_max) * img_area
    aspect = rng.uniform(aspect_range[0], aspect_range[1])

    bw = int(round(math.sqrt(target_area * aspect)))
    bh = int(round(math.sqrt(target_area / aspect)))
    bw = max(32, min(bw, w))
    bh = max(32, min(bh, h))

    cx = int(rng.integers(0, w))
    cy = int(rng.integers(0, h))

    shape_type = rng.choice(["ellipse", "poly", "stroke"], p=[0.55, 0.35, 0.10])

    if shape_type == "ellipse":
        ax = max(16, bw // 2)
        ay = max(16, bh // 2)
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 1, -1)

    elif shape_type == "poly":
        n = int(rng.integers(7, 12))
        pts = []
        for i in range(n):
            ang = 2 * math.pi * (i / n) + rng.uniform(-0.35, 0.35)
            rx = rng.uniform(0.35, 0.65) * bw
            ry = rng.uniform(0.35, 0.65) * bh
            x = int(np.clip(cx + rx * math.cos(ang), 0, w - 1))
            y = int(np.clip(cy + ry * math.sin(ang), 0, h - 1))
            pts.append([x, y])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)

    else:  # stroke
        thickness = int(rng.integers(14, max(18, min(bw, bh)//3)))
        nseg = int(rng.integers(3, 6))
        x, y = cx, cy
        for _ in range(nseg):
            dx = int(rng.integers(-bw//2, bw//2))
            dy = int(rng.integers(-bh//2, bh//2))
            x2 = int(np.clip(x + dx, 0, w - 1))
            y2 = int(np.clip(y + dy, 0, h - 1))
            cv2.line(mask, (x, y), (x2, y2), 1, thickness=thickness)
            x, y = x2, y2

    # close to make it coherent
    k = int(rng.integers(7, 15))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return (mask > 0).astype(np.uint8)


def force_area_fraction(mask01: np.ndarray, target_range: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """
    Adjust mask area into target_range by mild dilation/erosion.
    Keeps it as a single coherent blob.
    """
    h, w = mask01.shape
    tmin, tmax = target_range
    m = (mask01 > 0).astype(np.uint8)

    for _ in range(25):
        cur = float(m.mean())
        if tmin <= cur <= tmax:
            break
        if cur < tmin:
            k = int(rng.integers(9, 17))
            kernel = np.ones((k, k), np.uint8)
            m = cv2.dilate(m, kernel, iterations=1)
        else:
            k = int(rng.integers(7, 15))
            kernel = np.ones((k, k), np.uint8)
            m = cv2.erode(m, kernel, iterations=1)
        m = (m > 0).astype(np.uint8)

    # ensure not empty
    if m.sum() == 0:
        m = mask01.copy().astype(np.uint8)
    return m


# ------------------------
# Generate B inside complement (low-overlap)
# ------------------------
def make_blob_inside_allowed(
    allowed01: np.ndarray,
    area_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
    rng: np.random.Generator,
    max_tries: int = 30,
) -> np.ndarray:
    """
    Generate one blob mask that lies mostly inside allowed01.
    We generate candidate blobs on full image, then intersect with allowed, and
    keep it if area is reasonable and not too fragmented.
    """
    h, w = allowed01.shape
    target_range = area_range

    best = None
    best_score = -1.0

    for _ in range(max_tries):
        cand = make_random_blob_mask(h, w, area_range=target_range, aspect_range=aspect_range, rng=rng)
        cand = (cand & allowed01).astype(np.uint8)

        # If intersection makes it too small, try again
        frac = float(cand.mean())
        if frac < target_range[0] * 0.6:
            continue

        # Prefer candidates that keep large connected area (avoid tiny fragments)
        # Simple score = area - fragmentation penalty (using connected components)
        num_labels, labels = cv2.connectedComponents(cand)
        # num_labels includes background
        comp_count = max(0, num_labels - 1)
        score = frac - 0.02 * comp_count  # penalize fragmentation a bit

        if score > best_score:
            best = cand
            best_score = score

        # early accept if good enough
        if target_range[0] <= frac <= target_range[1] and comp_count <= 4:
            return cand

    # fallback: return best found or zeros
    return best if best is not None else np.zeros((h, w), dtype=np.uint8)


# ------------------------
# Blur application (soft edges)
# ------------------------
def apply_blur_on_mask(img_bgr: np.ndarray, mask01: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    ksize = ensure_odd(ksize)
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # soft alpha
    alpha = mask01.astype(np.float32)
    soft_ksize = ensure_odd(int(max(5, min(41, sigma * 2 + 1))))
    alpha = cv2.GaussianBlur(alpha, (soft_ksize, soft_ksize), sigmaX=sigma, sigmaY=sigma)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    out = img_bgr.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------
# Build masks (single blob, low overlap)
# ------------------------
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
    Updated requirement:
    - Single blur region for A and B.
    - A and B should overlap as little as possible.
    - Area fraction in [0.20, 0.50].
    """
    target_area_range = (0.35, 0.50)                    #在这里控制模糊区域大小


    # A: one blob
    maskA = make_random_blob_mask(h, w, area_range=target_area_range, aspect_range=aspect_range, rng=rng)
    maskA = force_area_fraction(maskA, target_area_range, rng)

    # B: generate inside complement (low overlap)
    allowed = (1 - maskA).astype(np.uint8)

    # Try to place B in allowed area
    maskB = make_blob_inside_allowed(
        allowed01=allowed,
        area_range=target_area_range,
        aspect_range=aspect_range,
        rng=rng,
        max_tries=40,
    )
    maskB = force_area_fraction(maskB, target_area_range, rng)

    # Enforce very low overlap: if still overlaps, shrink overlap by projecting into allowed again
    maskB = (maskB & allowed).astype(np.uint8)

    # If B becomes too small after enforcing allowed, do a controlled fallback:
    # allow tiny overlap but keep it minimal (e.g., IoU < 0.02)
    if maskB.mean() < target_area_range[0] * 0.5:
        # fallback: regenerate B on full image but require small IoU
        best = None
        best_iou = 1e9
        for _ in range(50):
            cand = make_random_blob_mask(h, w, area_range=target_area_range, aspect_range=aspect_range, rng=rng)
            cand = force_area_fraction(cand, target_area_range, rng)
            cand_iou = iou(maskA, cand)
            if cand_iou < best_iou:
                best = cand
                best_iou = cand_iou
            if cand_iou <= 0.02:
                best = cand
                best_iou = cand_iou
                break
        maskB = best if best is not None else maskB

    # Final safety: not identical
    if np.array_equal(maskA, maskB):
        maskB = cv2.erode(maskB, np.ones((11, 11), np.uint8), iterations=1)
        maskB = (maskB > 0).astype(np.uint8)

    return maskA.astype(np.uint8), maskB.astype(np.uint8)


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Images folder, e.g. train2017/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for pairN/A.jpg,B.jpg")
    parser.add_argument("--mode", type=str, default="complementary",
                        choices=["independent", "complementary"],
                        help="Kept for compatibility; low-overlap paired masks are used now.")
    parser.add_argument("--start_index", type=int, default=1, help="Start numbering from this index")
    parser.add_argument("--max_images", type=int, default=0, help="0 means process all images")
    parser.add_argument("--seed", type=int, default=42)

    # Blur settings
    parser.add_argument("--blur_ksize", type=int, default=31, help="Gaussian kernel size (odd)")
    parser.add_argument("--sigma", type=float, default=8.0, help="Gaussian sigma")

    # Region settings (kept; actual area range forced to [0.20, 0.50])
    parser.add_argument("--num_rects_a", type=int, default=1)
    parser.add_argument("--num_rects_b", type=int, default=1)
    parser.add_argument("--area_min_a", type=float, default=0.20)
    parser.add_argument("--area_max_a", type=float, default=0.50)
    parser.add_argument("--area_min_b", type=float, default=0.20)
    parser.add_argument("--area_max_b", type=float, default=0.50)
    parser.add_argument("--aspect_min", type=float, default=0.5)
    parser.add_argument("--aspect_max", type=float, default=2.0)

    parser.add_argument("--complement_subset_ratio", type=float, default=0.60)

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
        cv2.imwrite(os.path.join(pair_dir, "A.jpg"), imgA)
        cv2.imwrite(os.path.join(pair_dir, "B.jpg"), imgB)

        idx += 1

    print(f"Done. Wrote pairs: {idx - args.start_index} to {args.output_dir}")
if __name__ == "__main__":
    import sys

    sys.argv = [
        "add_blur.py",
        "--input_dir", r"E:\train2017_part2",
        "--output_dir", r"E:\train2017_part2\output",
        "--mode", "complementary"
    ]
    main()


#改进：
#让模糊区域由 COCO 标注（人/物体）控制，而不是随机矩形