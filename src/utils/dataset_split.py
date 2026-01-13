import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PairingRule:
    """
    配对规则：
    - markers: 用于区分两张图的标记，例如 ("A","B") 或 ("F","N") 等
    - index_pattern: 用于提取数字 i 的正则（默认提取文件名中出现的最后一段数字）
    """
    markers: Tuple[str, str] = ("A", "B")
    index_pattern: str = r"(\d+)"  # 默认抓到数字


class PairFolderOrganizer:
    """
    将散乱的多聚焦图片整理成：
      output_dir/
        pair1/
          A.jpg
          B.jpg
        pair2/
          A.jpg
          B.jpg
        ...

    识别逻辑（默认）：
    - 文件名中包含 marker（A或B，不区分大小写）
    - 文件名中包含数字 i（例如 01、1、001）
    - 同一 i 的 A 和 B 归为同一 pair{i}
    """
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        rule: PairingRule = PairingRule(),
        image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
        action: str = "copy",  # "copy" or "move"
        overwrite: bool = True,
        pair_prefix: str = "pair",  # 输出文件夹前缀
    ):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.rule = rule
        self.image_exts = tuple(e.lower() for e in image_exts)
        self.action = action.lower()
        self.overwrite = overwrite
        self.pair_prefix = pair_prefix

        if self.action not in ("copy", "move"):
            raise ValueError("action must be 'copy' or 'move'")

        self._marker_set = {m.upper() for m in self.rule.markers}
        if len(self._marker_set) != 2:
            raise ValueError("rule.markers must contain exactly two distinct markers, e.g., ('A','B')")

    def _is_image(self, filename: str) -> bool:
        return os.path.splitext(filename)[1].lower() in self.image_exts

    def _detect_marker(self, stem: str) -> Optional[str]:
        """
        从文件名（不含后缀）里识别 A/B 标记。
        规则：找独立的 marker（建议文件名用 - _ 空格 分隔）
        """
        # 允许：- _ 空格 . 等分隔符
        tokens = re.split(r"[-_.\s]+", stem)
        tokens_upper = [t.upper() for t in tokens if t]
        for t in tokens_upper:
            if t in self._marker_set:
                return t  # 返回 "A" or "B"
        # 如果不是独立 token，也尝试末尾/前缀形式（比如 xxxA 或 Axxx 的极端命名）
        for m in self._marker_set:
            if stem.upper().endswith(m):
                return m
            if stem.upper().startswith(m):
                return m
        return None

    def _extract_index(self, stem: str) -> Optional[int]:
        """
        提取数字 i（默认取文件名中出现的最后一段数字）。
        例如 lytro-01-A -> 01 -> 1
        """
        matches = re.findall(self.rule.index_pattern, stem)
        if not matches:
            return None
        # 取最后一个数字段更稳（如 lytro-01-A）
        return int(matches[-1])

    def _safe_copy_or_move(self, src: str, dst: str):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            if self.overwrite:
                os.remove(dst)
            else:
                raise FileExistsError(f"Target already exists: {dst}")

        if self.action == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

    def organize(self) -> Dict[str, int]:
        """
        执行整理。
        返回统计信息：
        - scanned: 扫描到的图片数量
        - matched: 成功匹配到 (index, marker) 的数量
        - pairs_created: 输出 pair 文件夹数量（至少包含一张图的 index 数）
        - complete_pairs: A/B 都齐全的 pair 数
        - skipped: 跳过数量（不符合命名规则、缺 marker、缺 index 等）
        """
        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"input_dir not found: {self.input_dir}")

        os.makedirs(self.output_dir, exist_ok=True)

        # index -> {"A": path, "B": path}
        buckets: Dict[int, Dict[str, str]] = {}

        scanned = matched = skipped = 0

        for root, _, files in os.walk(self.input_dir):
            for fn in files:
                if not self._is_image(fn):
                    continue

                scanned += 1
                src_path = os.path.join(root, fn)
                stem, _ = os.path.splitext(fn)

                marker = self._detect_marker(stem)
                idx = self._extract_index(stem)

                if marker is None or idx is None:
                    skipped += 1
                    continue

                matched += 1
                buckets.setdefault(idx, {})
                # 若同一个 idx/marker 有多张，默认保留“最后遇到的那张”
                buckets[idx][marker] = src_path

        pairs_created = 0
        complete_pairs = 0

        for idx in sorted(buckets.keys()):
            pairs_created += 1
            pair_dir = os.path.join(self.output_dir, f"{self.pair_prefix}{idx}")

            # 输出文件名固定为 A.jpg / B.jpg（统一后缀为 .jpg）
            for marker in self._marker_set:
                if marker in buckets[idx]:
                    dst_path = os.path.join(pair_dir, f"{marker}.jpg")
                    self._safe_copy_or_move(buckets[idx][marker], dst_path)

            if all(m in buckets[idx] for m in self._marker_set):
                complete_pairs += 1

        return {
            "scanned": scanned,
            "matched": matched,
            "pairs_created": pairs_created,
            "complete_pairs": complete_pairs,
            "skipped": skipped,
        }


if __name__ == "__main__":
    # 示例用法：
    # input_dir: 你原始图片所在目录（可以是扁平或多层目录）
    # output_dir: 输出的 pair 数据集目录
    organizer = PairFolderOrganizer(
        input_dir=r"E:\Lytro\d51c5-main\LytroDataset\LytroDataset\LytroDataset",
        output_dir=r"E:\Lytro",
        rule=PairingRule(markers=("A", "B")),  # 可改成 ("F","N") 等
        action="copy",          # "copy" 或 "move"
        overwrite=True,
        pair_prefix="pair"      # 输出 pair1, pair2...
    )
    stats = organizer.organize()
    print("Done:", stats)