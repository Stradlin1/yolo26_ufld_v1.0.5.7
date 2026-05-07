# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lane detection dataset for UFLD-style coordinate regression.

Auto-detects num_rows and row_anchors from label files.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Union

import cv2
import torch
from torch.utils.data import Dataset


class LaneDataset(Dataset):
    """Dataset for lane detection: loads images and parses (class_id, x1,y1, x2,y2,... ) labels.

    Each label file (.txt) contains one line per lane instance:
        class_id x1 y1 x2 y2 ... xN yN

    Where x_i = -1 means no lane at row i. All coordinates normalized [0,1].
    num_rows and row_anchors are auto-detected from the first label file.

    Attributes:
        img_dir (Path): Directory containing images.
        img_paths (list[Path]): Sorted list of image file paths.
        num_rows (int): Number of row anchors (auto-detected).
        row_anchors (list[float]): Fixed y-positions (auto-detected, top-to-bottom).
        imgsz (int): Target image size (square resize).
        augment (bool): Whether to apply augmentations.
    """

    def __init__(
        self,
        img_path: Union[str, Path],
        imgsz: int = 640,
        num_rows: int = 0,
        augment: bool = False,
        prefix: str = "",
    ):
        """Initialize LaneDataset.

        Args:
            img_path (Union[str, Path]): Path to directory containing images.
            imgsz (int): Target image size for resize.
            num_rows (int): Number of row anchors. 0 = auto-detect from labels.
            augment (bool): Enable augmentations.
            prefix (str): Log prefix.
        """
        self.img_dir = Path(img_path)
        self.imgsz = imgsz
        self.augment = augment

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.img_paths = sorted(
            p for p in self.img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        # Auto-detect num_rows and row_anchors from first label
        if num_rows > 0:
            self.num_rows = num_rows
            self.row_anchors = []
        else:
            self.num_rows, self.row_anchors = self._detect_config()

    @staticmethod
    def detect_config(img_dir: Union[str, Path]) -> tuple:
        """Read one label to detect num_rows and row_anchors."""
        return LaneDataset(img_dir, num_rows=0)._detect_config()

    def _detect_config(self) -> tuple:
        """Auto-detect num_rows and row_anchors from the first label file."""
        for img_path in self.img_paths:
            label_path = self._get_label_path(img_path)
            if not label_path.exists():
                continue
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    # parts = [class_id, x1, y1, x2, y2, ...]
                    n = (len(parts) - 1) // 2
                    ys = [float(parts[2 + 2 * i]) for i in range(n)]
                    return n, ys
        return 4, [1.0, 0.89, 0.78, 0.67]  # fallback

    def __len__(self):
        """Return dataset size."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        """Load image and lane label.

        Returns:
            dict with keys: 'img' (Tensor [3,H,W]), 'lane_labels' (Tensor [num_rows]),
                            'img_path' (str), 'ori_shape' (tuple).
        """
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Parse label
        lane_labels = self._load_label(img_path)
        # lane_labels shape: [num_rows], values in [0,1] or -1

        # Resize
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.imgsz, self.imgsz))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        return {
            "img": img,
            "lane_labels": torch.tensor(lane_labels, dtype=torch.float32),
            "img_path": str(img_path),
            "ori_shape": (h0, w0),
            "resized_shape": (self.imgsz, self.imgsz),
        }

    def _load_label(self, img_path: Path) -> np.ndarray:
        """Parse label file for lane x-coordinates.

        Format: class_id x1 y1 x2 y2 ... xN yN

        Only the x values are extracted (y is the row anchor, known from config).

        Returns:
            np.ndarray: [num_rows] x-coordinates, -1 where no lane.
        """
        label_path = self._get_label_path(img_path)
        lane_x = np.full(self.num_rows, -1.0, dtype=np.float32)

        if not label_path.exists():
            return lane_x

        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2 * self.num_rows + 1:
                    continue
                for i in range(self.num_rows):
                    x_str = parts[1 + 2 * i]
                    try:
                        x_val = float(x_str)
                    except ValueError:
                        x_val = -1.0
                    if x_val >= 0:
                        lane_x[i] = x_val
                break  # only first lane instance

        return lane_x

    @staticmethod
    def _get_label_path(img_path: Path) -> Path:
        """Get label path from image path (same stem, .txt extension).

        Labels are expected in a sibling 'labels' directory mirroring 'images'.
        """
        # e.g. datasets/images/train/001.jpg → datasets/labels/train/001.txt
        parts = list(img_path.parts)
        try:
            img_idx = parts.index("images")
            parts[img_idx] = "labels"
        except ValueError:
            # Fallback: same directory, .txt extension
            return img_path.with_suffix(".txt")
        return Path(*parts).with_suffix(".txt")

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate batch of samples into tensors.

        Args:
            batch: List of sample dicts from __getitem__.

        Returns:
            dict with 'img' [B,3,H,W], 'lane_labels' [B,num_rows], 'batch_idx' [B,1], 'im_file' list.
        """
        imgs = torch.stack([s["img"] for s in batch])
        lane_labels = torch.stack([s["lane_labels"] for s in batch])
        batch_idx = torch.arange(len(batch)).unsqueeze(1)
        return {
            "img": imgs,
            "lane_labels": lane_labels,
            "batch_idx": batch_idx,
            "cls": batch_idx.clone(),    # dummy; required by trainer loop
            "bboxes": torch.zeros(len(batch), 0, 4),  # dummy
            "im_file": [s["img_path"] for s in batch],
        }
