# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lane detection validator — computes MAE / RMSE on valid rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.lane_dataset import LaneDataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

class _LaneMetrics:
    """Minimal metrics container compatible with BaseValidator."""

    def __init__(self):
        self.mae = 0.0
        self.rmse = 0.0
        self.valid_count = 0
        self.results_dict = {}

    @property
    def keys(self):
        return ["metrics/MAE", "metrics/RMSE"]

    def clear(self):
        self.mae = 0.0
        self.rmse = 0.0
        self.valid_count = 0
        self.results_dict = {}

class LaneValidator(BaseValidator):
    """Validator for lane detection models.

    Computes MAE (mean absolute error) and RMSE (root mean squared error)
    only on rows where the ground-truth x-coordinate is valid (≥ 0).
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        """Initialize LaneValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "lane"
        self.metrics = _LaneMetrics()

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build LaneDataset for validation."""
        num_rows = getattr(unwrap_model(self.model).model[-1], "num_rows", 56)
        return LaneDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            num_rows=num_rows,   # must match model head
            augment=False,
            prefix=mode,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Build DataLoader for validation."""
        from torch.utils.data import DataLoader

        dataset = self.build_dataset(dataset_path, mode)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=LaneDataset.collate_fn,
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch to device."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Reset metrics."""
        self.metrics.clear()

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Pass through — no NMS needed for lane regression."""
        return preds

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Accumulate MAE and RMSE.

        Args:
            preds: [B, num_rows] predicted x.
            batch: Batch with 'lane_labels' [B, num_rows].
        """
        target = batch["lane_labels"].to(preds.device)
        mask = target != -1

        if mask.sum() == 0:
            return

        err = preds[mask] - target[mask]
        self.metrics.mae += err.abs().sum().item()
        self.metrics.rmse += (err ** 2).sum().item()
        self.metrics.valid_count += mask.sum().item()

    def get_stats(self) -> dict[str, float]:
        """Return final MAE / RMSE."""
        n = max(self.metrics.valid_count, 1)
        return {
            "metrics/MAE": round(self.metrics.mae / n, 6),
            "metrics/RMSE": round((self.metrics.rmse / n) ** 0.5, 6),
        }

    def get_desc(self) -> str:
        """Return description string for progress bar."""
        stats = self.get_stats()
        return ("%12s" * 3) % ("MAE", "RMSE", "Count") + "\n" + (
            "%12.5f%12.5f%12d" % (stats["metrics/MAE"], stats["metrics/RMSE"], self.metrics.valid_count)
        )

    def print_results(self) -> None:
        """Print validation results."""
        stats = self.get_stats()
        LOGGER.info(
            f"{colorstr('Lane val:')} MAE={stats['metrics/MAE']:.6f}  RMSE={stats['metrics/RMSE']:.6f}  "
            f"valid_points={self.metrics.valid_count}"
        )
