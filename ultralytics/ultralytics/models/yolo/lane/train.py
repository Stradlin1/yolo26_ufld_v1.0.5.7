# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lane detection trainer."""

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ultralytics.data.lane_dataset import LaneDataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import LaneDetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


class LaneTrainer(BaseTrainer):
    """Trainer for YOLO lane detection models.

    Specializes BaseTrainer for UFLD-style lane detection: uses LaneDataset,
    reports regression/smoothness losses, and sets up lane-specific model attributes.

    Attributes:
        loss_names (tuple): ("reg_loss", "smooth_loss")
        model (LaneDetectionModel): The lane detection model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize LaneTrainer."""
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build LaneDataset for training or validation."""
        num_rows = getattr(unwrap_model(self.model).model[-1], "num_rows", 56)
        return LaneDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            num_rows=num_rows,   # must match model head
            prefix=mode,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Build DataLoader for lane detection.

        Args:
            dataset_path (str): Path to image directory.
            batch_size (int): Batch size.
            rank (int): Process rank for DDP.
            mode (str): 'train' or 'val'.

        Returns:
            torch.utils.data.DataLoader
        """
        dataset = self.build_dataset(dataset_path, mode)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=LaneDataset.collate_fn,
            drop_last=self.args.compile and mode == "train",
        )
        # Ultralytics expects .reset() for infinite training; attach a no-op
        loader.reset = lambda: None
        return loader

    def preprocess_batch(self, batch: dict) -> dict:
        """Move batch tensors to device.

        Args:
            batch (dict): Batch dictionary with 'img' and 'lane_labels'.

        Returns:
            Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def set_model_attributes(self):
        """Set model attributes from dataset config."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a LaneDetectionModel.

        Args:
            cfg (str, optional): YAML config path.
            weights (str, optional): Weights path.
            verbose (bool): Verbose logging.

        Returns:
            LaneDetectionModel
        """
        model = LaneDetectionModel(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a LaneValidator."""
        self.loss_names = "reg_loss", "smooth_loss"
        return yolo.lane.LaneValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return labeled loss items for logging.

        Args:
            loss_items (list[float]): [reg_loss, smooth_loss].
            prefix (str): 'train' or 'val'.

        Returns:
            dict: {"train/reg_loss": ..., "train/smooth_loss": ...}
        """
        keys = [f"{prefix}/{name}" for name in self.loss_names]
        if loss_items is None:
            return keys
        return dict(zip(keys, [round(float(x), 5) for x in loss_items]))

    def progress_string(self):
        """Return formatted training progress string."""
        return ("\n" + "%11s" * (2 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
        )
