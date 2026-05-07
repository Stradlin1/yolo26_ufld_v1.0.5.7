# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lane detection predictor — runs inference and returns lane coordinates."""

from __future__ import annotations

import numpy as np

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results


class LanePredictor(BasePredictor):
    """Predictor for YOLO-UFLD lane detection models.

    Post-processes the model's [B, num_rows] x-coordinate output into
    (x, y) lane points using the head's fixed row_anchors.

    Attributes:
        row_anchors (list[float]): Fixed y-positions from the head.
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Convert [B, num_rows] predictions to lane point lists.

        Args:
            preds (torch.Tensor): [B, num_rows] normalized x-coordinates.
            img (torch.Tensor): Preprocessed input images.
            orig_imgs (list[np.ndarray]): Original images (H, W, C).

        Returns:
            list[Results]: Results objects with lane data attached.
        """
        # Get row anchors from head
        head = self.model.model[-1]
        row_anchors = getattr(head, "row_anchors", [1.0, 0.89, 0.78, 0.67])

        lane_results = []
        for i, px in enumerate(preds):
            px = px.cpu().numpy()  # [num_rows]
            points = []
            for j, x_val in enumerate(px):
                if 0.01 < x_val < 0.99:  # valid prediction
                    y_val = row_anchors[j] if j < len(row_anchors) else 0.0
                    points.append([float(x_val), float(y_val)])

            # Optional polyfit
            curve = None
            if len(points) >= 2:
                xs, ys = zip(*points)
                try:
                    curve = np.polyfit(ys, xs, 2).tolist()  # x = A*y² + B*y + C
                except np.linalg.LinAlgError:
                    curve = None

            lane_results.append({
                "lane_x": px.tolist(),
                "row_anchors": row_anchors,
                "lane_points": points,  # [[x, y], ...]
                "fit_curve": curve,     # [A, B, C] or None
            })

        # Wrap in Results
        results = self.construct_results(lane_results, img, orig_imgs)
        return results

    def construct_results(self, lane_data_list: list[dict], img, orig_imgs):
        """Build Results objects with lane data.

        Args:
            lane_data_list: List of lane prediction dicts.
            img: Preprocessed images.
            orig_imgs: Original images.

        Returns:
            list[Results]
        """
        results = []
        for i, ld in enumerate(lane_data_list):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            r = Results(
                orig_img=orig_img,
                path=self.batch[0][i] if self.batch else "",
                names={0: "lane"},
            )
            # Attach lane data
            r.lane = ld
            results.append(r)
        return results
