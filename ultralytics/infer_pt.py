"""
PyTorch .pt 推理脚本 — 车道线检测推理并可视化
GT=蓝  Pred=绿

用法:
    python infer_pt.py
"""

import sys
from pathlib import Path

ULTRA_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(ULTRA_PATH))

import cv2
import numpy as np
import torch

# ============================================================
# 配置（手动修改）
# ============================================================

PT_PATH = "runs/ufld/yolo26-ufld-7/weights/best.pt"
IMAGES_DIR = "/home/xhm/yolo_ufldv3/datasets2/images/train"
OUTPUT_DIR = "infer_pt"
IMSZ = 640

# ============================================================

def parse_label(img_path):
    label_path = Path(str(img_path).replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt"))
    if not label_path.exists():
        return []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            n = (len(parts) - 1) // 2
            return [(float(parts[1 + 2 * i]), float(parts[2 + 2 * i]))
                    for i in range(n) if float(parts[1 + 2 * i]) >= 0]
    return []


def detect_row_anchors(images):
    for img_path in images:
        label_path = Path(str(img_path).replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt"))
        if not label_path.exists():
            continue
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                return [float(parts[2 + 2 * i]) for i in range((len(parts) - 1) // 2)]
    return [1.0, 0.89, 0.78, 0.67]


def points_to_pixel(pts, w, h):
    return [(int(x * w), int(y * h)) for x, y in pts if 0.01 < x < 0.99]


def main():
    from ultralytics import YOLO

    print(f"Model: {PT_PATH}")
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(PT_PATH, verbose=False)
    model.model.eval()

    images = sorted(Path(IMAGES_DIR).glob("*.jpg")) + sorted(Path(IMAGES_DIR).glob("*.png"))
    print(f"Images: {len(images)}")

    row_anchors = detect_row_anchors(images)
    print(f"Row anchors ({len(row_anchors)}): {[round(y,3) for y in row_anchors[:3]]}...{[round(y,3) for y in row_anchors[-3:]]}\n")

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        h0, w0 = img_bgr.shape[:2]

        # 预处理（与训练完全一致）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMSZ, IMSZ))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

        # 推理
        with torch.no_grad():
            pred_x = model.model(img_tensor)[0].numpy()

        num_rows = len(pred_x)
        pred_norm = [(pred_x[i], row_anchors[i]) for i in range(num_rows)]
        pred_px = points_to_pixel(pred_norm, w0, h0)

        gt_norm = parse_label(img_path)
        gt_px = points_to_pixel(gt_norm, w0, h0)

        print(f"{img_path.name}:  pred={len(pred_px)}pts  GT={len(gt_px)}pts")

        # 可视化
        vis = img_bgr.copy()
        for px, py in gt_px:
            cv2.circle(vis, (px, py), 2, (255, 0, 0), -1)
        if len(gt_px) >= 2:
            cv2.polylines(vis, [np.array(gt_px)], False, (255, 0, 0), 1)

        for px, py in pred_px:
            cv2.circle(vis, (px, py), 2, (0, 255, 0), -1)
        if len(pred_px) >= 2:
            cv2.polylines(vis, [np.array(pred_px)], False, (0, 255, 0), 1)

        cv2.putText(vis, "GT=blue Pred=green", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / img_path.name), vis)

    print(f"\nDone → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
