"""
ONNX 推理脚本 — 车道线检测可视化
GT=蓝色  Pred=绿色

用法: python infer_onnx.py
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ====================== 配置 ======================
ONNX_PATH = "/home/xhm/yolo_ufldv3/ultralytics/runs/ufld/yolo26-ufld-9/weights/best.onnx"
IMAGES_DIR = "/home/xhm/yolo_ufldv3/datasets2/images/valid"
LABELS_DIR = "/home/xhm/yolo_ufldv3/datasets2/labels/valid"
OUTPUT_DIR = "infer_debug"
IMSZ = 640
MAX_IMAGES = 100  # 先跑少量调试
# ==================================================


def main():
    onnx_path = Path(ONNX_PATH)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    print(f"ONNX: {onnx_path}")
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    num_rows = int(sess.get_outputs()[0].shape[1])
    print(f"Input: {sess.get_inputs()[0].shape}")
    print(f"Output: {sess.get_outputs()[0].shape} → {num_rows} rows")

    images_root = Path(IMAGES_DIR)
    labels_root = Path(LABELS_DIR)
    imgs = sorted(images_root.glob("*.jpg")) + sorted(images_root.glob("*.png"))
    if MAX_IMAGES > 0:
        imgs = imgs[:MAX_IMAGES]
    print(f"Images: {len(imgs)}\n")

    # 从 labels 中自动读取 row_anchors（和训练一致）
    label_sample = labels_root / (imgs[0].stem + ".txt")
    row_anchors = []
    if label_sample.exists():
        with open(label_sample) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    n = (len(parts) - 1) // 2
                    row_anchors = [float(parts[2 + 2 * i]) for i in range(n)]
                    break
    if not row_anchors or len(row_anchors) != num_rows:
        row_anchors = [0.67 + i * 0.006 for i in range(num_rows)]

    print(f"Row anchors: {len(row_anchors)} → [{row_anchors[0]:.3f} ... {row_anchors[-1]:.3f}]\n")

    for idx, img_path in enumerate(imgs):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]

        # 预处理（和训练一致：resize 到 640×640，不做 letterbox）
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMSZ, IMSZ))
        tensor = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]

        # 推理
        pred_x = sess.run(None, {iname: tensor})[0][0]

        # 读 GT 标签
        label_path = labels_root / (img_path.stem + ".txt")
        gt_pts = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 * num_rows + 1:
                        for i in range(num_rows):
                            xv = float(parts[1 + 2 * i])
                            yv = float(parts[2 + 2 * i])
                            if xv >= 0:
                                gt_pts.append((xv, yv))
                        break

        # 统计预测值范围
        print(f"[{idx+1}/{len(imgs)}] {img_path.name}  size=({w0},{h0})")
        print(f"  pred_x min={pred_x.min():.4f}  max={pred_x.max():.4f}  mean={pred_x.mean():.4f}")
        print(f"  GT points: {len(gt_pts)}")

        # 可视化
        vis = bgr.copy()
        n_pred = 0
        for i in range(num_rows):
            xv = float(pred_x[i])
            yv = float(row_anchors[i])
            # 放宽过滤：只要不是极端值就画
            if 0.001 < xv < 0.999:
                px, py = int(xv * w0), int(yv * h0)
                cv2.circle(vis, (px, py), 2, (0, 255, 0), -1)
                n_pred += 1

        for gx, gy in gt_pts:
            cv2.circle(vis, (int(gx * w0), int(gy * h0)), 2, (255, 0, 0), -1)

        cv2.putText(vis, f"Pred={n_pred} GT={len(gt_pts)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out / img_path.name), vis)

    print(f"\nDone → {out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
