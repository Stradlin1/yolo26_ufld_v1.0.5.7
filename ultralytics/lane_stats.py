"""
统计每行误差分布（使用 ONNX）

用法:
    python lane_stats.py            # 使用默认路径（datasets2/valid）计算全部验证集
    python lane_stats.py --max 100  # 只跑前 100 张，用于快速调试

输出:
    - 在屏幕打印每一行的 MAE / RMSE / count
    - 写入 infer_debug/row_stats.csv

注意: 脚本默认使用 ONNX 权重路径，请确认 `ONNX_PATH` 指向有效的 onnx 文件。
"""

from pathlib import Path
import argparse
import csv
import sys

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    print("onnxruntime 未安装或导入失败:", e)
    print("请在 conda 环境中安装 onnxruntime，或使用 infer_pt.py 用 PT 版本统计。")
    raise

# 默认配置（可按需修改）
ONNX_PATH = Path("/home/xhm/yolo_ufldv3/ultralytics/runs/ufld/yolo26-ufld-7/weights/best.onnx")
IMGS_ROOT = Path("/home/xhm/yolo_ufldv3/datasets2/images/valid")
LABELS_ROOT = Path("/home/xhm/yolo_ufldv3/datasets2/labels/valid")
IMSZ = 640
OUT_DIR = Path("infer_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_label_path(img_path: Path) -> Path:
    rel = img_path.relative_to(IMGS_ROOT)
    return LABELS_ROOT / rel.with_suffix('.txt')


def parse_label(label_path: Path):
    if not label_path.exists():
        return None
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            n = (len(parts) - 1) // 2
            xs = [float(parts[1 + 2 * i]) for i in range(n)]
            ys = [float(parts[2 + 2 * i]) for i in range(n)]
            return xs, ys
    return None


def main(args):
    if not ONNX_PATH.exists():
        print(f"ONNX not found: {ONNX_PATH}")
        sys.exit(1)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"]) 
    iname = sess.get_inputs()[0].name
    num_rows = int(sess.get_outputs()[0].shape[1])
    print(f"ONNX: {ONNX_PATH}\nnum_rows: {num_rows}")

    imgs = sorted(IMGS_ROOT.rglob('*.jpg')) + sorted(IMGS_ROOT.rglob('*.png'))
    if args.max > 0:
        imgs = imgs[: args.max]
    print(f"Images: {len(imgs)} (from {IMGS_ROOT})")

    # accumulators
    abs_sum = np.zeros(num_rows, dtype=np.float64)
    sq_sum = np.zeros(num_rows, dtype=np.float64)
    cnt = np.zeros(num_rows, dtype=np.int64)

    for idx, ip in enumerate(imgs, 1):
        img = cv2.imread(str(ip))
        if img is None:
            print(f"[skip] can't read {ip}")
            continue
        h0, w0 = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMSZ, IMSZ))
        t = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]

        out = sess.run(None, {iname: t})[0][0]

        label = parse_label(resolve_label_path(ip))
        if label is None:
            # No GT: skip
            continue
        xs, ys = label
        n = min(len(xs), num_rows)
        for i in range(n):
            gt_x = xs[i]
            if gt_x < 0:
                continue
            pred_x = float(out[i])
            err = abs(pred_x - gt_x)
            abs_sum[i] += err
            sq_sum[i] += err * err
            cnt[i] += 1

    # compute stats
    rows = []
    for i in range(num_rows):
        if cnt[i] > 0:
            mae = abs_sum[i] / cnt[i]
            rmse = (sq_sum[i] / cnt[i]) ** 0.5
        else:
            mae = float('nan')
            rmse = float('nan')
        rows.append((i, i, cnt[i], mae, rmse))

    # print summary
    print('\nRow, idx, count, MAE, RMSE')
    for r in rows:
        print(f"row {r[0]:02d}: count={r[2]:4d}  MAE={r[3]:.6f}  RMSE={r[4]:.6f}")

    # write CSV
    csv_path = OUT_DIR / 'row_stats.csv'
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['row_idx', 'count', 'mae', 'rmse'])
        for i, _, c, mae, rmse in rows:
            writer.writerow([i, int(c), mae, rmse])

    print(f"\nSaved CSV → {csv_path.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=0, help='Max images to process (0=all)')
    args = parser.parse_args()
    main(args)
