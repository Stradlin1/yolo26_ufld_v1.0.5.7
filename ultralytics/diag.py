"""
诊断脚本：对比 PyTorch 模型 vs ONNX 模型的推理结果
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ULTRA_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(ULTRA_PATH))

# ============================================================
# 配置
# ============================================================

PT_PATH = "runs/ufld/yolo26-ufld-2/weights/best.pt"
ONNX_PATH = "runs/ufld/yolo26-ufld-2/weights/best.onnx"
TEST_IMG = "/home/xhm/yolo_ufldv3/datasets/images/train/1777120534969772000.jpg"
IMSZ = 640
ROW_ANCHORS = [1.0, 0.89, 0.78, 0.67]

# ============================================================

def preprocess(img_path):
    """与训练时完全一致的预处理"""
    img_bgr = cv2.imread(img_path)
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMSZ, IMSZ))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, (h0, w0), img_bgr


def main():
    img_tensor, (h0, w0), img_bgr = preprocess(TEST_IMG)

    # ====== PyTorch 推理 ======
    from ultralytics import YOLO
    pt_model = YOLO(PT_PATH, verbose=False)
    pt_model.model.eval()
    with torch.no_grad():
        pt_out = pt_model.model(img_tensor)
    print("PyTorch output:", pt_out.squeeze().tolist())
    print("PyTorch shape:", pt_out.shape)

    # ====== 看 GT 标签 ======
    label_path = TEST_IMG.replace("/images/", "/labels/").replace(".jpg", ".txt")
    if Path(label_path).exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                xs = [float(parts[1+2*i]) for i in range(4)]
                print(f"GT label x:  {xs}")
                # 用 x 映射到像素
                for i, (x, y) in enumerate(zip(xs, ROW_ANCHORS)):
                    if x >= 0:
                        px, py = int(x * w0), int(y * h0)
                        print(f"  row{i}: ({px},{py})")

    # ====== ONNX 推理 ======
    print()
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {iname: img_tensor.numpy()})[0]
    print("ONNX output:", onnx_out.squeeze().tolist())
    print("ONNX shape:", onnx_out.shape)

    # ====== 对比 ======
    diff = np.abs(pt_out.squeeze().numpy() - onnx_out.squeeze())
    print(f"\nPyTorch vs ONNX diff: {diff}")
    if diff.max() < 0.01:
        print("✅ PyTorch ≈ ONNX — 模型本身没问题，可能是预/后处理")
    else:
        print("❌ PyTorch ≠ ONNX — ONNX 导出有问题")


if __name__ == "__main__":
    main()
