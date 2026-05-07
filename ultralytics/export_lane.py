"""
YOLO26-UFLD 车道线模型导出脚本

用法:
    python export_lane.py
"""

import sys
from pathlib import Path

ULTRA_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(ULTRA_PATH))

from ultralytics import YOLO

# 训练好的权重路径（手动修改）
WEIGHTS = "runs/ufld/yolo26-ufld-9/weights/best.pt"

model = YOLO(WEIGHTS)
model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    opset=11,
    simplify=True,
    half=False,
    int8=False,
)
print(f"Exported to {WEIGHTS.replace('.pt', '.onnx')}")
