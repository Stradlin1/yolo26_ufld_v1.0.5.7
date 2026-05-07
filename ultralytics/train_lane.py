"""
YOLO26-UFLD 车道线检测训练脚本

用法:
    python train_lane.py

修改下面的配置参数即可。
"""

import sys
from pathlib import Path

# 确保 ultralytics 可导入
ULTRA_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(ULTRA_PATH))

from ultralytics import YOLO

# ============================================================
# 配置参数
# ============================================================

MODEL_CFG = "cfg/models/26/yolo26-ufld-56.yaml"   # 模型 YAML (56行版本)
DATA_CFG  = "/home/xhm/yolo_ufldv3/datasets2/data.yaml"  # 数据配置

EPOCHS   = 1000         # 训练轮数
IMSZ     = 640        # 图像尺寸
BATCH    = 8          # batch size
DEVICE   = "0"        # GPU 编号，"cpu" 用 CPU，"0,1" 多卡
WORKERS  = 4          # 数据加载线程数
LR0      = 0.001      # 初始学习率
COS_LR   = True       # 余弦退火
PATIENCE = 80         # 早停 patience
AMP      = False      # 混合精度（CPU 设 False，GPU 设 True）
RESUME   = True      # 是否断点续训

# 项目保存路径
PROJECT  = str(Path(__file__).resolve().parent / "runs" / "ufld")
NAME     = "yolo26-ufld"


def main():
    # 加载模型
    model = YOLO(MODEL_CFG)

    # 训练
    model.train(
        data=DATA_CFG,
        epochs=EPOCHS,
        imgsz=IMSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        lr0=LR0,
        cos_lr=COS_LR,
        patience=PATIENCE,
        resume=RESUME,
        project=PROJECT,
        name=NAME,
        amp=AMP,            # 混合精度
    )


if __name__ == "__main__":
    main()
