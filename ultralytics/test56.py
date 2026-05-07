"""测试 56 行 UFLD 模型构建和前向传播"""
import torch
from ultralytics.nn.tasks import LaneDetectionModel

m = LaneDetectionModel("cfg/models/26/yolo26-ufld-56.yaml", ch=3, nc=1, verbose=False)
print("num_rows:", m.model[-1].num_rows)
print("row_anchors[:3]:", m.model[-1].row_anchors[:3])
print("row_anchors[-3:]:", m.model[-1].row_anchors[-3:])

x = torch.randn(1, 3, 640, 640)
out = m(x)
print("output shape:", out.shape)
print("output sample:", out[0, :5].tolist(), "...")
print("OK")
