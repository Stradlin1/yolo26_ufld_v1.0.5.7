# YOLO26-UFLD 车道线检测实现说明

## 1. 概述

基于 Ultralytics YOLO26 框架的**纯车道线检测**（单任务 lane），采用 UFLD 风格的固定行锚点坐标回归方案。任务类型已注册为 `lane`，链路覆盖模型构建、数据加载、训练、验证、推理全流程。

### 1.1 技术路线

| 组件 | 选型 | 说明 |
|------|------|------|
| Backbone | YOLO26（C3k2 + C2PSA + SPPF） | 缩放因子由 `scales` 控制（n/s/m/l/x） |
| Neck | YOLO26 PAN-FPN（仅上采样路径） | 产生 P3 特征图，Head 只取单层 |
| Head | `UFLDHead`（行级池化 + 线性回归 + Sigmoid） | 每行输出一个归一化 x |
| 任务 | `lane` | 注册于 `task_map`，独立于 detect/segment/pose |
| 输出 | `[B, num_rows]` float | 每行一个 x ∈ [0, 1]；GT 中 x = -1 表示无线 |

### 1.2 与标准 YOLO 检测的关键差异

| | 标准 YOLO Detect | YOLO26-UFLD |
|---|---|---|
| Head | `Detect`（分类 + 回归 + DFL） | `UFLDHead`（纯坐标回归） |
| 原始输出 | `[B, reg_max×4+nc, H×W×anchors]` | `[B, num_rows]` |
| 损失 | box + cls + dfl | reg + smooth + smooth2 |
| 后处理 | NMS → bboxes | 阈值过滤 + row_anchors 拼接 |
| 任务路由 | `task_map["detect"]` | `task_map["lane"]` |

---

## 2. 文件结构

下表列出所有与 lane 任务相关的文件及其角色。

| # | 文件 | 角色 |
|---|------|------|
| 1 | `ultralytics/nn/modules/head.py` | `UFLDHead` 类定义（紧接在现有 head 模块末尾） |
| 2 | `ultralytics/nn/modules/__init__.py` | 导出 `UFLDHead` |
| 3 | `ultralytics/nn/tasks.py` | `LaneDetectionModel`（继承 `DetectionModel`） + `parse_model` 适配 + `guess_model_task` 识别 |
| 4 | `ultralytics/utils/loss.py` | `v8LaneDetectionLoss`（L1 + 一阶平滑 + 二阶曲率平滑） |
| 5 | `ultralytics/data/lane_dataset.py` | `LaneDataset`（图片加载 + 标签解析 + collate） |
| 6 | `ultralytics/models/yolo/lane/train.py` | `LaneTrainer` |
| 7 | `ultralytics/models/yolo/lane/val.py` | `LaneValidator`（MAE / RMSE） |
| 8 | `ultralytics/models/yolo/lane/predict.py` | `LanePredictor`（x → lane_points + 可选 polyfit） |
| 9 | `ultralytics/models/yolo/lane/__init__.py` | 模块导出 |
| 10 | `ultralytics/models/yolo/model.py` | `task_map["lane"]` 注册 |
| 11 | `ultralytics/cfg/default.yaml` | `lane_reg` / `lane_smooth` / `lane_smooth2` 超参 |
| 12 | `ultralytics/cfg/models/26/yolo26-ufld.yaml` | 4 行版 lane 模型定义 |
| 13 | `ultralytics/cfg/models/26/yolo26-ufld-56.yaml` | 56 行版 lane 模型定义（P2 特征图） |
| 14 | `train_lane.py`（仓库根） | 训练启动脚本 |
| 15 | `infer_pt.py`（仓库根） | PT 推理 + 可视化 |
| 16 | `infer_onnx.py`（仓库根） | ONNX 推理 + 可视化 |
| 17 | `lane_stats.py`（仓库根） | 逐行 MAE / RMSE 统计 |
| 18 | `test56.py`（仓库根） | 56 行模型构建 + 前向测试 |

---

## 3. 数据格式

### 3.1 标签文件 (.txt)

每行一个车道线实例，格式：

```
class_id x1 y1 x2 y2 ... xN yN
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `class_id` | int | 固定为 `0`（车道线类别） |
| $x_i$ | float | 归一化 x 坐标，值域 [0, 1]，**-1 表示该行无车道线** |
| $y_i$ | float | 归一化 y 坐标，值域 [0, 1]，从上到下递减 |

- `LaneDataset` 解析标签时**只提取 x 坐标**（y 坐标仅用于自动检测行数和锚点），训练时回归目标是 x。
- 多实例标签文件：当前 `_load_label()` 只取**第一条**车道实例。

### 3.2 标签示例

**完整车道线（4 行都有）：**
```
0 0.455 1.000 0.457 0.890 0.455 0.780 0.454 0.670
```

**第 3 行无线（虚线间隙）：**
```
0 0.450 1.000 0.452 0.890 -1 0.000 0.448 0.670
```

**只有底部两行有车道线：**
```
0 -1 0.000 -1 0.000 0.460 0.780 0.455 0.670
```

### 3.3 数据集目录结构

```
datasets2/
├── images/
│   ├── train/
│   │   └── *.jpg
│   └── valid/
│       └── *.jpg
├── labels/
│   ├── train/
│   │   └── *.txt          # 与图片同名
│   └── valid/
│       └── *.txt
└── data.yaml
```

### 3.4 数据配置文件 (data.yaml)

当前使用的配置（`datasets2/data.yaml`）：

```yaml
path: /home/xhm/yolo_ufldv3/datasets2
train: images/train
val: images/valid
nc: 1

names:
  0: lane
```

> **注意**：`LaneDataset` 会自动从标签文件推断 `num_rows` 和 `row_anchors`，因此 **不需要** 在 `data.yaml` 中显式写 `num_rows` / `row_anchors`。训练时 `num_rows` 以模型 head 为准（自动匹配）。

---

## 4. 模型架构

### 4.1 整体流程

```
输入: [B, 3, H, W]
    │
    ▼
YOLO26 Backbone (C3k2 + C2PSA + SPPF)
    │
    ├── P2/4  (仅 56 行版使用)
    ├── P3/8  ──────┐
    ├── P4/16 ────┐ │
    └── P5/32 ──┐ │ │
                │ │ │
    ▼           │ │ │
YOLO26 PAN-FPN Neck (仅上采样路径)
    │           │ │ │
    ├── (P2_feat)◄┘ │ │  ← 56 行版：上采样到 P2
    ├── P3_feat ◄───┘ │  ← 4 行版 / 56 行版的中间层
    ├── P4_feat ◄─────┘
    └── P5_feat ◄───────┘
    │
    ▼ (取最终特征层)
UFLDHead
    │
    ▼
输出: [B, num_rows]  (Sigmoid 约束到 [0, 1])
```

### 4.2 UFLDHead 内部结构

```
输入特征图 [B, C, H_feat, W_feat]
    │
    ▼ Conv2d 3×3 (C → C//2, stride=1, pad=1)
    ▼ BatchNorm2d + SiLU
    ▼ AdaptiveAvgPool2d(output_size=(num_rows, W_feat))
    │   → [B, C//2, num_rows, W_feat]   ← 每个池化条带对应一个行锚点
    │
    ▼ Conv2d 1×1 (C//2 → 128)
    ▼ BatchNorm2d + SiLU
    ▼ AdaptiveAvgPool2d(output_size=(num_rows, 1))
    │   → [B, 128, num_rows, 1]         ← 整行信息聚合
    │
    ▼ Flatten(1, 2) → [B, num_rows, 128]
    ▼ Linear(128 → 1) → [B, num_rows, 1]
    ▼ Squeeze(-1) → [B, num_rows]
    ▼ Sigmoid → [B, num_rows]  (值域 [0, 1])
```

### 4.3 两种模型配置

| 配置 | YAML | num_rows | 特征层 | 输入尺寸 | 行锚点 y 范围 |
|------|------|----------|--------|----------|---------------|
| 4 行版 | `yolo26-ufld.yaml` | 4 | P3 (stride 8) | 640×640 | 0.67 → 1.0 |
| 56 行版 | `yolo26-ufld-56.yaml` | 56 | P2 (stride 4) | 640×640 | 0.67 → 1.0，步长 0.006 |

**4 行版**（`yolo26-ufld.yaml`）：neck 上采样到 P3，适用于快速验证。
**56 行版**（`yolo26-ufld-56.yaml`）：neck 继续上采样到 P2（160×160），空间分辨率高 4 倍，适用于精确车道线定位。当前 `train_lane.py` 默认使用此配置。

### 4.4 切换尺度 (n/s/m/l/x)

模型 YAML 通过 `scales` + `scale` 控制 depth/width/max_channels。当前 lane YAML **只定义了 `s` 尺度**。如需使用 `m` 尺度网络，将 YAML 中的 `scales` 补全并修改 `scale`：

```yaml
scale: m
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
```

---

## 5. 损失函数

### 5.1 总损失

$$L_{total} = \lambda_{reg} \cdot L_{reg} + \lambda_{smooth} \cdot L_{smooth} + \lambda_{smooth2} \cdot L_{smooth2}$$

### 5.2 回归损失 $L_{reg}$

带行权重的 L1 损失，行越靠下权重越高（底部 5 行 ×3，底部 5~15 行 ×2）。**-1 行被 mask 掉，不参与梯度**：

$$L_{reg} = \frac{1}{N_{valid}} \sum w_{row} \cdot |x_{pred} - x_{gt}|$$

### 5.3 一阶平滑损失 $L_{smooth}$

约束两个相邻有效行的 x 差值不要过大，抑制锯齿：

$$L_{smooth} = \frac{\sum \mathbb{1}[valid_i \land valid_{i+1}] \cdot |x_i - x_{i+1}|}{\sum \mathbb{1}[valid_i \land valid_{i+1}]}$$

### 5.4 二阶曲率平滑损失 $L_{smooth2}$

约束三连有效行的二阶差分，惩罚曲率过大：

$$L_{smooth2} = \frac{\sum \mathbb{1}[valid_i \land valid_{i+1} \land valid_{i+2}] \cdot |x_i - 2x_{i+1} + x_{i+2}|}{\sum \mathbb{1}[\cdots]}$$

### 5.5 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lane_reg` | 1.0 | 回归损失权重 |
| `lane_smooth` | 0.5 | 一阶平滑损失权重 |
| `lane_smooth2` | 0.1 | 二阶曲率平滑损失权重 |

> 这三个参数只需在 `default.yaml` 中修改，框架会自动注入到 loss 中（`getattr(self.hyp, ...)`），**无需改任何 .py 文件**。

---

## 6. 训练、验证、推理

### 6.1 训练

入口脚本：`train_lane.py`

```python
from ultralytics import YOLO

model = YOLO("cfg/models/26/yolo26-ufld-56.yaml")
model.train(data="datasets2/data.yaml", epochs=1000, imgsz=640, batch=8, ...)
```

关键组件：
- **数据集**：`LaneDataset` 自动推断 `num_rows`/`row_anchors`，训练时 `num_rows` 与模型 head 对齐。
- **Trainer**：`LaneTrainer`（继承 `BaseTrainer`），覆盖 `build_dataset`、`get_dataloader`、`get_model`、`get_validator` 等。
- **Loss**：`v8LaneDetectionLoss`，训练过程中自动通过 `model.loss(batch)` 调用。
- **断点续训**：设置 `RESUME = True` 即可。

### 6.2 验证

验证器：`LaneValidator`，计算 MAE 和 RMSE（仅统计有效行，x ≥ 0）。

训练过程中每轮自动运行验证，也可单独调用。

### 6.3 逐行误差统计

脚本：`lane_stats.py`

```bash
python lane_stats.py            # 全量验证集
python lane_stats.py --max 100  # 前 100 张快速调试
```

输出每行的 count / MAE / RMSE，并写入 `infer_debug/row_stats.csv`。默认使用 ONNX 权重。

### 6.4 PT 推理 + 可视化

脚本：`infer_pt.py`

- GT 蓝色，Pred 绿色
- 配置 `PT_PATH` / `IMAGES_DIR` / `OUTPUT_DIR` 后直接运行
- `row_anchors` 从标签文件自动检测

### 6.5 ONNX 推理 + 可视化

脚本：`infer_onnx.py`

- 与 PT 版本输出格式相同
- 运行前确认 `ONNX_PATH` 指向有效 `.onnx` 文件

### 6.6 推理输出格式

`LanePredictor.postprocess()` 将 `[B, num_rows]` 转换为：

```python
{
    "lane_x":       [0.45, 0.46, 0.12, 0.45],   # 原始 x 坐标列表
    "row_anchors":  [1.0, 0.89, 0.78, 0.67],    # 固定 y 坐标
    "lane_points":  [[0.45,1.0], [0.46,0.89], ...],  # 有效 (x,y) 点
    "fit_curve":    [A, B, C] | None,            # 二次拟合 x=Ay²+By+C
}
```

有效点过滤阈值：`0.01 < x < 0.99`。极值（≈0 或 ≈1）视为无线。

---

## 7. 模型输出 vs 标准 YOLO

### 7.1 前向输出

```python
# 标准 YOLO Detect
out = model.model(torch.randn(1, 3, 640, 640))
# → dict with 'one2many'/'one2one' or Tensor [1, 84, 8400]

# YOLO26-UFLD
out = model.model(torch.randn(1, 3, 640, 640))
# → Tensor [1, num_rows]，如 [1, 56]（Sigmoid 约束到 [0, 1]）
```

| | YOLO26 Detect | YOLO26-UFLD |
|---|---|---|
| Head 类 | `Detect` | `UFLDHead` |
| 原始输出 shape | `[B, 84, 8400]`（n 尺度） | `[B, num_rows]` |
| 原始输出内容 | 8400 anchors × (4 bbox + 80 cls) | 每行一个 x 坐标 |
| 后处理 | NMS → bboxes | threshold + row_anchors → lane_points |
| 对外格式 | `Results.boxes` | `Results.lane` |

### 7.2 任务自动识别

`guess_model_task()` 支持通过以下方式自动识别 `lane` 任务：
- YAML head 末尾模块名包含 `ufld`
- 模型实例中有 `UFLDHead` 模块

---

## 8. 配置项速查

### 8.1 default.yaml 超参

```yaml
lane_reg: 1.0      # 回归损失权重
lane_smooth: 0.5   # 一阶平滑权重
lane_smooth2: 0.1  # 二阶曲率平滑权重
```

### 8.2 train_lane.py 主要配置

```python
MODEL_CFG = "cfg/models/26/yolo26-ufld-56.yaml"
DATA_CFG  = "/home/xhm/yolo_ufldv3/datasets2/data.yaml"
EPOCHS   = 1000
IMSZ     = 640
BATCH    = 8
DEVICE   = "0"
LR0      = 0.001
COS_LR   = True
PATIENCE = 80
AMP      = False
RESUME   = True
```

### 8.3 模型 YAML 关键字段

```yaml
nc: 1                              # 类别数（固定为 1）
num_rows: 56                       # 行锚点数
row_anchors: [0.67, 0.676, ...]    # 固定 y 坐标（从 YAML 传入 head）
scale: s                           # 缩放尺度（改为 m 即用 medium 网络）
scales:
  s: [0.50, 0.50, 1024]
```

---

## 9. 验证标准

```bash
# 1. 模型构建成功
python test56.py
# 预期：输出 num_rows / row_anchors / output shape / sample 值

# 2. 前向传播 shape 正确
python -c "
import torch
from ultralytics.nn.tasks import LaneDetectionModel
m = LaneDetectionModel('cfg/models/26/yolo26-ufld-56.yaml', ch=3, nc=1, verbose=False)
out = m(torch.randn(1, 3, 640, 640))
assert out.shape == (1, m.model[-1].num_rows), f'Expected (1,{m.model[-1].num_rows}), got {out.shape}'
print(f'OK: {out.shape}')
"

# 3. 训练不报错
python train_lane.py

# 4. 推理产出可视化
python infer_pt.py
python infer_onnx.py

# 5. 逐行误差统计
python lane_stats.py
```

