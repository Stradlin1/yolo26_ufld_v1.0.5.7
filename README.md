# YOLO26-UFLD：基于 Ultralytics 的车道线检测

在 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 框架上扩展的**纯车道线检测**实现，采用 UFLD 风格的固定行锚点坐标回归，新增 `lane` 任务类型，训练/验证/推理全链路已打通。

## 目录结构

```
├── datasets2/                  # 示例数据集（56 行标注）
│   ├── images/train/  valid/
│   ├── labels/train/  valid/
│   └── data.yaml
├── ultralytics/                # 修改后的 Ultralytics 框架
│   ├── train_lane.py           # 训练入口脚本
│   ├── infer_pt.py             # PyTorch 推理 + 可视化
│   ├── infer_onnx.py           # ONNX 推理 + 可视化
│   ├── lane_stats.py           # 逐行误差统计（MAE/RMSE）
│   ├── test56.py               # 56 行模型构建测试
│   └── ultralytics/            # 框架核心修改
│       ├── nn/modules/head.py          # 新增 UFLDHead
│       ├── nn/tasks.py                 # 新增 LaneDetectionModel
│       ├── utils/loss.py               # 新增 v8LaneDetectionLoss
│       ├── data/lane_dataset.py        # 新增 LaneDataset
│       ├── models/yolo/lane/           # LaneTrainer / Validator / Predictor
│       ├── models/yolo/model.py        # task_map["lane"] 注册
│       └── cfg/
│           ├── default.yaml            # lane_reg / lane_smooth / lane_smooth2
│           └── models/26/
│               ├── yolo26-ufld.yaml        #  4 行版（P3 特征图）
│               └── yolo26-ufld-56.yaml     # 56 行版（P2 特征图）
├── UFLD_LANE_DESIGN.md         # 详细实现说明
└── DESIGN2.md                  # 补充设计笔记
```

## 快速开始

### 环境

```bash
pip install torch ultralytics
# 如需 ONNX 推理
pip install onnxruntime
```

### 数据准备

标签为 `.txt` 文件，每行一个车道实例：

```
class_id x1 y1 x2 y2 ... xN yN
```

- `class_id`：固定为 `0`
- `x_i`：归一化 x 坐标 [0, 1]，`-1` 表示该行无线
- `y_i`：归一化 y 坐标（从上到下递减）

数据配置 `data.yaml` 只需声明路径和类别名：

```yaml
path: /path/to/datasets
train: images/train
val: images/valid
nc: 1
names:
  0: lane
```

`num_rows` / `row_anchors` 由 `LaneDataset` 自动从标签推断，无需在 yaml 中显式指定。

### 训练

编辑 `ultralytics/train_lane.py` 中的配置，然后运行：

```bash
cd ultralytics
python train_lane.py
```

主要可调配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_CFG` | `yolo26-ufld-56.yaml` | 56 行版模型；可选 `yolo26-ufld.yaml`（4 行） |
| `DATA_CFG` | `datasets2/data.yaml` | 数据配置路径 |
| `EPOCHS` | 1000 | 训练轮数 |
| `BATCH` | 8 | 批大小 |
| `IMSZ` | 640 | 输入尺寸 |
| `LR0` | 0.001 | 初始学习率 |
| `RESUME` | True | 断点续训 |

Loss 权重在 `ultralytics/ultralytics/cfg/default.yaml` 中调整：

```yaml
lane_reg: 1.0      # 回归损失
lane_smooth: 0.5   # 一阶平滑
lane_smooth2: 0.1  # 二阶曲率平滑
```

### 推理

**PyTorch：**
```bash
cd ultralytics
# 先修改 infer_pt.py 中的 PT_PATH / IMAGES_DIR
python infer_pt.py
```

**ONNX：**
```bash
cd ultralytics
# 先修改 infer_onnx.py 中的 ONNX_PATH / IMAGES_DIR
python infer_onnx.py
```

可视化输出：GT 蓝色圆点，预测绿色圆点。

### 误差统计

```bash
cd ultralytics
python lane_stats.py            # 全量验证集
python lane_stats.py --max 100  # 前 100 张快速调试
```

输出每行 count / MAE / RMSE，写入 `infer_debug/row_stats.csv`。

### 模型测试

```bash
cd ultralytics
python test56.py
# 输出: num_rows, row_anchors, output shape, 前向结果采样
```

## 模型配置

### 两种行数

| 配置 | 行数 | 特征层 | 适用场景 |
|------|------|--------|----------|
| `yolo26-ufld.yaml` | 4 | P3 (80×80) | 快速验证、轻量部署 |
| `yolo26-ufld-56.yaml` | 56 | P2 (160×160) | 精确定位、当前主力 |

### 切换网络尺度

默认使用 `s` 尺度（~10M 参数）。如需 `m` 尺度（~22M 参数），编辑模型 YAML：

```yaml
scale: m
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
```

## 技术要点

- **任务注册**：`lane` 任务已注册到 Ultralytics `task_map`，`YOLO("yolo26-ufld-56.yaml")` 自动路由到 `LaneDetectionModel`
- **损失函数**：带行权重的 L1 回归 + 一阶平滑 + 二阶曲率平滑，-1 行自动 mask
- **输出格式**：`[B, num_rows]` → 后处理为 `{lane_x, row_anchors, lane_points, fit_curve}`
- **推理过滤**：`0.01 < x < 0.99` 视为有效点，极值试为无线

详细设计见 [UFLD_LANE_DESIGN.md](UFLD_LANE_DESIGN.md)。

## 致谢

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0) 扩展开发。
