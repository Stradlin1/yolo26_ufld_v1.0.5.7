# DESIGN2 — YOLO26-UFLD v2 改进设计

## 变更动机

v1 模型在验证集上表现为「下半段行位严重偏离 GT，且预测呈曲线状而非近直线」。
分析逐行误差 CSV（`infer_debug/row_stats.csv`）后发现：

- 上半段（row 0~17）MAE ≈ 0.10~0.13，尚可
- 下半段（row 37~54）MAE ≈ 0.17~0.23，且样本数极少（仅 8 张有标注），泛化差

根因定位：**v1 的 UFLDHead 用 `adaptive_avg_pool2d(W→1) + FC(128→1)` 回归 x，**
宽度方向的空间信息被完全压掉，模型只能从 128 维全局向量「猜」坐标，
自然出现整体偏移、曲线状失真。

---

## 本次改动总览

| # | 文件 | 改动 | 动机 |
|---|------|------|------|
| 1 | `nn/modules/head.py` | UFLDHead 从 FC 回归改为 **spatial softmax 分类** | 强制空间定位，消除全局偏移 |
| 2 | `cg/models/26/yolo26-ufld-56.yaml` | Neck 增加 **P2 输出路径**，Head 接入 P2 | 每格 8px → 4px，空间精度翻倍 |
| 3 | `utils/loss.py` | 已有改动（行权重 + 二阶平滑） | 抑制下半段漂移和曲线化 |
| 4 | `ultralytics/infer_onnx.py` | 已有调试版（GT/Pred 对照 + 逐行日志） | 快速定量诊断 |

---

## 1. UFLDHead：FC 回归 → Spatial Softmax 分类

### 1.1 v1 的问题

```
P3 [B,256,80,80]
  → conv1 → [B,128,80,80]
  → pool H  → [B,128,56,80]     ← 宽度 80 列仍保留
  → conv2 → [B,128,56,80]
  → pool W  → [B,128,56,1]      ← ❌ 80 列压成 1，空间信息丢失
  → FC(128→1) → [B,56]          ← ❌ 从 128 维全局向量硬猜 x
  → sigmoid
```

### 1.2 v2 的设计

```
P2 [B,C,160,160]                ← stride 4，4px/格（或 P3 80×80）
  → conv1 (3×3) → [B,C_mid,160,160]
  → pool H  → [B,C_mid,56,160]  ← 保持 160 列宽度
  → conv2 (1×1) → [B,256,56,160]
  → conv3 (3×3) → [B,128,56,160]  ← 3×3 聚合局部上下文
  → spatial_conv (1×1) → [B,1,56,160]
  → softmax(沿宽度) → 概率分布 [B,56,160]
  → Σ(prob × bin_center) → [B,56]  ← 期望 x ∈ [0,1]
```

**关键差异**：不再把宽度压成 1。模型必须对每行输出一个「车道线在哪一列」的概率分布，
然后用期望值算出归一化 x。这直接解决「全局向量猜坐标」的问题。

### 1.3 新增的 conv3 层

`conv3` 是 3×3 卷积，在 `conv2`（1×1 逐点）之后对每行的局部邻域做一次上下文聚合。
这比直接 `1×1 → softmax` 多了一点点感受野，让每列能看到左右邻居，平滑预测。

### 1.4 ONNX 兼容性

`softmax` + `linspace` + 加权求和 都是 ONNX 标准算子，不再需要 `adaptive_avg_pool2d` 的
除不尽问题，也不需要 `torch.onnx.is_in_onnx_export()` 分支。

---

## 2. Neck：增加 P2 输出路径

### 2.1 v1 结构

```
P5 → upsample → +P4 → P4_out
                → upsample → +P3 → P3_out (80×80, stride 8) → UFLDHead
```

每格 640/80 = 8 px。

### 2.2 v2 结构

```
P5 → upsample → +P4 → P4_out
                → upsample → +P3 → P3_out
                            → upsample → +P2 → P2_out (160×160, stride 4) → UFLDHead
```

每格 640/160 = 4 px，空间精度翻倍。

### 2.3 层索引对照

| 层号 | 内容 | 分辨率 | 说明 |
|------|------|--------|------|
| 2 | backbone C3k2 | 160×160 | P2 源特征（concat 来源） |
| 4 | backbone C3k2 | 80×80 | P3 源特征 |
| 6 | backbone C3k2 | 40×40 | P4 源特征 |
| 16 | neck C3k2 | 80×80 | P3_out |
| 17 | nn.Upsample | 160×160 | P3_out → 2× |
| 18 | Concat(17, 2) | 160×160 | + P2 backbone |
| 19 | neck C3k2 | 160×160 | **P2_out**（Head 输入） |
| 20 | UFLDHead | — | 接 P2_out |

### 2.4 通道数（scale=s, width=0.50）

| 层 | 输入通道 | 输出通道 |
|----|---------|---------|
| P2 backbone (layer 2) | — | 128 |
| P3_out (layer 16) | — | 128 |
| Concat (layer 18) | 128+128 | 256 |
| P2_out (layer 19) | 256 | 128 |
| UFLDHead conv1 | 128 | 128 |

---

## 3. 损失函数（已在 v1 后期加入）

### 3.1 行权重

```python
row_weights[-5:] = 3.0      # 最底部 5 行 3× 权重
row_weights[-15:-5] = 2.0   # 中下部 10 行 2× 权重
```

在 L1 回归损失中对有效行做加权平均，让模型更关注底部行。

### 3.2 二阶平滑（曲率惩罚）

```python
second_diff = pred[:-2] - 2*pred[1:-1] + pred[2:]
loss_smooth2 = |second_diff|.mean()  # 仅在连续三行都有效时计算
```

超参：`lane_smooth2=0.05`（默认）。

### 3.3 总损失

$$L = \lambda_{reg} \cdot L_{reg}^{weighted} + \lambda_{smooth} \cdot L_{smooth} + \lambda_{smooth2} \cdot L_{smooth2}$$

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lane_reg | 1.0 | 加权 L1 回归 |
| lane_smooth | 0.1 | 一阶平滑 |
| lane_smooth2 | 0.05 | 二阶曲率平滑 |

---

## 4. 推理与可视化

调试脚本 `ultralytics/infer_onnx.py` 已更新：
- 自动读取 `datasets2/labels/` 对应标签
- 每行打印 `pred_x`、`gt_x`、像素差 `dx`
- 输出蓝（GT）绿（Pred）叠加图到 `infer_debug/`

统计脚本 `ultralytics/lane_stats.py` 可生成全验证集的 `row_stats.csv`。

---

## 5. 预期效果

| 指标 | v1 (FC回归) | v2 (Spatial Softmax + P2) |
|------|------------|--------------------------|
| 下半段 MAE | 0.17~0.23 | 预期 < 0.10 |
| 整体偏移 | 存在（全局特征偏差） | 消除（空间定位） |
| 曲线化 | 明显 | 大幅减轻（二阶平滑 + 空间约束） |
| 训练收敛 | 正常 | 正常（softmax 梯度优于 sigmoid+L1） |

---

## 6. 验证步骤

```bash
# 1. 构建新模型
python -c "
import sys; sys.path.insert(0, '.')
from ultralytics.nn.tasks import LaneDetectionModel
m = LaneDetectionModel('cfg/models/26/yolo26-ufld-56.yaml', ch=3, nc=1, verbose=False)
print('Model OK, last layer:', type(m.model[-1]).__name__)
print('num_rows:', m.model[-1].num_rows)
import torch
out = m.model(torch.randn(1, 3, 640, 640))
print('Output shape:', out.shape, 'range:', out.min().item(), out.max().item())
"

# 2. 短训 3~5 epoch
python train_lane.py --epochs 5 --batch 8

# 3. 导出 ONNX
python export_lane.py

# 4. 统计逐行误差
python lane_stats.py --max 200

# 5. 可视化对比
python infer_onnx.py
# 查看 infer_debug/ 下图片中蓝绿点对齐情况
```
