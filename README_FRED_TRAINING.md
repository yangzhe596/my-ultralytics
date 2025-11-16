# Fred自定义COCO数据集YOLOv8训练指南

本指南介绍如何使用Fred自定义COCO数据集训练YOLOv8模型。

## 数据集结构

你的数据集应该位于 `datasets/fred_coco/rgb/` 目录下，结构如下：

```
datasets/fred_coco/
├── rgb/
│   ├── [图片文件或子目录]
└── annotations/
    ├── instances_train.json  # 训练集标注文件
    ├── instances_val.json    # 验证集标注文件
    └── instances_test.json   # 测试集标注文件（可选）
```

## 一键训练脚本

我们提供了两种训练方式：

### 1. 完整一键训练（推荐）

使用 `train_fred_coco_complete.py` 脚本，它会自动：
- 检查数据集结构
- 创建YOLO训练所需的目录结构
- 移动图片到正确的位置
- 自动检测类别信息并创建配置文件
- 开始训练

```bash
# 使用默认参数训练
python train_fred_coco_complete.py

# 使用自定义参数训练
python train_fred_coco_complete.py --model-size s --epochs 200 --batch-size 32
```

### 2. 分步训练

如果你想要更多控制，可以分步执行：

#### 步骤1：准备数据集

```bash
python prepare_fred_dataset.py
```

#### 步骤2：训练模型

```bash
# 使用默认参数训练
python train_fred_coco.py

# 使用自定义参数训练
python train_fred_coco.py --model-size s --epochs 200 --batch-size 32
```

### 3. 使用Shell脚本

```bash
# 使用默认参数训练
./run_training.sh

# 指定模型大小
./run_training.sh s  # 使用YOLOv8s模型
```

## 参数说明

### 模型参数
- `--model-size`: YOLOv8模型大小，可选值为 `n`, `s`, `m`, `l`, `x`（从Nano到Extra Large）

### 训练参数
- `--epochs`: 训练轮数（默认：100）
- `--batch-size`: 批次大小（默认：16）
- `--img-size`: 图像尺寸（默认：640）
- `--device`: 训练设备（默认：0，表示使用第一个GPU；使用cpu表示CPU训练）

### 优化参数
- `--optimizer`: 优化器类型（默认：AdamW，可选：SGD, Adam）
- `--learning-rate`: 初始学习率（默认：0.01）
- `--weight-decay`: 权重衰减（默认：0.0005）
- `--patience`: 早停耐心值（默认：50）

### 输出参数
- `--save-period`: 模型保存周期（默认：10）
- `--workers`: 数据加载工作进程数（默认：8）

## 训练结果

训练完成后，模型和结果将保存在 `runs/train/fred_coco_[model_size]/` 目录下：

```
runs/train/fred_coco_n/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最后一个epoch的模型
├── results.csv      # 训练结果
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── PR_curve.png
└── R_curve.png
```

## 使用训练好的模型

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/fred_coco_n/weights/best.pt')

# 进行预测
results = model('path/to/image.jpg')

# 可视化结果
for r in results:
    r.show()
```

## 常见问题

### 1. 内存不足
如果遇到内存不足问题，可以尝试：
- 减小 `batch-size`
- 减小 `img-size`
- 使用更小的模型（例如 `yolov8n` 而不是 `yolov8x`）

### 2. 训练速度慢
如果训练速度慢，可以尝试：
- 增加 `workers` 数量
- 使用GPU训练（设置 `--device 0`）
- 使用更小的模型

### 3. 检测精度低
如果检测精度不理想，可以尝试：
- 增加训练轮数（`--epochs`）
- 调整学习率（`--learning-rate`）
- 使用更大的模型
- 检查数据集质量和数量

## 高级用法

### 1. 使用自定义配置

你可以创建自己的数据集配置文件，参考 `datasets/fred_coco.yaml`。

### 2. 恢复训练

如果训练中断，可以从上次保存的模型继续训练：

```python
from ultralytics import YOLO

# 加载最后一个模型
model = YOLO('runs/train/fred_coco_n/weights/last.pt')

# 继续训练
model.train(data='datasets/fred_coco.yaml', epochs=100, resume=True)
```

### 3. 超参数调优

Ultralytics提供了自动超参数调优功能：

```bash
yolo tune data=datasets/fred_coco.yaml model=yolov8n.pt epochs=50
```

## 更多信息

- [Ultralytics官方文档](https://docs.ultralytics.com/)
- [COCO数据集格式说明](https://cocodataset.org/#format-data)
- [YOLOv8模型说明](https://docs.ultralytics.com/models/yolov8/)