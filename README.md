# 语义分割学习

基于 PyTorch 的语义分割入门与实践项目。通过三个经典数据集、两种代表性模型架构，系统练习从数据加载、增强、训练到评估与可视化的完整流程。

本项目以 **Jupyter Notebook** 为主要形式，每个数据集对应一个独立 Notebook，可按难度由浅入深依次学习。

---

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [学习路线建议](#学习路线建议)
- [子项目概览](#子项目概览)
  - [1. Oxford-IIIT Pet（入门）](#1-oxford-iiit-pet入门)
  - [2. Pascal VOC 2012（进阶）](#2-pascal-voc-2012进阶)
  - [3. Cityscapes（高级）](#3-cityscapes高级)
- [核心概念](#核心概念)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

---

## 项目结构

```
语义分割学习/
├── README.md
├── Oxford-IIIT Pet数据集/
│   ├── code.ipynb                 # 主 Notebook
│   └── data/
│       ├── images/                # 宠物 RGB 图像（已包含）
│       └── annotations/
│           ├── trimaps/           # 三分割标注（前景/背景/边界）
│           └── xmls/              # 原始 XML 标注
├── Pascal VOC 2012数据集/
│   ├── code.ipynb
│   └── data/
│       ├── splits/                # 官方 train / val / trainval 划分
│       │   ├── train.txt          # 1464 张
│       │   ├── val.txt            # 1449 张
│       │   └── trainval.txt       # 2913 张
│       ├── images/                # 需自行下载
│       └── masks/                 # 需自行下载
└── Cityscapes数据集/
    └── code.ipynb                 # 数据集需自行下载并修改路径
```

---

## 环境配置

### 硬件建议

| 子项目 | 最低配置 | 推荐配置 |
|--------|----------|----------|
| Oxford-IIIT Pet | CPU 可运行 | GPU + 8GB 显存 |
| Pascal VOC 2012 | GPU 4GB+ | GPU + 8GB 显存 |
| Cityscapes | GPU 8GB+ | GPU + 12GB+ 显存 |

### 依赖安装

建议使用 Python 3.9+ 与虚拟环境：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy matplotlib pillow tqdm opencv-python albumentations jupyter
```

> **说明**：Cityscapes 子项目额外依赖 `albumentations` 与 `opencv-python`；Oxford Pet 与 Pascal VOC 主要使用 `torchvision`。

### 验证 GPU

在 Notebook 中运行：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## 学习路线建议

```
Oxford-IIIT Pet  →  Pascal VOC 2012  →  Cityscapes
   (3 类 U-Net)      (21 类 U-Net)     (19 类 DeepLabV3+)
   像素准确率          mIoU 评估          预训练 + 复杂增强
```

| 阶段 | 重点掌握 |
|------|----------|
| **Pet** | `Dataset` 编写、标签映射、U-Net 结构、训练循环 |
| **VOC** | 多类别分割、`ignore_index`、混淆矩阵与 mIoU |
| **Cityscapes** | 预训练骨干网络、ASPP、分层学习率、混合精度训练 |

---

## 子项目概览

### 1. Oxford-IIIT Pet（入门）

**任务**：对宠物图像进行三分割——前景、背景、边界（trimap）。

**数据集**

- 规模：7390 张图像（约 600×400）
- 下载：[Kaggle - The Oxford-IIIT Pet Dataset](https://www.kaggle.com/datasets/devdgohil/the-oxfordiiit-pet-dataset)
- 本地路径：`data/images/` 与 `data/annotations/trimaps/`

**标签说明**

| 原始值 | 映射后 | 含义 |
|--------|--------|------|
| 1 | 0 | 前景（宠物） |
| 2 | 1 | 背景 |
| 3 | 2 | 边界 |

**模型**：U-Net（Encoder-Decoder + Skip Connection）

- 通道：`3 → 32 → 64 → 128 → 256 → 128 → 64 → 32 → 3`
- 核心模块：`DoubleConv`、`Down`（MaxPool + DoubleConv）、`Up`（ConvTranspose + 拼接 + DoubleConv）

**训练配置**

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 256 × 256 |
| Batch Size | 8 |
| 优化器 | Adam，lr = 1e-3 |
| Epochs | 10 |
| 损失函数 | CrossEntropyLoss |
| 数据划分 | 随机 80% / 20% |
| 评估指标 | Pixel Accuracy |

**数据增强**

- Resize（图像双线性，Mask 最近邻）
- 随机水平翻转（p = 0.5）
- ImageNet 标准化（mean = [0.485, 0.456, 0.406]，std = [0.229, 0.224, 0.225]）

**参考结果**（10 epoch）

- 验证集 Pixel Accuracy ≈ **88.5%**

**运行方式**

```bash
cd "Oxford-IIIT Pet数据集"
jupyter notebook code.ipynb
```

---

### 2. Pascal VOC 2012（进阶）

**任务**：21 类语义分割（1 个背景 + 20 个物体类别）。

**数据集**

- 规模：2913 张（trainval），官方划分 train 1464 / val 1449
- 下载：[Kaggle - Pascal VOC 2012](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)
- 本地目录结构：

```
data/
├── images/          # {id}.jpg
├── masks/           # {id}.png
└── splits/
    ├── train.txt
    ├── val.txt
    └── trainval.txt
```

**标签说明**

| 像素值 | 含义 |
|--------|------|
| 0 | 背景 |
| 1–20 | 20 个物体类别 |
| 255 | 忽略区域（不参与 loss 计算） |

> **注意**：VOC 标签 **不需要** 像 Pet 数据集那样减 1；损失函数使用 `ignore_index=255`。

**模型**：U-Net（与 Pet 相同结构，`num_classes=21`）

**训练配置**

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 256 × 256 |
| Batch Size | 16 |
| 优化器 | Adam，lr = 1e-4 |
| Epochs | 20 |
| 损失函数 | CrossEntropyLoss(ignore_index=255) |
| 数据划分 | 官方 train/val，或 trainval 随机 80/20 |
| 评估指标 | Loss、Pixel Accuracy、mIoU |

**mIoU 计算**

对每个类别计算 IoU，再对有效类别取平均；忽略标签 255 的像素。

**参考结果**（20 epoch，256×256，U-Net 从零训练）

- 验证集 Pixel Accuracy ≈ **74.4%**
- 验证集 mIoU ≈ **6.2%**

> mIoU 偏低是正常现象：21 类任务类别多、样本不均衡，且未使用预训练骨干。可通过预训练 Encoder、更大输入尺寸、更长训练等方式提升。

**Notebook 主要内容**

1. 数据探索与 Mask 可视化
2. `VOCDataset` 与 `SegmentationTransform`
3. U-Net 从零实现
4. 训练 / 验证循环与 mIoU 计算
5. Loss / Accuracy / mIoU 曲线
6. 多样本预测可视化

---

### 3. Cityscapes（高级）

**任务**：19 类城市场景语义分割（道路、车辆、行人等）。

**数据集**

- 原始分辨率：约 1024×2048
- 下载：[Kaggle - Cityscapes](https://www.kaggle.com/datasets/lqdisme/cityscapes)
- Notebook 中默认使用 Kaggle 路径，**本地运行需修改**：

```python
TRAIN_IMG_DIR  = "path/to/images/train"
TRAIN_MASK_DIR = "path/to/gtFine/train"
VAL_IMG_DIR    = "path/to/images/val"
VAL_MASK_DIR   = "path/to/gtFine/val"
```

Mask 文件命名规则：将 `_leftImg8bit.png` 替换为 `_gtFine_labelTrainIds.png`。

**模型**：DeepLabV3+

| 组件 | 说明 |
|------|------|
| Backbone | ResNet-50（ImageNet 预训练） |
| ASPP | 空洞空间金字塔池化，多尺度特征 |
| Decoder | 融合低层与高层特征，上采样至输入尺寸 |
| 输出 | 19 类逐像素 logits |

**训练配置**

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 512 × 512 |
| Batch Size | 16 |
| Epochs | 10 |
| 损失函数 | CrossEntropyLoss(ignore_index=255) |
| 优化器 | AdamW（分层学习率） |
| 学习率 | Backbone: 2e-5，ASPP: 3e-4，Decoder: 3e-4 |
| 学习率调度 | Poly LR：`(1 - step/max_iters)^0.9` |
| 混合精度 | `torch.amp.GradScaler` |
| 评估指标 | Loss、mIoU、Pixel Accuracy |

**数据增强**（Albumentations）

- RandomScale + RandomCrop
- HorizontalFlip
- ColorJitter、RandomBrightnessContrast
- GaussianBlur（低概率）

**训练策略**

- 初始阶段 **冻结 ResNet-50 骨干**，仅训练 ASPP 与 Decoder
- 提供 `unfreeze_backbone()` 用于后续微调

**参考结果**（10 epoch，512×512）

- 验证集 mIoU ≈ **42.0%**
- 验证集 Pixel Accuracy ≈ **88.0%**

**Notebook 主要内容**

1. `CityscapesDataset` 与城市目录遍历
2. Albumentations 增强流水线
3. DeepLabV3+ 完整实现（ResNet50 + ASPP + Decoder）
4. 分层优化器与 Poly LR
5. 混合精度训练
6. 19 类调色板可视化与预测对比

---

## 核心概念

### 语义分割 vs 其他任务

| 任务 | 输出 | 示例 |
|------|------|------|
| 图像分类 | 整张图一个类别 | "这是一只猫" |
| 目标检测 | 边界框 + 类别 | 框出猫的位置 |
| **语义分割** | **每个像素的类别** | 逐像素标注猫/背景 |
| 实例分割 | 区分同类不同个体 | 猫 A、猫 B 分开 |

### U-Net 架构

```
输入图像
   ↓
[Encoder: 逐层下采样，提取语义特征]
   ↓
[Bottleneck: 最深层特征]
   ↓
[Decoder: 逐层上采样 + Skip Connection 融合细节]
   ↓
1×1 Conv → 逐像素类别预测
```

Skip Connection 将 Encoder 同尺度特征拼接到 Decoder，帮助恢复边界与细节。

### 评估指标

**Pixel Accuracy（像素准确率）**

```
正确分类像素数 / 总像素数
```

简单直观，但在类别不均衡时容易偏高。

**mIoU（Mean Intersection over Union）**

```
IoU_c = |预测∩真值| / |预测∪真值|
mIoU = mean(IoU_c)  # 对所有有效类别取平均
```

语义分割更常用的指标，对少样本类别更敏感。

### Mask Resize 注意事项

Resize 图像时使用双线性插值；Resize **Mask 时必须使用最近邻插值（NEAREST）**，否则会产生无效的中间标签值，污染训练。

```python
# 正确做法
mask = TF.resize(mask, (H, W), interpolation=Image.NEAREST)
```

---

## 常见问题

### Q1：Pascal VOC 的 images/masks 目录为空？

从 [Kaggle](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012) 下载后，将图像与 Mask 分别放入 `data/images/` 与 `data/masks/`。`data/splits/` 中的划分文件已包含在项目中。

### Q2：Cityscapes 报路径找不到？

Notebook 默认路径为 Kaggle 环境。请根据本机数据集位置修改 `TRAIN_IMG_DIR`、`TRAIN_MASK_DIR` 等变量。

### Q3：显存不足（CUDA OOM）？

可尝试：

- 减小 `batch_size`
- 减小输入尺寸（如 VOC 256→128，Cityscapes 512→384）
- 关闭其他占用 GPU 的程序

### Q4：训练 loss 下降但 mIoU 很低？

常见原因：

- 类别极度不均衡（背景占大多数像素）
- 仅看 Pixel Accuracy 会掩盖少数类表现
- 建议同时观察 mIoU 与可视化预测结果

### Q5：如何在本地保存模型？

在训练循环结束后添加：

```python
torch.save(model.state_dict(), "unet_voc.pth")
```

加载：

```python
model.load_state_dict(torch.load("unet_voc.pth", map_location=device))
model.eval()
```

---

## 参考资料

### 论文

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597)
- [Encoder-Decoder with Atrous Separable Convolution (DeepLabV3+) (2018)](https://arxiv.org/abs/1802.02611)
- [The PASCAL Visual Object Classes Challenge (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [The Cityscapes Dataset for Semantic Urban Scene Understanding (2016)](https://arxiv.org/abs/1604.01685)

### 数据集链接

| 数据集 | 链接 |
|--------|------|
| Oxford-IIIT Pet | https://www.kaggle.com/datasets/devdgohil/the-oxfordiiit-pet-dataset |
| Pascal VOC 2012 | https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012 |
| Cityscapes | https://www.kaggle.com/datasets/lqdisme/cityscapes |

### 相关库文档

- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [torchvision.transforms.functional](https://pytorch.org/vision/stable/transforms.html)
- [Albumentations](https://albumentations.ai/docs/)

---

## 后续扩展方向

- 使用 ResNet / EfficientNet 作为 U-Net 的 Encoder（预训练迁移学习）
- 尝试 FCN、PSPNet、SegFormer 等其他架构
- 引入 Dice Loss、Focal Loss 等处理类别不均衡
- 使用 TensorBoard 或 wandb 记录实验
- 导出 ONNX / TorchScript 进行部署推理

---

*本项目用于语义分割学习与实验，欢迎在此基础上继续改进与扩展。*
