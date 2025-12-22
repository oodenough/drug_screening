# 💊 基于大数据分析的药物筛选系统

## Drug Screening System Based on Big Data Analysis

---

## 📋 项目概述

本项目是一个完整的**基于深度学习的药物筛选系统课程设计**，使用真实的 **MoleculeNet** 数据集进行分子活性预测。系统涵盖了从数据处理、特征提取、模型训练到Web界面部署的完整流程。

### 核心功能
- 🧬 **分子特征提取**: 使用ECFP (扩展连接性指纹) 和Morgan指纹
- 🤖 **深度学习模型**: 多层感知机(MLP)神经网络
- 📊 **多任务预测**: 支持分类任务(BBBP)和回归任务(ESOL)
- 🖥️ **Web界面**: 基于Streamlit的交互式预测界面
- ⚡ **GPU加速**: 支持CUDA加速训练

---

## 📊 使用的真实MoleculeNet数据集

### 1. BBBP (Blood-Brain Barrier Penetration)
| 属性 | 说明 |
|------|-----|
| **任务类型** | 二分类 |
| **预测目标** | 分子能否穿透血脑屏障 |
| **训练集** | 1,631 个分子 (正例 82.22%) |
| **验证集** | 204 个分子 (正例 54.90%) |
| **测试集** | 204 个分子 (正例 52.45%) |

### 2. ESOL (Aqueous Solubility)
| 属性 | 说明 |
|------|-----|
| **任务类型** | 回归 |
| **预测目标** | 分子水溶解度 (log mol/L) |
| **训练集** | 902 个分子 |
| **验证集** | 113 个分子 |
| **测试集** | 113 个分子 |
| **溶解度范围** | -4.23 至 2.15 log mol/L |

---

## 🎯 模型训练结果

### BBBP 血脑屏障穿透性预测 (分类)

| 指标 | 测试集结果 |
|------|-----------|
| **Accuracy** | 61.27% |
| **Precision** | 59.09% |
| **Recall** | 85.05% |
| **F1-Score** | 69.73% |
| **AUC-ROC** | 65.90% |

### ESOL 水溶解度预测 (回归)

| 指标 | 测试集结果 |
|------|-----------|
| **RMSE** | 0.7570 log mol/L |
| **MAE** | 0.5687 log mol/L |
| **R²** | 0.4551 |
| **Pearson r** | 0.7316 |

---

## 🧠 模型架构 (MLP神经网络)

```
输入层 (1024) → [ECFP分子指纹]
    ↓
隐藏层1 (512) → BatchNorm → ReLU → Dropout(0.5)
    ↓
隐藏层2 (256) → BatchNorm → ReLU → Dropout(0.5)
    ↓
隐藏层3 (128) → BatchNorm → ReLU → Dropout(0.5)
    ↓
输出层 (1) → [分类: Sigmoid / 回归: 直接输出]

总参数量: 690,945
```

---

## 📂 项目结构

```
drug/
├── data/                      # 数据处理模块
│   └── data_loader.py         # MoleculeNet数据加载器
├── features/                  # 特征工程模块
│   └── feature_extraction.py  # 分子指纹提取
├── models/                    # 模型定义模块
│   └── drug_models.py         # MLP神经网络模型
├── training/                  # 训练模块
│   └── trainer.py             # 训练器(支持早停)
├── evaluation/                # 评估模块
│   ├── metrics.py             # 评估指标计算
│   └── figures/               # 评估图表输出
│       ├── bbbp_*.png         # BBBP模型评估图
│       └── esol_*.png         # ESOL模型评估图
├── inference/                 # 推理模块
│   └── predictor.py           # 模型推理器
├── web/                       # Web界面
│   └── app.py                 # Streamlit应用
├── saved_models/              # 保存的模型
│   ├── bbbp_model.pth         # BBBP分类模型
│   └── esol_model.pth         # ESOL回归模型
├── train_full.py              # 完整训练脚本
├── train_model.py             # 主训练脚本
└── README.md                  # 本文档
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建conda环境
conda create -n drug_screen python=3.9 -y
conda activate drug_screen

# 安装RDKit
conda install -c conda-forge rdkit -y

# 安装PyTorch (GPU版本 CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install deepchem pandas scikit-learn matplotlib seaborn streamlit tqdm
```

### 2. 训练模型

```bash
# 运行训练脚本
python train_model.py
```

### 3. 启动Web界面

```bash
streamlit run web/app.py
```

浏览器访问: http://localhost:8501

### 3. 完整训练流程

```bash
# 训练BBBP和ESOL两个模型
python train_full.py
```

---

## 📊 支持的MoleculeNet数据集

| 数据集 | 样本数 | 任务类型 | 描述 |
|-------|--------|---------|------|
| **BBBP** | 2,039 | 二分类 | 血脑屏障穿透性预测 |
| **ESOL** | 1,128 | 回归 | 水溶解度预测 |
| Tox21 | 7,831 | 多任务分类 | 12种毒性指标 |
| BACE | 1,513 | 分类/回归 | β-分泌酶抑制 |

---

## 🔬 使用示例

### 预测单个分子

```python
from features.feature_extraction import MolecularFeaturizer
from models.drug_models import DrugPredictorMLP
import torch

# 加载模型
model = DrugPredictorMLP(input_dim=1024, hidden_dims=[512, 256, 128])
model.load_state_dict(torch.load('saved_models/bbbp_model.pth'))
model.eval()

# 提取特征
featurizer = MolecularFeaturizer()
smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # 阿司匹林
features = featurizer.extract_features(smiles)

# 预测
with torch.no_grad():
    logits = model(torch.tensor(features).float().unsqueeze(0))
    prob = torch.sigmoid(logits).item()
    print(f"BBB穿透概率: {prob:.4f}")
```

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **编程语言** | Python 3.9 |
| **深度学习** | PyTorch 2.7.1 + CUDA 11.8 |
| **分子处理** | RDKit 2025.03.5 |
| **数据集** | DeepChem 2.8.0 (MoleculeNet) |
| **可视化** | Matplotlib, Seaborn |
| **Web界面** | Streamlit |
| **GPU** | NVIDIA RTX 3050 Ti |

---

## 📈 评估图表

训练完成后生成的图表:

| 模型 | 图表 |
|------|------|
| BBBP | `bbbp_training_history.png` - 训练曲线 |
| BBBP | `bbbp_roc_curve.png` - ROC曲线 |
| BBBP | `bbbp_confusion_matrix.png` - 混淆矩阵 |
| ESOL | `esol_training_history.png` - 训练曲线 |
| ESOL | `esol_scatter.png` - 预测vs真实值 |

---

## 📚 参考文献

1. Wu, Z., et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning*. Chemical Science.
2. Rogers, D., & Hahn, M. (2010). *Extended-Connectivity Fingerprints*. Journal of Chemical Information and Modeling.
3. Ramsundar, B., et al. (2019). *Deep Learning for the Life Sciences*. O'Reilly Media.

---

## 📝 许可

本项目仅用于学习和研究目的。

---

**课程设计项目** | **完成日期**: 2025年12月22日
