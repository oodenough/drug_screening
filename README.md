# 🧬 基于大数据分析的药物筛选系统

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2025.03-green.svg)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](https://github.com)

> 基于深度学习的药物虚拟筛选系统 - 课程设计项目  
> 完成日期：2025年12月23日  
> 开发者：falunwen123go

---

## 📋 项目概述

本项目是一个完整的**基于深度学习的药物虚拟筛选系统**，使用真实的 **MoleculeNet** 数据集进行分子活性预测。系统实现了从数据处理、特征提取、模型训练、评估到Web界面部署的**完整深度学习流水线**。

### ✨ 核心功能
- 🧬 **分子特征提取**: ECFP指纹 (1024维) + 12种分子描述符
- 🤖 **深度学习模型**: DrugPredictorMLPv2 (304K参数，增强正则化)
- 📊 **双任务预测**: BBBP血脑屏障分类 + ESOL溶解度回归
- 🖥️ **Web交互界面**: Streamlit界面，支持单分子/批量预测
- ⚡ **GPU加速训练**: CUDA 11.8 + 早停机制 + 学习率调度
- 🔍 **智能批量筛选**: 自动检测SMILES列 + Lipinski规则过滤
- 📈 **完整评估体系**: ROC曲线、混淆矩阵、训练历史可视化

### 🎯 项目亮点
- ✅ **解决过拟合问题**: 通过分层采样 + 渐进式Dropout，AUC-ROC从70%提升至91%
- ✅ **美观的训练输出**: 表格式进度显示，实时指标监控
- ✅ **完整的项目文档**: 实验报告、运行指南、更新日志
- ✅ **真实数据验证**: MoleculeNet官方数据集，2039个BBBP样本 + 1128个ESOL样本

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

## 🎯 模型性能

### ⭐ 最新结果 (优化后)

#### BBBP 血脑屏障穿透性预测 (分类)

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **Accuracy** | 61.27% | **87.25%** | ⬆️ +43% |
| **Precision** | 59.09% | **89.63%** | ⬆️ +52% |
| **Recall** | 85.05% | **94.23%** | ⬆️ +11% |
| **F1-Score** | 69.73% | **91.87%** | ⬆️ +33% |
| **AUC-ROC** | 65.90% | **90.71%** | ⬆️ +38% |

#### ESOL 水溶解度预测 (回归)

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **RMSE** | 0.7570 | **0.5528** | ⬇️ -27% |
| **MAE** | 0.5687 | **0.4217** | ⬇️ -26% |
| **R²** | 0.4551 | **0.6802** | ⬆️ +49% |
| **Pearson r** | 0.7316 | **0.8314** | ⬆️ +14% |

### 🔧 关键优化措施

1. **分层采样**: 使用`StratifiedShuffleSplit`解决数据分布不均（训练集82% → 76.5%正例）
2. **增强正则化**: 渐进式Dropout (0.5→0.6→0.7) + 输入Dropout + 权重衰减1e-3
3. **更小网络**: 256→128→64 (304K参数，相比690K减少56%)
4. **训练优化**: 更长训练轮数(200 epochs) + 更大耐心值(patience=25/35)
5. **Xavier初始化**: 稳定训练过程

**最佳验证损失**: Epoch 21 (优化前在Epoch 1，严重过拟合)

---

## 🧠 模型架构

### DrugPredictorMLPv2 (优化版)

```
输入层 (1024) → [ECFP分子指纹]
    ↓
Input Dropout (0.25) → [输入层正则化]
    ↓
隐藏层1 (256) → BatchNorm → ReLU → Dropout(0.5)
    ↓
隐藏层2 (128) → BatchNorm → ReLU → Dropout(0.6) [渐进式]
    ↓
隐藏层3 (64)  → BatchNorm → ReLU → Dropout(0.7) [渐进式]
    ↓
输出层 (1) → [Sigmoid/Linear] → 预测结果

总参数量: 304,385 (相比v1减少56%)
```

**关键改进**：
- ✅ 更小的网络结构 (256→128→64)，降低过拟合风险
- ✅ 输入Dropout (dropout * 0.5)，防止输入层过拟合
- ✅ 渐进式Dropout (0.5→0.6→0.7)，随层深度增加
- ✅ Xavier权重初始化，稳定训练
- ✅ 更强的权重衰减 (1e-3)，增强L2正则化
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
│   └── data_loader.py         # MoleculeNet数据加载 + 分层采样
├── features/                  # 特征工程模块
│   └── molecular_features.py  # ECFP指纹 + 分子描述符
├── models/                    # 模型定义模块
│   └── drug_models.py         # DrugPredictorMLPv2 + 其他模型
├── training/                  # 训练模块
│   ├── trainer.py             # 标准训练器
│   └── trainer_v2.py          # 优化版训练器（表格输出）
├── evaluation/                # 评估模块
│   ├── metrics.py             # 评估指标 + 可视化
│   └── figures/               # 评估图表
│       ├── bbbp_roc_curve.png
│       ├── bbbp_confusion_matrix.png
│       └── training_history.png
├── inference/                 # 推理模块
│   └── predictor.py           # 单分子预测 + 批量筛选
├── web/                       # Web界面
│   └── app.py                 # Streamlit应用
├── saved_models/              # 训练好的模型
│   ├── bbbp_model.pth         # 血脑屏障模型
│   └── esol_model.pth         # 溶解度模型
├── train_model.py             # 主训练脚本 ⭐
├── 实验报告-整合版.md          # 课程设计实验报告 📄
├── 运行指南.md                 # 详细使用指南 📖
├── 更新日志.md                 # 版本更新记录 📝
└── README.md                  # 本文档
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda create -n drug_screen python=3.9 -y
conda activate drug_screen

# 安装RDKit (必须用conda)
conda install -c conda-forge rdkit -y

# 安装PyTorch (GPU版本 - CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install deepchem pandas scikit-learn matplotlib seaborn streamlit tqdm pyyaml
```

**💡 提示**: 详细配置指南请参考 [运行指南.md](运行指南.md)

### 2. 训练模型

```bash
# 训练BBBP和ESOL两个模型
python train_model.py
```

**预期输出**:
```
============================================================
训练BBBP模型 - 血脑屏障穿透性预测
============================================================

数据集统计:
  训练集: 1631 样本 (正例: 76.52%)
  验证集: 204 样本 (正例: 76.47%)
  测试集: 204 样本 (正例: 76.47%)

Epoch   1/100 ━━━━━━━━━━━━━━━━━━━━━ 100%
  Train Loss: 0.4821 | Train Acc: 0.7654 | Val Loss: 0.3421 ✓

...

BBBP 测试集评估结果:
  Accuracy:  0.8725
  AUC-ROC:   0.9071
```

### 3. 启动Web界面

```bash
streamlit run web/app.py
```

浏览器自动打开 http://localhost:8501

### 4. 使用预测API

```python
from inference.predictor import DrugPredictor

predictor = DrugPredictor(
    model_path='saved_models/bbbp_model.pth',
    device='cuda'
)

# 预测阿司匹林
result = predictor.predict_with_properties('CC(=O)OC1=CC=CC=C1C(=O)O')
print(f"BBB穿透概率: {result['score']:.2%}")
print(f"分子量: {result['properties']['MW']:.2f}")
```

---

## � 主要文档

| 文档 | 说明 |
|------|------|
| [实验报告-整合版.md](实验报告-整合版.md) | 📄 课程设计完整实验报告（技术方案、实验过程、结果分析） |
| [运行指南.md](运行指南.md) | 📖 详细的环境配置和使用指南 |
| [更新日志.md](更新日志.md) | 📝 项目版本更新记录 |
| [README.md](README.md) | 📋 本文档（项目概览） |

---

## 🌟 功能展示

### Web界面功能

1. **单分子预测**
   - 输入SMILES字符串
   - 显示分子2D结构
   - 预测BBB穿透概率
   - 计算12种分子性质
   - **Lipinski规则检查**（类药性评估）

2. **批量筛选**
   - 上传CSV文件（自动检测SMILES列）
   - 设置Top-K筛选数量
   - 应用Lipinski过滤
   - 显示Top分子结构
   - 下载筛选结果

3. **Lipinski五规则**（类药性评估）
   - ✅ 分子量 ≤ 500 Da
   - ✅ LogP ≤ 5
   - ✅ 氢键供体 ≤ 5
   - ✅ 氢键受体 ≤ 10

### 命令行训练特性

- ✅ 美观的表格式输出
- ✅ 实时训练指标显示
- ✅ 早停机制防止过拟合
- ✅ 自动保存最佳模型
- ✅ GPU加速训练
- ✅ 学习率自动调度
- ✅ 完整的评估报告

---

## 🔬 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| **编程语言** | Python | 3.9.25 |
| **深度学习** | PyTorch | 2.7.1+cu118 |
| **GPU加速** | CUDA | 11.8 |
| **分子处理** | RDKit | 2025.03.5 |
| **数据集** | DeepChem | 2.8.0 |
| **Web框架** | Streamlit | 最新版 |
| **数据分析** | NumPy, Pandas, Scikit-learn | 最新版 |
| **可视化** | Matplotlib, Seaborn | 最新版 |

---

## 📈 项目亮点

### 1. 解决过拟合问题

**问题识别**:
- 原始模型在Epoch 1就达到最佳验证损失
- 训练集与验证集数据分布严重不均（82% vs 54%正例）

**解决方案**:
1. ✅ 分层采样 (`StratifiedShuffleSplit`)
2. ✅ 更小的网络结构（参数量减少56%）
3. ✅ 渐进式Dropout + 输入Dropout
4. ✅ 更强的权重衰减（1e-3）
5. ✅ Xavier初始化

**效果**:
- AUC-ROC: 70% → 91% (+30%)
- 最佳验证损失: Epoch 1 → Epoch 21
- 模型泛化能力显著提升

### 2. 美观的训练输出

```
Epoch   21/100 ━━━━━━━━━━━━━━━━━━━━━ 100%
  Train Loss: 0.1234 | Train Acc: 0.8921 | Val Loss: 0.2471 ✓
```

- 表格式进度条
- 实时指标显示
- 最佳模型标记

### 3. 智能批量筛选

- 自动检测CSV中的SMILES列
- 智能默认选择包含"smiles"的列
- Top-K排序 + Lipinski过滤
- 分子结构可视化

---

## 📊 支持的数据集

| 数据集 | 样本数 | 任务类型 | 描述 |
|-------|--------|---------|------|
| **BBBP** ⭐ | 2,039 | 二分类 | 血脑屏障穿透性预测 |
| **ESOL** ⭐ | 1,128 | 回归 | 水溶解度预测 (log mol/L) |
| Tox21 | 7,831 | 多任务分类 | 12种毒性指标 |
| BACE | 1,513 | 分类/回归 | β-分泌酶抑制活性 |

**⭐ 标记为已实现**

---

## 📈 评估图表

训练完成后自动生成的可视化图表：

| 模型 | 图表文件 | 说明 |
|------|---------|------|
| **BBBP** | `bbbp_training_history.png` | 训练/验证损失和准确率曲线 |
| **BBBP** | `bbbp_roc_curve.png` | ROC曲线（AUC=0.91） |
| **BBBP** | `bbbp_confusion_matrix.png` | 混淆矩阵（TP/FP/TN/FN） |
| **ESOL** | `esol_training_history.png` | 训练/验证损失曲线 |
| **ESOL** | `esol_scatter.png` | 预测值vs真实值散点图 |

所有图表保存在 [evaluation/figures/](evaluation/figures/) 目录。

---

## 🏆 项目成果总结

✅ **完整的深度学习流水线**: 数据加载 → 特征提取 → 模型训练 → 评估 → Web部署  
✅ **优秀的模型性能**: BBBP AUC-ROC 91%, ESOL R² 68%  
✅ **解决实际问题**: 通过分层采样和增强正则化解决过拟合  
✅ **用户友好界面**: Streamlit Web应用 + 批量筛选功能  
✅ **完善的文档体系**: 实验报告、运行指南、更新日志  
✅ **GPU加速**: CUDA训练加速，推理效率高  
✅ **可扩展架构**: 模块化设计，易于添加新功能

---

## 📖 参考文献

[1] Wu Z, Ramsundar B, Feinberg E N, et al. **MoleculeNet: a benchmark for molecular machine learning**. *Chemical Science*, 2018, 9(2): 513-530.

[2] Rogers D, Hahn M. **Extended-connectivity fingerprints**. *Journal of Chemical Information and Modeling*, 2010, 50(5): 742-754.

[3] Ramsundar B, Eastman P, Walters P, et al. **Deep Learning for the Life Sciences**. O'Reilly Media, 2019.

[4] Lipinski CA, Lombardo F, Dominy BW, et al. **Experimental and computational approaches to estimate solubility and permeability**. *Advanced Drug Delivery Reviews*, 1997, 23(1-3): 3-25.

---

## 📞 联系方式

- **开发者**: falunwen123go
- **项目仓库**: [GitHub - drug_screening](https://github.com/falunwen123go/drug_screening)
- **问题反馈**: 请提交Issue

---

## 📄 许可证

本项目采用 **MIT 许可证**。详见 [LICENSE](LICENSE) 文件。

本项目仅用于**学习和研究目的**。

---

## 🙏 致谢

感谢以下开源项目的支持：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [RDKit](https://www.rdkit.org/) - 化学信息学工具包
- [DeepChem](https://deepchem.io/) - 分子机器学习库
- [MoleculeNet](http://moleculenet.ai/) - 分子数据集基准
- [Streamlit](https://streamlit.io/) - Web应用框架

---

<p align="center">
  <strong>⭐ 如果这个项目对你有帮助，请给一个Star！⭐</strong>
</p>

<p align="center">
  Made with ❤️ by falunwen123go | 课程设计项目 | 2025年12月23日
</p>
