# CHARLS-Stacking-Diabetes-Prediction

基于 **CHARLS 中国健康与养老追踪调查数据库**，使用 **Stacking 集成学习（逻辑回归 / 随机森林 / XGBoost / LightGBM）** 构建中老年人 2 型糖尿病风险预测模型，并引入 **SHAP 可解释性分析**揭示关键风险因素。

> **模型 AUC = 0.8895**，适用于三级医院（高特异度）与基层医院（高灵敏度）双场景阈值优化。

---

## 目录

- [亮点](#亮点)
- [数据来源](#数据来源)
- [方法](#方法)
- [结果](#结果)
- [SHAP 可解释性](#shap-可解释性)
- [文件说明](#文件说明)
- [使用方式](#使用方式)
- [依赖](#依赖)
- [引用](#引用)

---

## 亮点

- **Stacking 集成**：4 个异质基模型 + 逻辑回归元模型，5 折交叉验证
- **三级阈值优化**：加权约登指数适配基层医院（高灵敏度）与三级医院（高特异度）
- **SHAP 可解释性**：4 基模型 SHAP 按元模型系数加权融合，剔除标签泄露特征
- **Bootstrap AUC 置信区间**：1000 次重抽样，稳定性可量化
- **AUC 0.8895**，灵敏度最高 0.858（基层场景），特异度最高 0.872（三级场景）

---

## 数据来源

**CHARLS**（China Health and Retirement Longitudinal Study，中国健康与养老追踪调查）

- 覆盖全国 45 岁及以上中老年人
- 包含人口学、身体状况、实验室检查（血常规、生化）、生活方式等多维度变量
- 本项目使用 Wave 3（2015）和 Wave 4（2018）数据

---

## 方法

### 模型架构

```
训练集 (80%)
    │
    ├── 基模型 1: 逻辑回归 (C=0.07, l2)
    ├── 基模型 2: 随机森林 (180棵树, max_depth=11)
    ├── 基模型 3: XGBoost (lr=0.05, max_depth=3)
    ├── 基模型 4: LightGBM (lr=0.03, num_leaves=25)
    │
    └── 元模型: 逻辑回归 (C=0.1, l2)
          │
          └── Stacking 集成预测
```

### 特征工程

- 基础特征：人口学、社会经济、生活方式、疾病史、身体测量
- 衍生特征：`glu_hbalc_ratio`（血糖/糖化比值）、`tyg_bmi_squared`、`metabolic_abnormality`（代谢异常标记）
- 标签泄露剔除：`chronic_num`（慢病数量）不参与 SHAP 分析

### 三级阈值场景

| 场景 | 权重 (灵敏度 : 特异度) | 适用场景 |
|------|------------------------|----------|
| 标准平衡 | 1.0 : 1.0 | 一般筛查 |
| 基层医院 | 1.2 : 0.8 | 首诊、高危人群初筛，**不漏诊优先** |
| 三级医院 | 0.8 : 1.2 | 确诊、专科复核，**误诊控制优先** |

---

## 结果

### 模型性能对比

| 模型 | AUC | 灵敏度 | 特异度 | 约登指数 |
|------|-----|--------|--------|----------|
| **Stacking 集成** | **0.8894** | 0.844 | 0.767 | 0.611 |
| XGBoost | 0.8869 | 0.860 | 0.744 | 0.604 |
| LightGBM | 0.8874 | 0.765 | 0.848 | **0.613** |
| 逻辑回归 | 0.8742 | 0.806 | 0.796 | 0.602 |
| 随机森林 | 0.8739 | 0.830 | 0.750 | 0.581 |

### 三场景阈值性能（Stacking）

| 场景 | 阈值 | 灵敏度 | 特异度 | F1 | 约登指数 |
|------|------|--------|--------|-----|----------|
| 标准平衡 | 0.129 | 0.844 | 0.767 | 0.426 | 0.611 |
| 基层医院 | 0.108 | **0.858** | 0.748 | 0.413 | 0.606 |
| 三级医院 | 0.228 | 0.721 | **0.872** | **0.500** | 0.593 |

---

## SHAP 可解释性

### 融合权重

| 基模型 | 元模型系数归一化权重 |
|--------|----------------------|
| XGBoost | 0.2931 |
| 逻辑回归 | 0.2896 |
| LightGBM | 0.2790 |
| 随机森林 | 0.1383 |

### 全局特征重要性 Top 10

| 排名 | 特征 | 平均 |SHAP| 值 |
|------|------|---------------|
| 1 | bl_hbalc（糖化血红蛋白） | 0.4115 |
| 2 | frailtya（虚弱指数） | 0.1799 |
| 3 | bl_glu（空腹血糖） | 0.1075 |
| 4 | dyslipe_是（血脂异常） | 0.1069 |
| 5 | rgrip（右手握力） | 0.0641 |
| 6 | region_东部 | 0.0631 |
| 7 | frailtyb（虚弱指数 b） | 0.0595 |
| 8 | bl_ldl（低密度脂蛋白） | 0.0560 |
| 9 | hrural_城市_True | 0.0523 |
| 10 | retire（退休） | 0.0513 |

> 已剔除标签泄露特征 `chronic_num`，糖代谢指标（hbalc + glu）合计贡献 **0.519**，与临床指南一致。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `最终版（三级基层约登-阈值）_优化版.py` | **主程序**：Stacking 训练 + 三级阈值评估 + SHAP 分析 |
| `最终版（三级基层约登-阈值）.py` | 原始版（不含 SHAP） |
| `数据预处理_优化版.py` | 数据清洗与特征工程 |
| `源数据读取特征.py` | CHARLS 原始数据读取与特征提取 |
| `stacking_ensemble_4models.py` | 4 模型 Stacking 集成（早期版） |
| `logistic.py` / `random_forest.py` / `xgboost_1.py` / `lightgbm_1.py` | 各基模型单模训练与调参 |
| `lasso_1.py` / `lasso_optimized.py` | LASSO 特征选择 |
| `gen_shap_docx.py` | 生成 SHAP 分析 Word 报告 |
| `requirements.txt` | Python 依赖 |
| `tuned_stacking_results/` | 输出目录（指标 CSV、ROC 图、SHAP 图、模型 pkl） |

---

## 使用方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行主程序（训练 + 评估 + SHAP）
python "最终版（三级基层约登-阈值）_优化版.py"
```

输出文件位于 `tuned_stacking_results/`：
- `all_metrics.csv` — 全部模型 × 场景指标
- `experiment_log.txt` — 运行日志
- `shap_global_importance.png` — SHAP 全局特征重要性图
- `stacking_model_with_scene_thresholds.pkl` — 训练好的模型

---

## 依赖

- Python >= 3.8
- pandas, numpy
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- shap >= 0.40.0
- matplotlib, seaborn
- python-docx >= 1.0.0

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{charls-stacking-diabetes,
  author = {Your Name},
  title = {基于CHARLS的Stacking集成糖尿病预测模型},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/sjt1818/-CHARLS-Stacking-Diabetes-Prediction}
}
```

---

**免责声明**：本项目仅供学习研究，不构成医疗建议。
