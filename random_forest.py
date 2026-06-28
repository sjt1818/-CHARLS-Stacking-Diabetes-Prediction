import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, make_scorer, recall_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform

# ========== 环境配置 ==========
temp_dir = os.path.join(os.getcwd(), "rf_temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = temp_dir
os.environ["PYTHONUTF8"] = "1"

# 全局配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 核心代谢指标定义
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']

# 自定义评分函数
def specificity_score(y_true, y_pred):
    """计算特异度（真阴性率）"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# 创建scikit-learn兼容的评分器
scoring = {
    'roc_auc': 'roc_auc',
    'sensitivity': make_scorer(recall_score, pos_label=1),
    'specificity': make_scorer(specificity_score)
}


# ========================
# 1. 数据加载与预处理
# ========================
def load_and_preprocess_data(train_path, test_path, label_col='diabe'):
    # 兼容多编码读取
    def safe_read(file):
        for encoding in ['utf-8', 'gbk']:
            try:
                return pd.read_csv(file, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解析文件：{file}")
    
    train_df = safe_read(train_path)
    test_df = safe_read(test_path)
    print(f"✅ 成功读取数据集：")
    print(f"   - 训练集：{os.path.basename(train_path)}（{train_df.shape[0]}行 × {train_df.shape[1]}列）")
    print(f"   - 测试集：{os.path.basename(test_path)}（{test_df.shape[0]}行 × {test_df.shape[1]}列）")

    # 验证核心代谢指标
    missing_metabolic = [col for col in CORE_METABOLIC if col not in train_df.columns]
    if missing_metabolic:
        print(f"⚠️  警告：核心代谢指标缺失：{missing_metabolic}")
    else:
        print(f"✅ 核心代谢指标完整：{CORE_METABOLIC}")

    # 分离特征与标签
    X_train = train_df.drop(label_col, axis=1)
    y_train = train_df[label_col].copy()
    X_test = test_df.drop(label_col, axis=1)
    y_test = test_df[label_col].copy()

    # 标签处理
    if y_train.dtype == 'object':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        print(f"✅ 标签编码：{le.classes_} → [0, 1]")
    else:
        y_train = y_train.astype(int).clip(0, 1)
        y_test = y_test.astype(int).clip(0, 1)
        print(f"✅ 标签验证：数值型（0=非糖尿病，1=糖尿病）")

    # 特征分类与预处理
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [col for col in X_train.select_dtypes(include=['object']) if col not in num_cols]
    
    print(f"\n✅ 特征分类：")
    print(f"   - 数值特征：{len(num_cols)}个（含代谢指标）")
    print(f"   - 分类特征：{len(cat_cols)}个（示例：{cat_cols[:3]}...）")

    # 缺失值填充
    num_imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    # 分类特征编码
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        # 对齐训练集与测试集特征
        X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    # 数据验证
    print(f"\n======= 数据验证 =======")
    print(f"训练集样本：{len(X_train)} | 患病率：{y_train.mean():.2%}（{y_train.sum()}例）")
    print(f"测试集样本：{len(X_test)} | 患病率：{y_test.mean():.2%}（{y_test.sum()}例）")
    print(f"最终特征数：{X_train.shape[1]} | 缺失值：{'无' if X_train.isna().sum().sum() == 0 else '有'}")

    return X_train, y_train, X_test, y_test, train_df, test_df


# ========================
# 2. 模型训练与超参数优化
# ========================
def train_optimized_rf(X_train, y_train):
    param_dist = {
        'n_estimators': randint(150, 300),
        'max_depth': randint(6, 10),
        'min_samples_split': randint(30, 60),
        'min_samples_leaf': randint(8, 15),
        'max_features': ['sqrt'],
        'class_weight': ['balanced', {0:1, 1:9}],
        'bootstrap': [True],
        'min_impurity_decrease': uniform(0.0, 0.005)
    }

    base_rf = RandomForestClassifier(random_state=42, n_jobs=1)

    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        random_state=42,
        verbose=1
    )

    start_time = time.time()
    print(f"\n📊 开始随机搜索（{len(param_dist)}个参数，30次迭代）...")
    random_search.fit(X_train, y_train)
    print(f"✅ 搜索完成（耗时：{time.time()-start_time:.2f}秒）")
    print(f"   - 最佳参数：{random_search.best_params_}")
    print(f"   - 交叉验证AUC：{random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


# ========================
# 3. 模型评估与阈值优化
# ========================
def evaluate_model(model, X_test, y_test, save_dir):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': auc,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'best_threshold': best_threshold
    }

    print("\n======= 模型评估结果 =======")
    print(f"AUC：{metrics['auc']:.4f} | 最佳阈值：{best_threshold:.3f}")
    print(f"灵敏度：{metrics['sensitivity']:.4f} | 特异度：{metrics['specificity']:.4f}")
    print(f"阳性预测值：{metrics['ppv']:.4f} | 阴性预测值：{metrics['npv']:.4f}")
    print("\n📊 混淆矩阵：")
    print(f"非糖尿病正确：{tn} | 误诊：{fp}")
    print(f"糖尿病漏诊：{fn} | 正确识别：{tp}")
    print("\n📋 分类报告：")
    print(classification_report(
        y_test, y_pred, target_names=['非糖尿病', '糖尿病'], digits=4
    ))

    return metrics, y_pred_proba, y_pred, fpr, tpr


# ========================
# 4. 可视化功能
# ========================
def plot_roc_curve(fpr, tpr, auc, best_threshold, save_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#E63946', linewidth=3, label=f'随机森林 (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='随机猜测')
    # 标注最佳阈值点
    best_idx = np.argmin(np.abs(thresholds - best_threshold))
    plt.scatter(fpr[best_idx], tpr[best_idx], color='#A23B72', s=100, zorder=5)
    plt.annotate(
        f'最佳阈值: {best_threshold:.3f}\n灵敏度: {tpr[best_idx]:.3f}\n特异度: {1-fpr[best_idx]:.3f}',
        xy=(fpr[best_idx], tpr[best_idx]),
        xytext=(fpr[best_idx]+0.1, tpr[best_idx]-0.2),
        arrowprops=dict(arrowstyle='->', color='#A23B72')
    )
    plt.xlabel('假阳性率（1-特异度）')
    plt.ylabel('真阳性率（灵敏度）')
    plt.title('随机森林ROC曲线（优化阈值）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rf_roc_curve.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存：{os.path.join(save_dir, 'rf_roc_curve.png')}")


def plot_feature_importance(model, X_test, y_test, save_dir):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1
    )
    importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': result.importances_mean,
        'Is_Metabolic': [col in CORE_METABOLIC for col in X_test.columns]
    }).sort_values('Importance', ascending=False)

    top15 = importance.head(15)
    plt.figure(figsize=(10, 6))
    colors = ['#DC2626' if meta else '#64748B' for meta in top15['Is_Metabolic']]
    sns.barplot(x='Importance', y='Feature', data=top15, palette=colors)
    for i, (_, row) in enumerate(top15.iterrows()):
        if row['Is_Metabolic']:
            plt.text(row['Importance']+0.001, i, '✓ 代谢指标', va='center', color='#DC2626')
    plt.xlabel('特征重要性（排列重要性）')
    plt.ylabel('特征名称')
    plt.title('随机森林核心特征重要性（Top15）')
    plt.grid(axis='x', alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rf_feature_importance.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 特征重要性图已保存：{os.path.join(save_dir, 'rf_feature_importance.png')}")

    return importance


def plot_dca(y_true, y_proba, save_dir):
    """决策曲线分析（临床净获益评估）"""
    def net_benefit(threshold):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return (tp - fp * (threshold / (1 - threshold))) / len(y_true) if (1 - threshold) > 0 else 0
    
    thresholds = np.linspace(0, 0.99, 100)
    nb_model = [net_benefit(t) for t in thresholds]
    # 计算"全部筛查"的净获益
    nb_all = [ (sum(y_true) - len(y_true)*t/(1-t))/len(y_true) for t in thresholds ]
    nb_none = [0 for _ in thresholds]  # 不筛查

    plt.figure(figsize=(10, 6))
    # 修复：将颜色和线条样式分开指定，避免格式字符串错误
    plt.plot(thresholds, nb_model, label='随机森林', color='#E63946', linewidth=3)
    plt.plot(thresholds, nb_all, label='全部筛查', color='gray', linestyle='--')  # 重点修复此行
    plt.plot(thresholds, nb_none, label='不筛查', color='black', linestyle=':')
    plt.xlabel('风险阈值（预测患病概率）')
    plt.ylabel('临床净获益')
    plt.title('随机森林决策曲线分析')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rf_dca.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 决策曲线已保存：{os.path.join(save_dir, 'rf_dca.png')}")


# ========================
# 5. 分层验证
# ========================
def stratified_validation(model, X_test, y_test, original_test, save_dir):
    y_proba = model.predict_proba(X_test)[:, 1]
    original_test = original_test.reset_index(drop=True)
    
    # 年龄分层
    age_groups = pd.cut(original_test['age'], bins=[0, 60, 120], labels=['<60岁', '≥60岁'])
    age_auc = {g: roc_auc_score(y_test[age_groups==g], y_proba[age_groups==g]) 
               for g in age_groups.unique()}
    
    # BMI分层
    bmi_groups = pd.cut(original_test['bmi'], bins=[0, 18.5, 24, 28, 100], labels=['偏瘦', '正常', '超重', '肥胖'])
    bmi_auc = {g: roc_auc_score(y_test[bmi_groups==g], y_proba[bmi_groups==g]) 
               for g in bmi_groups.unique() if g is not np.nan}
    
    # 输出结果
    print("\n======= 分层验证结果 =======")
    print("【年龄分层AUC】")
    for g, auc in age_auc.items():
        print(f"   {g}：{auc:.4f}")
    print("\n【BMI分层AUC】")
    for g, auc in bmi_auc.items():
        print(f"   {g}：{auc:.4f}")
    
    # 保存结果
    pd.DataFrame({'年龄分组': age_auc.keys(), 'AUC': age_auc.values()}).to_csv(
        os.path.join(save_dir, 'stratified_age.csv'), index=False, encoding='utf-8-sig'
    )
    pd.DataFrame({'BMI分组': bmi_auc.keys(), 'AUC': bmi_auc.values()}).to_csv(
        os.path.join(save_dir, 'stratified_bmi.csv'), index=False, encoding='utf-8-sig'
    )


# ========================
# 6. 主函数
# ========================
def main():
    base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, 'rf_optimized_results')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据预处理
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_preprocess_data(
        os.path.join(base_dir, 'train_dataset_optimized.csv'),
        os.path.join(base_dir, 'test_dataset_optimized.csv')
    )

    # 2. 模型训练与优化
    best_rf, best_params = train_optimized_rf(X_train, y_train)

    # 3. 模型评估
    metrics, y_pred_proba, y_pred, fpr, tpr = evaluate_model(best_rf, X_test, y_test, save_dir)

    # 4. 交叉验证稳定性分析
    cv_results = cross_validate(
        best_rf, X_train, y_train,
        cv=5, scoring=scoring,
        n_jobs=1
    )
    print("\n======= 交叉验证稳定性 =======")
    print(f"AUC均值±标准差：{cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"灵敏度均值±标准差：{cv_results['test_sensitivity'].mean():.4f} ± {cv_results['test_sensitivity'].std():.4f}")
    print(f"特异度均值±标准差：{cv_results['test_specificity'].mean():.4f} ± {cv_results['test_specificity'].std():.4f}")

    # 5. 可视化
    global thresholds
    thresholds = np.linspace(0, 1, len(fpr))
    plot_roc_curve(fpr, tpr, metrics['auc'], metrics['best_threshold'], save_dir)
    feature_importance = plot_feature_importance(best_rf, X_test, y_test, save_dir)
    plot_dca(y_test, y_pred_proba, save_dir)  # 修复后可正常运行

    # 6. 分层验证
    stratified_validation(best_rf, X_test, y_test, test_df, save_dir)

    # 7. 保存结果
    joblib.dump(best_rf, os.path.join(save_dir, 'rf_best_model.pkl'))
    pd.DataFrame({
        'rf_proba': y_pred_proba, 'true_label': y_test
    }).to_csv(os.path.join(save_dir, 'rf_pred_proba.csv'), index=False, encoding='utf-8-sig')
    feature_importance.to_csv(os.path.join(save_dir, 'feature_importance.csv'), index=False, encoding='utf-8-sig')

    print("\n✅ 所有结果已保存至：", save_dir)


if __name__ == "__main__":
    main()
