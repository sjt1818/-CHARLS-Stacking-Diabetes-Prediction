import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import warnings
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from scipy.stats import bootstrap

# 配置环境
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# 核心配置
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']
RANDOM_STATE = 42
BEST_PARAMS = {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear', 'class_weight': 'balanced'}


# 数据预处理
def preprocess_data(train_path, test_path, label_col='diabe'):
    """数据预处理，与其他模型保持一致的处理流程"""
    def safe_read(file):
        try:
            return pd.read_csv(file, encoding='utf-8')
        except:
            return pd.read_csv(file, encoding='gbk')
    
    train = safe_read(train_path)
    test = safe_read(test_path)
    print(f"✅ 加载数据：训练集{train.shape[0]}例，测试集{test.shape[0]}例")

    # 分离特征与标签
    def extract_label(df):
        y = df[label_col].map({'是':1, '否':0, 1:1, 0:0}).astype(int)
        return df.drop(label_col, axis=1), y
    
    X_train, y_train = extract_label(train)
    X_test, y_test = extract_label(test)

    # 特征分类
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    print(f"🔍 特征分类：数值特征{len(num_cols)}个，分类特征{len(cat_cols)}个")

    # 缺失值处理
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # 分类特征编码
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        test_map = {val: i for i, val in enumerate(le.classes_)}
        X_test[col] = X_test[col].astype(str).map(test_map).fillna(-1).astype(int)

    # 标准化（Lasso必需）
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, y_train, X_test, y_test, scaler, train, test


# 模型训练与交叉验证
def train_lasso_model(X_train, y_train):
    """训练Lasso模型并进行交叉验证"""
    # 初始化模型
    lasso = LogisticRegression(
        **BEST_PARAMS,
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        lasso, X_train, y_train, 
        cv=cv, scoring='roc_auc'
    )
    
    # 训练最终模型
    lasso.fit(X_train, y_train)
    
    print(f"📊 5折交叉验证结果：")
    print(f"   平均AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return lasso, cv_scores


# 决策曲线分析 (DCA)
def plot_dca(y_true, y_proba, model_name="Lasso", save_dir='results'):
    """绘制决策曲线分析图"""
    os.makedirs(save_dir, exist_ok=True)
    
    def calculate_net_benefit(threshold):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if (1 - threshold) < 1e-6:
            return 0
        return (tp - fp * (threshold / (1 - threshold))) / len(y_true)
    
    thresholds = np.linspace(0, 0.99, 100)
    nb_model = [calculate_net_benefit(t) for t in thresholds]
    nb_all = [calculate_net_benefit(t) for t in thresholds]  # 全部预测为阳性
    nb_none = [0 for _ in thresholds]  # 全部预测为阴性

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb_model, label=model_name, color='#3B82F6', linewidth=3)
    plt.plot(thresholds, nb_all, label='全部筛查', color='gray', linestyle='--')
    plt.plot(thresholds, nb_none, label='不筛查', color='black', linestyle=':')
    
    # 标记最佳净获益点
    best_idx = np.argmax(nb_model)
    plt.axvline(x=thresholds[best_idx], color='#EF4444', linestyle='-.', 
                label=f'最佳阈值 ({thresholds[best_idx]:.2f})')
    
    plt.xlabel('风险阈值（预测患病概率）', fontsize=12)
    plt.ylabel('临床净获益', fontsize=12)
    plt.title(f'{model_name}决策曲线分析', fontsize=14, pad=20)
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(save_dir, f'{model_name.lower()}_dca.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ DCA图已保存：{save_path}")


# 分层验证
def stratified_validation(model, X_test, y_test, original_test, save_dir='results'):
    """按年龄、BMI等分层验证模型性能"""
    os.makedirs(save_dir, exist_ok=True)
    y_proba = model.predict_proba(X_test)[:, 1]
    original_test = original_test.reset_index(drop=True)
    
    def evaluate_subgroup(feature, bins=None, labels=None):
        if bins:
            original_test[f'{feature}_group'] = pd.cut(
                original_test[feature], bins=bins, labels=labels
            )
        else:
            original_test[f'{feature}_group'] = original_test[feature]
        
        results = []
        for group in original_test[f'{feature}_group'].unique():
            mask = original_test[f'{feature}_group'] == group
            if sum(mask) < 50:  # 过滤样本量过小的亚组
                continue
            auc = roc_auc_score(y_test[mask], y_proba[mask])
            results.append({
                '特征': feature,
                '亚组': group,
                '样本数': sum(mask),
                'AUC': auc
            })
        return pd.DataFrame(results)
    
    # 按年龄分层
    age_results = evaluate_subgroup(
        'age', bins=[0, 60, 120], labels=['<60岁', '≥60岁']
    )
    
    # 按BMI分层
    bmi_results = evaluate_subgroup(
        'bmi', bins=[0, 18.5, 24, 28, 100], labels=['偏瘦', '正常', '超重', '肥胖']
    )
    
    # 按性别分层（如果有性别特征）
    gender_results = None
    if 'gender' in original_test.columns:
        gender_results = evaluate_subgroup('gender')
    
    # 保存结果
    age_results.to_csv(os.path.join(save_dir, 'stratified_age.csv'), index=False, encoding='utf-8-sig')
    bmi_results.to_csv(os.path.join(save_dir, 'stratified_bmi.csv'), index=False, encoding='utf-8-sig')
    if gender_results is not None:
        gender_results.to_csv(os.path.join(save_dir, 'stratified_gender.csv'), index=False, encoding='utf-8-sig')
    
    # 打印结果
    print("\n======= 分层验证结果 =======")
    print("【年龄分层】")
    print(age_results.round(4))
    print("\n【BMI分层】")
    print(bmi_results.round(4))
    if gender_results is not None and not gender_results.empty:
        print("\n【性别分层】")
        print(gender_results.round(4))
    
    return age_results, bmi_results, gender_results


# 可视化功能
def plot_roc_curve(y_test, y_proba, model_name="Lasso", save_dir='results'):
    """绘制ROC曲线"""
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3B82F6', linewidth=3, label=f'{model_name} (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    
    # 标记最佳阈值
    best_idx = np.argmax(tpr - fpr)
    plt.scatter(fpr[best_idx], tpr[best_idx], color='#EF4444', s=100, zorder=5)
    plt.annotate(f'最佳阈值\n({tpr[best_idx]:.3f}灵敏度)',
                 xy=(fpr[best_idx], tpr[best_idx]),
                 xytext=(fpr[best_idx]+0.1, tpr[best_idx]-0.2),
                 arrowprops=dict(arrowstyle='->', color='#EF4444'))
    
    plt.xlabel('假阳性率（1-特异度）', fontsize=12)
    plt.ylabel('真阳性率（灵敏度）', fontsize=12)
    plt.title(f'{model_name} ROC曲线', fontsize=14, pad=20)
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(save_dir, f'{model_name.lower()}_roc.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存：{save_path}")


def plot_feature_importance(model, feature_names, core_metabolic, save_dir='results'):
    """绘制特征重要性图"""
    os.makedirs(save_dir, exist_ok=True)
    
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': np.abs(model.coef_[0]),
        'Is_Metabolic': [f in core_metabolic for f in feature_names]
    }).sort_values('Coefficient', ascending=False)
    
    top20 = importance.head(20)
    plt.figure(figsize=(12, 8))
    colors = ['#DC2626' if meta else '#64748B' for meta in top20['Is_Metabolic']]
    
    sns.barplot(x='Coefficient', y='Feature', data=top20, palette=colors)
    plt.title('Lasso特征重要性（Top20，代谢指标标红）', fontsize=14, pad=20)
    plt.xlabel('系数绝对值', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    save_path = os.path.join(save_dir, 'lasso_feature_importance.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征重要性图已保存：{save_path}")
    
    return importance


# 主函数
def main():
    # 路径配置
    base_dir = r''
    save_dir = os.path.join(base_dir, 'lasso_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 数据预处理
    X_train, y_train, X_test, y_test, scaler, train_original, test_original = preprocess_data(
        os.path.join(base_dir, 'train_dataset_optimized.csv'),
        os.path.join(base_dir, 'test_dataset_optimized.csv')
    )
    
    # 2. 模型训练与交叉验证
    print("\n🔍 使用最佳参数训练Lasso模型：")
    print(f"最佳参数: {BEST_PARAMS}")
    lasso_model, cv_scores = train_lasso_model(X_train, y_train)
    
    # 3. 模型评估
    y_proba = lasso_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_proba >= best_threshold).astype(int)
    
    print("\n======= 模型性能（测试集） =======")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))
    
    # 4. 可视化
    plot_roc_curve(y_test, y_proba, save_dir=save_dir)
    plot_feature_importance(lasso_model, X_train.columns, CORE_METABOLIC, save_dir=save_dir)
    plot_dca(y_test, y_proba, save_dir=save_dir)
    
    # 5. 分层验证
    stratified_validation(lasso_model, X_test, y_test, test_original, save_dir=save_dir)
    
    # 6. 保存模型与结果
    joblib.dump({
        'model': lasso_model,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'feature_names': X_train.columns.tolist()
    }, os.path.join(save_dir, 'lasso_model.pkl'))
    
    # 保存预测概率用于集成
    pd.DataFrame({
        'lasso_proba': y_proba,
        'true_label': y_test
    }).to_csv(os.path.join(base_dir, 'lasso_for_ensemble.csv'), index=False, encoding='utf-8-sig')
    
    print("\n✅ 所有结果已保存，Lasso模型训练完成")


if __name__ == "__main__":
    main()
    