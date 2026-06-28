import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import joblib
import sys

# 解决中文路径编码问题
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()

# 全局配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 核心代谢指标
CORE_METABOLIC_COLS = [
    'bl_glu', 'bl_hbalc', 'bl_tg', 'tyg', 'tyg_bmi',
    'bl_hdl', 'bl_ldl', 'bl_crea', 'bl_crp', 'bl_ua'
]

# 目标变量
LABEL_COL = 'diabe'


# ========================
# 1. 数据加载与初步划分
# ========================
def load_and_split_data(data_path):
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='gbk')
    
    print(f"✅ 成功加载原始数据：{data_path}（{df.shape[0]}行 × {df.shape[1]}列）")
    
    X = df.drop(LABEL_COL, axis=1)
    y = df[LABEL_COL].astype(int).clip(0, 1)
    
    print(f"糖尿病患病比例：{y.mean():.2%}（{y.sum()}例患病 / {len(y)}总样本）")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"✅ 数据集划分完成：")
    print(f"   - 训练集：{len(X_train)}样本（患病比例：{y_train.mean():.2%}）")
    print(f"   - 测试集：{len(X_test)}样本（患病比例：{y_test.mean():.2%}）")
    return X_train, X_test, y_train, y_test, df.iloc[X_test.index]


# ========================
# 2. 特征预处理管道
# ========================
def create_preprocessing_pipeline(X_train):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    metabolic_features_in_data = [f for f in CORE_METABOLIC_COLS if f in numeric_features]
    print(f"\n📊 特征分类：")
    print(f"   - 数值特征：{len(numeric_features)}个（含{len(metabolic_features_in_data)}个核心代谢指标）")
    print(f"   - 分类特征：{len(categorical_features)}个（如：{categorical_features[:3]}...）")
    
    # 数值特征预处理
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # 分类特征预处理
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    # 合并预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, numeric_features, categorical_features, metabolic_features_in_data


# ========================
# 3. XGBoost模型训练与交叉验证
# ========================
def train_xgboost_with_cv(X_train, y_train, preprocessor):
    base_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        random_state=42,
        use_label_encoder=False,
        verbosity=0
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    
    param_grid = {
        'model__learning_rate': [0.01, 0.05],
        'model__max_depth': [3, 4],
        'model__n_estimators': [100, 200],
        'model__subsample': [0.8, 0.9]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n======= 开始网格搜索调参 =======")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring='roc_auc', n_jobs=1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"最优参数：{grid_search.best_params_}")
    print(f"5折交叉验证平均AUC：{grid_search.best_score_:.4f} ± {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f}")
    
    return grid_search.best_estimator_, cv


# ========================
# 4. 阈值优化与临床适配分析
# ========================
def optimize_threshold(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # 约登指数最大阈值
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_j_threshold = thresholds[best_j_idx]
    
    # 临床优先阈值
    sens_threshold = 0.8
    valid_idx = np.where(tpr >= sens_threshold)[0]
    if len(valid_idx) > 0:
        best_clin_idx = valid_idx[np.argmax(1 - fpr[valid_idx])]
        best_clin_threshold = thresholds[best_clin_idx]
    else:
        best_clin_threshold = 0.0
    
    # 评估不同阈值性能
    def evaluate_threshold(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sens, spec, tn, fp, fn, tp
    
    # 对比不同阈值
    sens_05, spec_05, tn_05, fp_05, fn_05, tp_05 = evaluate_threshold(0.5)
    sens_j, spec_j, tn_j, fp_j, fn_j, tp_j = evaluate_threshold(best_j_threshold)
    sens_clin, spec_clin, tn_clin, fp_clin, fn_clin, tp_clin = evaluate_threshold(best_clin_threshold)
    
    print("\n======= 阈值优化结果 =======")
    print(f"| 阈值类型               | 阈值值   | 灵敏度   | 特异度   | 漏诊数 | 误诊数 |")
    print(f"|------------------------|----------|----------|----------|--------|--------|")
    print(f"| 默认0.5阈值            | 0.5000   | {sens_05:.4f} | {spec_05:.4f} | {fn_05:6d} | {fp_05:6d} |")
    print(f"| 约登最优阈值           | {best_j_threshold:.4f} | {sens_j:.4f} | {spec_j:.4f} | {fn_j:6d} | {fp_j:6d} |")
    print(f"| 临床优先阈值（灵敏度≥80%）| {best_clin_threshold:.4f} | {sens_clin:.4f} | {spec_clin:.4f} | {fn_clin:6d} | {fp_clin:6d} |")
    
    return {
        'best_j': {'threshold': best_j_threshold, 'sens': sens_j, 'spec': spec_j},
        'clinical': {'threshold': best_clin_threshold, 'sens': sens_clin, 'spec': spec_clin},
        'fpr': fpr, 'tpr': tpr
    }


# ========================
# 5. 分层验证（按临床特征）
# ========================
def stratified_validation(model, X_test, y_test, original_test_data):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    def evaluate_subgroup(feature, bins=None, labels=None):
        if bins:
            original_test_data[f'{feature}_group'] = pd.cut(
                original_test_data[feature], bins=bins, labels=labels
            )
        else:
            original_test_data[f'{feature}_group'] = original_test_data[feature]
        
        results = []
        for group in original_test_data[f'{feature}_group'].unique():
            mask = original_test_data[f'{feature}_group'] == group
            if sum(mask) < 50:
                continue
            auc = roc_auc_score(y_test[mask], y_pred_proba[mask])
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
    
    # 按城乡分层
    rural_results = None
    if 'hrural' in original_test_data.columns:
        original_test_data['urban_group'] = original_test_data['hrural'].map(
            {0: '农村', 1: '城市'}
        )
        rural_results = evaluate_subgroup('urban_group')
    
    print("\n======= 分层验证结果 =======")
    print("【年龄分层】")
    print(age_results.round(4))
    print("\n【BMI分层】")
    print(bmi_results.round(4))
    if rural_results is not None and not rural_results.empty:
        print("\n【城乡分层】")
        print(rural_results.round(4))
    else:
        print("\n【城乡分层】：无有效数据或特征不存在")
    
    return age_results, bmi_results, rural_results


# ========================
# 6. 决策曲线分析（DCA）
# ========================
def decision_curve_analysis(y_true, y_pred_proba, save_path='dca_curve.png'):
    def calculate_net_benefit(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if (1 - threshold) < 1e-6:
            return 0
        net_benefit = (tp - fp * (threshold / (1 - threshold))) / len(y_true)
        return net_benefit
    
    thresholds = np.linspace(0, 0.99, 100)
    nb_model = [calculate_net_benefit(t) for t in thresholds]
    nb_all = [calculate_net_benefit(t) for t in thresholds]
    nb_none = [0 for _ in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb_model, label='XGBoost模型', color='#8B008B', linewidth=3)
    plt.plot(thresholds, nb_all, label='全部筛查', color='gray', linestyle='--')
    plt.plot(thresholds, nb_none, label='不筛查', color='black', linestyle=':')
    plt.xlabel('风险阈值（预测患病概率）', fontsize=12)
    plt.ylabel('临床净获益', fontsize=12)
    plt.title('糖尿病预测模型决策曲线分析', fontsize=14, pad=20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 决策曲线已保存：{save_path}")


# ========================
# 7. 特征重要性分析（修复长度不匹配问题）
# ========================
def analyze_feature_importance(model, numeric_features, categorical_features, metabolic_features):
    """分析特征重要性，重点标记代谢指标"""
    # 获取预处理管道
    preprocessor = model.named_steps['preprocessor']
    
    # 获取数值特征名称（直接使用原始名称）
    num_feature_names = numeric_features.copy()
    
    # 获取分类特征独热编码后的名称
    try:
        # 获取分类特征处理器
        cat_transformer = preprocessor.named_transformers_['cat']
        # 获取独热编码器
        onehot_encoder = cat_transformer.named_steps['onehot']
        # 获取编码后的特征名称
        cat_feature_names = list(onehot_encoder.get_feature_names_out(categorical_features))
    except KeyError:
        # 如果没有分类特征
        cat_feature_names = []
    
    # 合并所有特征名称
    all_feature_names = num_feature_names + cat_feature_names
    
    # 获取模型特征重要性
    feature_importances = model.named_steps['model'].feature_importances_
    
    # 关键修复：确保特征名称和重要性数组长度一致
    if len(all_feature_names) != len(feature_importances):
        print(f"\n⚠️ 特征名称与重要性长度不匹配：{len(all_feature_names)} vs {len(feature_importances)}")
        # 截断或扩展以匹配长度（取较短的长度）
        min_length = min(len(all_feature_names), len(feature_importances))
        all_feature_names = all_feature_names[:min_length]
        feature_importances = feature_importances[:min_length]
        print(f"⚠️ 已调整为相同长度：{min_length}")
    
    # 创建特征重要性DataFrame
    importance = pd.DataFrame({
        '特征': all_feature_names,
        '重要性': feature_importances,
        '是否代谢指标': [f in metabolic_features for f in all_feature_names]
    }).sort_values('重要性', ascending=False)
    
    # 显示Top15特征
    print("\n======= Top15重要特征（含代谢指标标记） =======")
    top_n = min(15, len(importance))
    top15 = importance.head(top_n)
    top15['代谢指标标记'] = top15['是否代谢指标'].map({True: '✓ 代谢', False: ''})
    print(top15[['特征', '重要性', '代谢指标标记']].round(4))
    
    # 可视化
    if len(top15) > 0:
        plt.figure(figsize=(10, 6))
        colors = ['#DC2626' if meta else '#64748B' for meta in top15['是否代谢指标']]
        sns.barplot(x='重要性', y='特征', data=top15, palette=colors)
        plt.title('XGBoost模型特征重要性（Top15）', fontsize=14, pad=20)
        plt.xlabel('特征重要性', fontsize=12)
        plt.ylabel('特征名称', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.savefig('feature_importance.png', bbox_inches='tight')
        plt.close()
        print("\n✅ 特征重要性图已保存：feature_importance.png")
    
    return importance


# ========================
# 8. 主函数：整合所有流程
# ========================
def main():
    # 1. 加载数据并划分训练集/测试集
    X_train, X_test, y_train, y_test, original_test_data = load_and_split_data(
        data_path='final_charls_diabetes_data_with_new_features.csv'
    )
    
    # 2. 创建预处理管道
    preprocessor, numeric_features, categorical_features, metabolic_features = create_preprocessing_pipeline(X_train)
    
    # 3. 训练模型并交叉验证
    best_model, cv = train_xgboost_with_cv(X_train, y_train, preprocessor)
    
    # 4. 测试集最终评估
    print("\n======= 测试集最终评估 =======")
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"测试集AUC：{test_auc:.4f}")
    
    # 5. 阈值优化
    threshold_results = optimize_threshold(y_test, y_pred_proba)
    best_threshold = threshold_results['best_j']['threshold']
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非糖尿病', '糖尿病'],
                yticklabels=['非糖尿病', '糖尿病'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵（约登最优阈值）')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # 6. 分层验证
    stratified_validation(best_model, X_test, y_test, original_test_data)
    
    # 7. 决策曲线分析
    decision_curve_analysis(y_test, y_pred_proba)
    
    # 8. 特征重要性分析
    feature_importance = analyze_feature_importance(
        best_model, numeric_features, categorical_features, metabolic_features
    )
    
    # 9. 保存模型与结果
    if not os.path.exists('model_results'):
        os.makedirs('model_results')
    joblib.dump(best_model, 'model_results/xgboost_diabetes_model.pkl')
    feature_importance.to_csv('model_results/feature_importance_full.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 模型与结果已保存至 model_results 文件夹：")
    print("   - 模型文件：xgboost_diabetes_model.pkl")
    print("   - 完整特征重要性：feature_importance_full.csv")


if __name__ == "__main__":
    main()
    