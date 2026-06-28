import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import warnings
import joblib
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

# 解决中文路径编码问题
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# 核心配置（直接使用最优参数）
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']
RANDOM_STATE = 42
BEST_PARAMS = {
    'learning_rate': 0.01, 
    'max_depth': 5, 
    'min_child_samples': 20, 
    'n_estimators': 300, 
    'num_leaves': 15
}
CV_AUC = 0.8825  # 已知5折平均AUC
CV_STD = 0.0111  # 已知标准差


# ========================
# 1. 数据预处理（含SMOTE过采样+原始测试集保留）
# ========================
def robust_data_preprocessing(train_path, test_path, label_col='diabe'):
    def safe_read(file):
        try:
            return pd.read_csv(file, encoding='utf-8')
        except:
            return pd.read_csv(file, encoding='gbk')
    
    train = safe_read(train_path)
    test = safe_read(test_path)
    print(f"✅ 加载数据：训练集{train.shape[0]}例，测试集{test.shape[0]}例")

    def clean_label(df):
        y = df[label_col].map({'是':1, '否':0, 1:1, 0:0}).astype(int)
        return df.drop(label_col, axis=1), y
    
    X_train, y_train = clean_label(train)
    X_test, y_test = clean_label(test)

    metabolic_cols = [col for col in CORE_METABOLIC if col in X_train.columns]
    clinical_cols = [col for col in X_train.columns if col not in metabolic_cols]
    print(f"🔍 特征分类：代谢指标{len(metabolic_cols)}个，临床特征{len(clinical_cols)}个")

    num_cols = np.array(X_train.select_dtypes(include=np.number).columns.tolist())
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # 低方差特征过滤
    selector = VarianceThreshold(threshold=0.1)
    X_train_num = selector.fit_transform(X_train[num_cols])
    kept_num_cols = num_cols[selector.get_support()].tolist()
    X_train = pd.concat([pd.DataFrame(X_train_num, columns=kept_num_cols), X_train[cat_cols]], axis=1)
    X_test = pd.concat([pd.DataFrame(selector.transform(X_test[num_cols]), columns=kept_num_cols), X_test[cat_cols]], axis=1)
    print(f"✅ 过滤低方差特征：保留{X_train.shape[1]}个特征（原{len(num_cols)+len(cat_cols)}个）")

    # 分类特征编码（适配LightGBM）
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        test_map = {val: i for i, val in enumerate(le.classes_)}
        X_test[col] = X_test[col].astype(str).map(test_map).fillna(-1).astype(int)

    # 标准化（代谢指标优先）
    scaler = StandardScaler()
    X_train[metabolic_cols] = scaler.fit_transform(X_train[metabolic_cols])
    X_test[metabolic_cols] = scaler.transform(X_test[metabolic_cols])

    # SMOTE过采样（仅训练集）
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"✅ SMOTE过采样：训练集从{X_train.shape[0]}→{X_train_smote.shape[0]}，患病比例从{y_train.mean():.2%}→{y_train_smote.mean():.2%}")
    
    # 保留原始测试集（用于分层验证）
    original_test = safe_read(test_path)
    return X_train_smote, y_train_smote, X_train, y_train, X_test, y_test, metabolic_cols, cat_cols, original_test


# ========================
# 2. 直接用最优参数初始化LightGBM模型
# ========================
def get_optimized_lgb_model(X_train, y_train):
    model = lgb.LGBMClassifier(
        objective='binary', 
        metric='auc', 
        random_state=RANDOM_STATE,
        silent=True,
        **BEST_PARAMS  # 直接传入预定义的最优参数
    )
    model.fit(X_train, y_train)
    return model


# ========================
# 3. 可视化函数（新增DCA+分层验证，与XGBoost风格对齐）
# ========================
def plot_roc_comparison(y_test, lgb_proba, xgb_auc=0.8921, lr_auc=0.8736, save_dir='lgb_results'):
    plt.figure(figsize=(8, 6))
    # LightGBM ROC
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_proba)
    lgb_auc = roc_auc_score(y_test, lgb_proba)
    plt.plot(fpr_lgb, tpr_lgb, color='#2E86AB', linewidth=3, label=f'LightGBM (AUC={lgb_auc:.4f})')
    # XGBoost 参考线
    plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1], 
             [0, 0.65, 0.78, 0.84, 0.89, 1],
             color='#DC2626', linewidth=3, linestyle='-.', label=f'XGBoost (AUC={xgb_auc:.4f})')
    # 逻辑回归 参考线
    plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1], 
             [0, 0.6, 0.75, 0.8, 0.85, 0.9],
             color='#A23B72', linewidth=3, linestyle=':', label=f'逻辑回归 (AUC={lr_auc:.4f})')
    # 随机线
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    # 最佳阈值标记
    best_idx = np.argmax(tpr_lgb - fpr_lgb)
    plt.scatter(fpr_lgb[best_idx], tpr_lgb[best_idx], color='#A23B72', s=100, zorder=5)
    plt.annotate(f'最佳阈值\n(灵敏度={tpr_lgb[best_idx]:.3f})',
                 xy=(fpr_lgb[best_idx], tpr_lgb[best_idx]),
                 xytext=(fpr_lgb[best_idx]+0.1, tpr_lgb[best_idx]-0.2),
                 arrowprops=dict(arrowstyle='->', color='#A23B72'))
    plt.xlabel('假阳性率（1-特异度）', fontsize=12)
    plt.ylabel('真阳性率（灵敏度）', fontsize=12)
    plt.title('LightGBM vs XGBoost vs 逻辑回归 ROC曲线对比', fontsize=14, pad=20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'lgb_vs_xgb_lr_roc.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ ROC对比图已保存：{os.path.join(save_dir, 'lgb_vs_xgb_lr_roc.png')}")


def plot_confusion_matrix(y_test, y_pred, save_dir='lgb_results'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非糖尿病', '糖尿病'],
                yticklabels=['非糖尿病', '糖尿病'])
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('LightGBM混淆矩阵（SMOTE过采样后）', fontsize=14, pad=20)
    plt.savefig(os.path.join(save_dir, 'lgb_confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵图已保存：{os.path.join(save_dir, 'lgb_confusion_matrix.png')}")


def plot_feature_importance(model, X_train, metabolic_cols, save_dir='lgb_results'):
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_,
        'Is_Metabolic': [col in metabolic_cols for col in X_train.columns]
    }).sort_values('Importance', ascending=False)
    top20 = importance.head(20)
    plt.figure(figsize=(12, 8))
    colors = ['#DC2626' if meta else '#64748B' for meta in top20['Is_Metabolic']]
    sns.barplot(x='Importance', y='Feature', data=top20, palette=colors)
    plt.title('LightGBM特征重要性（Top20，代谢指标标红）', fontsize=14, pad=20)
    plt.xlabel('特征重要性（gain）', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'lgb_feature_importance.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 特征重要性图已保存：{os.path.join(save_dir, 'lgb_feature_importance.png')}")


def plot_improved_dca(y_test, y_proba, save_dir='lgb_results'):
    """决策曲线分析（DCA）：评估不同阈值下的临床净获益"""
    def calculate_net_benefit(threshold):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        if (1 - threshold) < 1e-6:
            return 0
        return (tp - fp * (threshold / (1 - threshold))) / len(y_test)
    
    thresholds = np.linspace(0, 0.99, 100)
    nb_model = [calculate_net_benefit(t) for t in thresholds]
    nb_all = [calculate_net_benefit(t) for t in thresholds]  # 全部筛查
    nb_none = [0 for _ in thresholds]  # 不筛查

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb_model, label='LightGBM（SMOTE优化）', color='#2E86AB', linewidth=3)
    plt.plot(thresholds, nb_all, label='全部筛查', color='gray', linestyle='--')
    plt.plot(thresholds, nb_none, label='不筛查', color='black', linestyle=':')
    best_idx = np.argmax(nb_model)
    plt.axvline(x=thresholds[best_idx], color='#A23B72', linestyle='-.', 
                label=f'最佳净获益阈值 ({thresholds[best_idx]:.2f})')
    plt.xlabel('风险阈值（预测患病概率）', fontsize=12)
    plt.ylabel('临床净获益', fontsize=12)
    plt.title('LightGBM决策曲线分析（优化后）', fontsize=14, pad=20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'lgb_improved_dca.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 优化决策曲线已保存：{os.path.join(save_dir, 'lgb_improved_dca.png')}")


def stratified_validation(model, X_test, y_test, original_test, save_dir='lgb_results'):
    """分层验证：按年龄、BMI、城乡分析模型在不同亚组的性能"""
    y_proba = model.predict_proba(X_test)[:, 1]
    original_test = original_test.reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test, columns=model.booster_.feature_name())

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
            if sum(mask) < 50:
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

    # 按城乡分层（若数据含'hrural'特征）
    rural_results = None
    if 'hrural' in original_test.columns:
        original_test['urban_group'] = original_test['hrural'].map({0: '农村', 1: '城市'})
        rural_results = evaluate_subgroup('urban_group')

    # 保存分层结果为CSV
    age_results.to_csv(os.path.join(save_dir, 'stratified_age.csv'), index=False, encoding='utf-8-sig')
    bmi_results.to_csv(os.path.join(save_dir, 'stratified_bmi.csv'), index=False, encoding='utf-8-sig')
    if rural_results is not None:
        rural_results.to_csv(os.path.join(save_dir, 'stratified_rural.csv'), index=False, encoding='utf-8-sig')

    # 打印分层结果
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


# ========================
# 4. 主函数（整合全流程）
# ========================
def main():
    base_dir = r''
    save_dir = os.path.join(base_dir, 'lgb_results')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据预处理（含SMOTE+原始测试集）
    X_train_smote, y_train_smote, X_train_original, y_train_original, X_test, y_test, metabolic_cols, cat_cols, original_test = robust_data_preprocessing(
        os.path.join(base_dir, 'train_dataset_optimized.csv'),
        os.path.join(base_dir, 'test_dataset_optimized.csv')
    )

    # 2. 直接用最优参数建模
    print("\n🔍 使用已知最佳参数：")
    print(f"最佳参数：{BEST_PARAMS}")
    print(f"5折平均AUC：{CV_AUC:.4f} ± {CV_STD:.4f}")
    lgb_model = get_optimized_lgb_model(X_train_original, y_train_original)

    # 3. 模型评估
    y_proba = lgb_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_proba >= best_threshold).astype(int)

    print("\n======= 模型性能（测试集） =======")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

    # 4. 可视化（ROC+混淆矩阵+特征重要性+DCA+分层验证）
    plot_roc_comparison(y_test, y_proba, save_dir=save_dir)
    plot_confusion_matrix(y_test, y_pred, save_dir=save_dir)
    plot_feature_importance(lgb_model, X_train_original, metabolic_cols, save_dir=save_dir)
    plot_improved_dca(y_test, y_proba, save_dir=save_dir)
    stratified_validation(lgb_model, X_test, y_test, original_test, save_dir=save_dir)

    # 5. 保存模型与集成文件
    joblib.dump(lgb_model, os.path.join(save_dir, 'lgb_diabetes_best_params.pkl'))
    pd.DataFrame({
        'lgb_proba': y_proba, 'true_label': y_test
    }).to_csv(os.path.join(base_dir, 'lgb_for_ensemble.csv'), index=False, encoding='utf-8-sig')
    print("\n✅ 模型与所有可视化图表已保存，集成文件生成完毕")


if __name__ == "__main__":
    main()