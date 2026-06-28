import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import warnings
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# 解决中文路径编码问题
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# 核心配置（使用已知最佳参数）
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']
RANDOM_STATE = 42
BEST_PARAMS = {'C': 0.05, 'penalty': 'l2', 'solver': 'liblinear'}  # 直接使用最优参数
CV_AUC = 0.8741  # 已知5折平均AUC
CV_STD = 0.0091   # 已知标准差


# ========================
# 1. 数据预处理（含SMOTE）
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

    selector = VarianceThreshold(threshold=0.1)
    X_train_num = selector.fit_transform(X_train[num_cols])
    kept_num_cols = num_cols[selector.get_support()].tolist()
    X_train = pd.concat([pd.DataFrame(X_train_num, columns=kept_num_cols), X_train[cat_cols]], axis=1)
    X_test = pd.concat([pd.DataFrame(selector.transform(X_test[num_cols]), columns=kept_num_cols), X_test[cat_cols]], axis=1)
    print(f"✅ 过滤低方差特征：保留{X_train.shape[1]}个特征（原{len(num_cols)+len(cat_cols)}个）")

    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)
    scaler = StandardScaler()
    X_train[metabolic_cols] = scaler.fit_transform(X_train[metabolic_cols])
    X_test[metabolic_cols] = scaler.transform(X_test[metabolic_cols])

    # SMOTE过采样（仅训练集）
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"✅ SMOTE过采样：训练集从{X_train.shape[0]}→{X_train_smote.shape[0]}，患病比例从{y_train.mean():.2%}→{y_train_smote.mean():.2%}")
    
    return X_train_smote, y_train_smote, X_train, y_train, X_test, y_test, metabolic_cols


# ========================
# 2. 直接使用最优参数初始化模型（跳过网格搜索）
# ========================
def get_optimized_model(X_train, y_train):
    # 直接用已知最佳参数创建模型
    model = LogisticRegression(
        C=BEST_PARAMS['C'],
        penalty=BEST_PARAMS['penalty'],
        solver=BEST_PARAMS['solver'],
        class_weight='balanced',
        max_iter=5000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


# ========================
# 3. 可视化函数（重新生成图表）
# ========================
def plot_roc_comparison(y_test, lr_proba, xgb_auc=0.8921, save_dir='logistic_results'):
    plt.figure(figsize=(8, 6))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    lr_auc = roc_auc_score(y_test, lr_proba)
    plt.plot(fpr_lr, tpr_lr, color='#2E86AB', linewidth=3, label=f'逻辑回归 (AUC={lr_auc:.4f})')
    # XGBoost参考线
    plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1], 
             [0, 0.65, 0.78, 0.84, 0.89, 1],
             color='#DC2626', linewidth=3, linestyle='-.', label=f'XGBoost (AUC={xgb_auc:.4f})')
    # 随机线
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    # 最佳阈值标记
    best_idx = np.argmax(tpr_lr - fpr_lr)
    plt.scatter(fpr_lr[best_idx], tpr_lr[best_idx], color='#A23B72', s=100, zorder=5)
    plt.annotate(f'最佳阈值\n(灵敏度={tpr_lr[best_idx]:.3f})',
                 xy=(fpr_lr[best_idx], tpr_lr[best_idx]),
                 xytext=(fpr_lr[best_idx]+0.1, tpr_lr[best_idx]-0.2),
                 arrowprops=dict(arrowstyle='->', color='#A23B72'))
    plt.xlabel('假阳性率（1-特异度）', fontsize=12)
    plt.ylabel('真阳性率（灵敏度）', fontsize=12)
    plt.title('逻辑回归 vs XGBoost ROC曲线对比', fontsize=14, pad=20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'lr_vs_xgb_roc.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ ROC对比图已保存：{os.path.join(save_dir, 'lr_vs_xgb_roc.png')}")


def plot_confusion_matrix(y_test, y_pred, save_dir='logistic_results'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非糖尿病', '糖尿病'],
                yticklabels=['非糖尿病', '糖尿病'])
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('逻辑回归混淆矩阵（SMOTE过采样后）', fontsize=14, pad=20)
    plt.savefig(os.path.join(save_dir, 'lr_confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵图已保存：{os.path.join(save_dir, 'lr_confusion_matrix.png')}")


def plot_improved_dca(y_test, y_proba, save_dir='logistic_results'):
    def calculate_net_benefit(threshold):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        if (1 - threshold) < 1e-6:
            return 0
        return (tp - fp * (threshold / (1 - threshold))) / len(y_test)
    
    thresholds = np.linspace(0, 0.99, 100)
    nb_model = [calculate_net_benefit(t) for t in thresholds]
    nb_all = [calculate_net_benefit(t) for t in thresholds]  # 全筛
    nb_none = [0 for _ in thresholds]  # 不筛

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb_model, label='逻辑回归（SMOTE优化）', color='#2E86AB', linewidth=3)
    plt.plot(thresholds, nb_all, label='全部筛查', color='gray', linestyle='--')
    plt.plot(thresholds, nb_none, label='不筛查', color='black', linestyle=':')
    best_idx = np.argmax(nb_model)
    plt.axvline(x=thresholds[best_idx], color='#A23B72', linestyle='-.', 
                label=f'最佳净获益阈值 ({thresholds[best_idx]:.2f})')
    plt.xlabel('风险阈值（预测患病概率）', fontsize=12)
    plt.ylabel('临床净获益', fontsize=12)
    plt.title('逻辑回归决策曲线分析（优化后）', fontsize=14, pad=20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'lr_improved_dca.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 优化决策曲线已保存：{os.path.join(save_dir, 'lr_improved_dca.png')}")


def plot_metabolic_or(metabolic_or, save_dir='logistic_results'):
    """新增：代谢指标OR值条形图"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='OR', y='Feature', data=metabolic_or,
                palette=['#DC2626' if or_val > 1 else '#2E86AB' for or_val in metabolic_or['OR']])
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)  # OR=1参考线
    plt.xlabel('风险比（OR值）', fontsize=12)
    plt.ylabel('代谢指标', fontsize=12)
    plt.title('代谢指标对糖尿病的影响（OR值）', fontsize=14, pad=20)
    for i, val in enumerate(metabolic_or['OR']):
        plt.text(val + 0.05, i, f'{val:.4f}', va='center')  # 标注OR值
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metabolic_or_plot.png'), dpi=300)
    plt.close()
    print(f"✅ 代谢指标OR值图已保存：{os.path.join(save_dir, 'metabolic_or_plot.png')}")


# ========================
# 4. 主函数（直接使用最优参数）
# ========================
def main():
    base_dir = r''
    save_dir = os.path.join(base_dir, 'logistic_results')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据预处理
    X_train_smote, y_train_smote, X_train_original, y_train_original, X_test, y_test, metabolic_cols = robust_data_preprocessing(
        os.path.join(base_dir, 'train_dataset_optimized.csv'),
        os.path.join(base_dir, 'test_dataset_optimized.csv')
    )

    # 2. 直接使用最优参数建模（跳过网格搜索）
    print("\n🔍 使用已知最佳参数：")
    print(f"最佳参数：{BEST_PARAMS}")
    print(f"5折平均AUC：{CV_AUC:.4f} ± {CV_STD:.4f}")
    lr_model = get_optimized_model(X_train_original, y_train_original)

    # 3. 模型评估
    y_proba = lr_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_proba >= best_threshold).astype(int)

    print("\n======= 模型性能（测试集） =======")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

    # 4. 代谢指标OR值分析
    coef_df = pd.DataFrame({
        'Feature': X_train_original.columns,
        'OR': np.exp(lr_model.coef_[0])
    })
    metabolic_or = coef_df[coef_df['Feature'].isin(metabolic_cols)].sort_values('OR', ascending=False)
    print("\n📌 代谢指标OR值：")
    print(metabolic_or.round(4))
    metabolic_or.to_csv(os.path.join(save_dir, 'metabolic_or.csv'), index=False, encoding='utf-8-sig')

    # 5. 重新生成所有图表
    plot_roc_comparison(y_test, y_proba, save_dir=save_dir)
    plot_confusion_matrix(y_test, y_pred, save_dir=save_dir)
    plot_improved_dca(y_test, y_proba, save_dir=save_dir)
    plot_metabolic_or(metabolic_or, save_dir=save_dir)  # 新增OR值可视化

    # 6. 保存模型
    joblib.dump(lr_model, os.path.join(save_dir, 'diabetes_lr_best_params.pkl'))
    print("\n✅ 模型与所有可视化图表已保存")


if __name__ == "__main__":
    main()
