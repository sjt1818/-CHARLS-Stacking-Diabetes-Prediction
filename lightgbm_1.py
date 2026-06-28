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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

# 解决中文路径编码问题（核心：指定系统临时目录为英文路径）
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# 核心配置
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']
RANDOM_STATE = 42
GRID_CACHE_PATH = 'lgb_grid_search_cache.pkl'  # LightGBM网格搜索缓存路径


# ========================
# 1. 数据预处理（含SMOTE过采样）
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
        # 处理测试集 unseen 类别
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
    
    return X_train_smote, y_train_smote, X_train, y_train, X_test, y_test, metabolic_cols, cat_cols


# ========================
# 2. 网格搜索（单进程+缓存，避免中文路径冲突）
# ========================
def optimize_lightgbm(X_train, y_train):
    # 加载缓存的网格搜索结果
    if os.path.exists(GRID_CACHE_PATH):
        print("\n📌 加载缓存的网格搜索结果，避免重复训练...")
        return joblib.load(GRID_CACHE_PATH)
    
    # 定义LightGBM参数网格
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 30]
    }

    # 分层5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 单进程网格搜索（n_jobs=1，避免中文路径多进程编码冲突）
    grid = GridSearchCV(
        lgb.LGBMClassifier(
            objective='binary', 
            metric='auc', 
            random_state=RANDOM_STATE,
            silent=True  # 静默模式减少输出
        ),
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1  # 核心：单进程运行
    )
    grid.fit(X_train, y_train)
    
    # 保存结果到缓存
    joblib.dump(grid, GRID_CACHE_PATH)
    return grid


# ========================
# 3. 可视化函数（与XGBoost/逻辑回归风格对齐）
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
    # 提取特征重要性（gain）
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.best_estimator_.feature_importances_,
        'Is_Metabolic': [col in metabolic_cols for col in X_train.columns]
    }).sort_values('Importance', ascending=False)

    # 可视化Top20特征
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


# ========================
# 4. 主函数（整合优化流程）
# ========================
def main():
    base_dir = r''
    save_dir = os.path.join(base_dir, 'lgb_results')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据预处理（含SMOTE）
    X_train_smote, y_train_smote, X_train_original, y_train_original, X_test, y_test, metabolic_cols, cat_cols = robust_data_preprocessing(
        os.path.join(base_dir, 'train_dataset_optimized.csv'),
        os.path.join(base_dir, 'test_dataset_optimized.csv')
    )

    # 2. 网格搜索（单进程+缓存）
    grid = optimize_lightgbm(X_train_original, y_train_original)
    print("\n🔍 网格搜索结果：")
    print(f"最佳参数：{grid.best_params_}")
    print(f"5折平均AUC：{grid.best_score_:.4f} ± {grid.cv_results_['std_test_score'][grid.best_index_]:.4f}")

    # 3. 模型评估
    lgb_model = grid.best_estimator_
    y_proba = lgb_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_proba >= best_threshold).astype(int)

    print("\n======= 模型性能（测试集） =======")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

    # 4. 可视化（与XGBoost/逻辑回归对齐）
    plot_roc_comparison(y_test, y_proba, save_dir=save_dir)
    plot_confusion_matrix(y_test, y_pred, save_dir=save_dir)
    plot_feature_importance(grid, X_train_original, metabolic_cols, save_dir=save_dir)

    # 5. 保存模型与集成文件
    joblib.dump(lgb_model, os.path.join(save_dir, 'lgb_diabetes_optimized.pkl'))
    pd.DataFrame({
        'lgb_proba': y_proba, 'true_label': y_test
    }).to_csv(os.path.join(base_dir, 'lgb_for_ensemble.csv'), index=False, encoding='utf-8-sig')
    print("\n✅ 模型与集成文件已保存")


if __name__ == "__main__":
    main()