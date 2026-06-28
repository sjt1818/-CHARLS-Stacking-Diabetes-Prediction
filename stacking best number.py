import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tempfile
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer  # 新增：缺失值处理工具
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # 新增：构建预处理管道
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings

# 解决中文路径编码问题
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 全局参数
RANDOM_STATE = 42
BASE_DIR = os.getcwd()
SAVE_DIR = os.path.join(BASE_DIR, 'tuned_stacking_results')
os.makedirs(SAVE_DIR, exist_ok=True)

# 核心代谢特征
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']

# 调整后的基模型参数
XGBOOST_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 0.85,
    'scale_pos_weight': None,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

LIGHTGBM_PARAMS = {
    'learning_rate': 0.03,
    'max_depth': 5,
    'min_child_samples': 20,
    'n_estimators': 250,
    'num_leaves': 25,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE
}

LOGISTIC_PARAMS = {
    'C': 0.07,
    'penalty': 'l2',
    'solver': 'liblinear',
    'class_weight': 'balanced',
    'max_iter': 5000,
    'random_state': RANDOM_STATE
}

RANDOM_FOREST_PARAMS = {
    'bootstrap': True,
    'class_weight': 'balanced',
    'max_depth': 11,
    'max_features': 'sqrt',
    'min_impurity_decrease': 0.0007143340896097039,
    'min_samples_leaf': 10,
    'min_samples_split': 60,
    'n_estimators': 180,
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}

# 自定义评估指标
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# 数据预处理（增强缺失值处理）
def preprocess_data(train_path, test_path, label_col='diabe'):
    def safe_read(file):
        try:
            return pd.read_csv(file, encoding='utf-8')
        except:
            return pd.read_csv(file, encoding='gbk')
    
    # 加载数据
    train = safe_read(train_path)
    test = safe_read(test_path)
    print(f"✅ 数据加载完成：训练集{train.shape[0]}例，测试集{test.shape[0]}例")

    # 医学特征工程
    def create_medical_features(df):
        df['glu_hbalc_ratio'] = df['bl_glu'] / (df['bl_hbalc'] + 1e-6)  # 糖代谢指数
        df['tyg_bmi_squared'] = df['tyg_bmi'] **2  # 胰岛素抵抗相关
        df['metabolic_abnormality'] = ((df['bl_glu'] > 7.0) | (df['bl_hbalc'] > 6.5)).astype(int)
        return df
    
    train = create_medical_features(train)
    test = create_medical_features(test)

    # 提取标签
    y_train = train[label_col].map({'是':1, '否':0, 1:1, 0:0}).astype(int)
    y_test = test[label_col].map({'是':1, '否':0, 1:1, 0:0}).astype(int)
    X_train = train.drop(label_col, axis=1)
    X_test = test.drop(label_col, axis=1)

    # 特征分类（数值/分类）
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    print(f"📊 特征分类：数值特征{len(num_cols)}个，分类特征{len(cat_cols)}个")

    # ---------- 核心修正：缺失值处理管道 ----------
    # 数值特征：用中位数填充（适合临床指标，避免受极端值影响）
    # 分类特征：用最频繁值填充（适合类别变量）
    num_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 缺失值填充
        ('scaler', StandardScaler())  # 标准化
    ])
    
    cat_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 缺失值填充
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))  # 编码
    ])
    
    # 整合预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_preprocessor, num_cols),
            ('cat', cat_preprocessor, cat_cols)
        ])
    
    # 处理后的数据确保无NaN
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 验证是否还有缺失值
    if np.isnan(X_train_processed).any():
        print("⚠️ 警告：训练集仍有缺失值，请检查预处理步骤！")
    if np.isnan(X_test_processed).any():
        print("⚠️ 警告：测试集仍有缺失值，请检查预处理步骤！")
    
    return X_train_processed, y_train, X_test_processed, y_test, preprocessor

# 构建Stacking集成模型
def build_stacking_ensemble(n_pos, n_neg):
    XGBOOST_PARAMS['scale_pos_weight'] = n_neg / n_pos  # 处理不平衡
    
    # 基模型定义（使用调整后参数）
    base_estimators = [
        ('logistic', LogisticRegression(** LOGISTIC_PARAMS)),
        ('random_forest', RandomForestClassifier(**RANDOM_FOREST_PARAMS)),
        ('xgboost', xgb.XGBClassifier(** XGBOOST_PARAMS)),
        ('lightgbm', lgb.LGBMClassifier(**LIGHTGBM_PARAMS))
    ]
    
    # 元模型（逻辑回归，支持概率输出）
    meta_model = LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    
    # 构建Stacking模型
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method='predict_proba',
        n_jobs=1,  # 单进程避免编码问题
        passthrough=False
    )
    
    return stacking_clf, base_estimators

# 模型评估与可视化
class Evaluator:
    @staticmethod
    def evaluate(y_true, y_proba, model_name):
        auc = roc_auc_score(y_true, y_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        best_threshold = thresholds[np.argmax(tpr - fpr)]  # 约登指数最优
        y_pred = (y_proba >= best_threshold).astype(int)
        
        sensitivity = recall_score(y_true, y_pred)
        specificity = specificity_score(y_true, y_pred)
        
        print(f"\n======= {model_name} 性能评估 =======")
        print(f"AUC: {auc:.4f}")
        print(f"最佳阈值: {best_threshold:.4f}")
        print(f"灵敏度: {sensitivity:.4f} (漏诊率: {1-sensitivity:.4f})")
        print(f"特异度: {specificity:.4f} (误诊率: {1-specificity:.4f})")
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=['非糖尿病', '糖尿病']))
        
        return {'auc': auc, 'y_proba': y_proba, 'threshold': best_threshold}
    
    @staticmethod
    def plot_roc_comparison(y_true, proba_dict):
        plt.figure(figsize=(10, 8))
        for name, proba in proba_dict.items():
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc = roc_auc_score(y_true, proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        plt.xlabel('假阳性率 (误诊率)', fontsize=12)
        plt.ylabel('真阳性率 (灵敏度)', fontsize=12)
        plt.title('Stacking集成与基模型ROC曲线对比', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'roc_comparison.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ ROC对比图已保存至 {SAVE_DIR}")
    
    @staticmethod
    def plot_meta_weights(meta_model, base_names):
        weights = np.abs(meta_model.coef_[0])
        weights = weights / np.sum(weights)
        
        plt.figure(figsize=(10, 6))
        plt.bar(base_names, weights, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
        for i, w in enumerate(weights):
            plt.text(i, w+0.01, f'{w:.4f}', ha='center')
        
        plt.title('元模型对基模型输出的权重分配', fontsize=14)
        plt.xlabel('基模型', fontsize=12)
        plt.ylabel('归一化权重', fontsize=12)
        plt.ylim(0, max(weights)+0.1)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'meta_weights.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ 元模型权重图已保存至 {SAVE_DIR}")

# 主函数
def main():
    # 数据预处理（含缺失值处理）
    X_train, y_train, X_test, y_test, preprocessor = preprocess_data(
        os.path.join(BASE_DIR, 'train_dataset_optimized.csv'),
        os.path.join(BASE_DIR, 'test_dataset_optimized.csv')
    )
    
    # 处理数据不平衡（糖尿病数据通常正样本少）
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    print(f"训练集正负样本比: {n_pos}:{n_neg} ({n_pos/len(y_train):.1%}患病)")
    if n_pos / len(y_train) < 0.2:
        print("⚠️ 检测到数据不平衡，已通过模型参数（class_weight/scale_pos_weight）处理")
    
    # 构建并训练Stacking模型
    print("\n======= 训练调整参数后的Stacking集成模型 =======")
    stacking_clf, base_estimators = build_stacking_ensemble(n_pos, n_neg)
    stacking_clf.fit(X_train, y_train)
    print("✅ Stacking模型训练完成")
    
    # 预测与评估
    evaluator = Evaluator()
    proba_dict = {}
    base_names = [name for name, _ in base_estimators]
    
    # Stacking集成预测
    proba_dict['Stacking集成'] = stacking_clf.predict_proba(X_test)[:, 1]
    evaluator.evaluate(y_test, proba_dict['Stacking集成'], 'Stacking集成')
    
    # 基模型单独预测
    for name, _ in base_estimators:
        proba = stacking_clf.named_estimators_[name].predict_proba(X_test)[:, 1]
        proba_dict[name] = proba
        evaluator.evaluate(y_test, proba, name)
    
    # 可视化
    evaluator.plot_roc_comparison(y_test, proba_dict)
    evaluator.plot_meta_weights(stacking_clf.final_estimator_, base_names)
    
    # 保存模型
    joblib.dump({
        'stacking_clf': stacking_clf,
        'preprocessor': preprocessor,
        'proba_dict': proba_dict
    }, os.path.join(SAVE_DIR, 'tuned_stacking_model.pkl'))
    print(f"\n✅ 模型已保存至 {SAVE_DIR}")

if __name__ == "__main__":
    main()