import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tempfile
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report, recall_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings

# ---------- 解决中文路径编码问题 ----------
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()
# -----------------------------------------

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning, message="findfont")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 配置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 全局配置参数
RANDOM_STATE = 42
BASE_DIR = os.getcwd()
SAVE_DIR = os.path.join(BASE_DIR, 'stacking_4models_results')
os.makedirs(SAVE_DIR, exist_ok=True)

# 4个基础模型的最佳参数
BEST_PARAMS = {
    'logistic': {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'},
    'random_forest': {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 10},
    'xgboost': {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 5},
    'lightgbm': {'learning_rate': 0.05, 'n_estimators': 150, 'num_leaves': 31}
}

# 自定义特异度评估函数
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# 数据预处理函数
def robust_data_preprocessing(train_path, test_path, label_col='diabe'):
    def safe_read(file):
        try:
            return pd.read_csv(file, encoding='utf-8')
        except:
            return pd.read_csv(file, encoding='gbk')
    
    train = safe_read(train_path)
    test = safe_read(test_path)
    print(f"✅ 加载数据：训练集{train.shape[0]}例，测试集{test.shape[0]}例")

    # 标签清洗
    def clean_label(df):
        y = df[label_col].map({'是':1, '否':0, 1:1, 0:0}).astype(int)
        return df.drop(label_col, axis=1), y
    
    X_train, y_train = clean_label(train)
    X_test, y_test = clean_label(test)

    # 特征分类
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    
    # 缺失值处理
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # 特征缩放和编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return (X_train_processed, y_train, X_test_processed, y_test, 
            preprocessor, train, test)

# 构建Stacking集成模型（使用支持概率输出的元模型）
def build_stacking_ensemble(n_pos, n_neg):
    # 基础模型（第一层）
    base_estimators = [
        ('logistic', LogisticRegression(
            C=BEST_PARAMS['logistic']['C'],
            penalty=BEST_PARAMS['logistic']['penalty'],
            solver=BEST_PARAMS['logistic']['solver'],
            class_weight='balanced',
            max_iter=5000,
            random_state=RANDOM_STATE
        )),
        ('random_forest', RandomForestClassifier(
            n_estimators=BEST_PARAMS['random_forest']['n_estimators'],
            max_depth=BEST_PARAMS['random_forest']['max_depth'],
            min_samples_split=BEST_PARAMS['random_forest']['min_samples_split'],
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
        ('xgboost', xgb.XGBClassifier(
            learning_rate=BEST_PARAMS['xgboost']['learning_rate'],
            n_estimators=BEST_PARAMS['xgboost']['n_estimators'],
            max_depth=BEST_PARAMS['xgboost']['max_depth'],
            scale_pos_weight=n_neg / n_pos,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )),
        ('lightgbm', lgb.LGBMClassifier(
            learning_rate=BEST_PARAMS['lightgbm']['learning_rate'],
            n_estimators=BEST_PARAMS['lightgbm']['n_estimators'],
            num_leaves=BEST_PARAMS['lightgbm']['num_leaves'],
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ]
    
    # 元模型（第二层）：使用支持predict_proba的逻辑回归
    meta_model = LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    
    # 构建Stacking集成模型
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method='predict_proba',  # 基于概率输出堆叠
        n_jobs=1,  # 单进程避免编码问题
        passthrough=False
    )
    
    return stacking_clf, base_estimators

# 评估与可视化工具
class EnsembleEvaluator:
    @staticmethod
    def evaluate(y_true, y_proba, model_name, save_dir):
        auc = roc_auc_score(y_true, y_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        y_pred = (y_proba >= best_threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = recall_score(y_true, y_pred)
        specificity = specificity_score(y_true, y_pred)
        
        print(f"\n======= {model_name}性能评估 =======")
        print(f"AUC: {auc:.4f}")
        print(f"最佳阈值: {best_threshold:.4f}")
        print(f"灵敏度: {sensitivity:.4f}")
        print(f"特异度: {specificity:.4f}")
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=['非糖尿病', '糖尿病']))
        
        return {
            'auc': auc,
            'best_threshold': best_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'y_proba': y_proba,
            'y_pred': y_pred
        }
    
    @staticmethod
    def plot_roc_comparison(y_true, proba_dict, save_dir):
        plt.figure(figsize=(10, 8))
        
        for name, proba in proba_dict.items():
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc = roc_auc_score(y_true, proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        plt.xlabel('假阳性率', fontsize=12)
        plt.ylabel('真阳性率', fontsize=12)
        plt.title('Stacking集成 vs 基础模型 ROC曲线对比', fontsize=14, pad=20)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'roc_comparison.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ ROC对比图已保存至: {os.path.join(save_dir, 'roc_comparison.png')}")
    
    @staticmethod
    def plot_meta_model_weights(meta_model, base_model_names, save_dir):
        # 提取元模型系数（逻辑回归的系数代表对基础模型输出的权重）
        weights = np.abs(meta_model.coef_[0])
        weights = weights / np.sum(weights)  # 归一化
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(base_model_names, weights, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title('元模型对基础模型输出的权重分配', fontsize=14, pad=20)
        plt.xlabel('基础模型', fontsize=12)
        plt.ylabel('归一化权重', fontsize=12)
        plt.ylim(0, max(weights) + 0.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'meta_model_weights.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ 元模型权重图已保存至: {os.path.join(save_dir, 'meta_model_weights.png')}")

# 主函数
def main():
    print("======= 数据预处理 =======")
    (X_train, y_train, X_test, y_test, 
     preprocessor, train_df, test_df) = robust_data_preprocessing(
        os.path.join(BASE_DIR, 'train_dataset_optimized.csv'),
        os.path.join(BASE_DIR, 'test_dataset_optimized.csv')
    )
    
    # 计算正负样本比例
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    
    # 构建Stacking集成模型
    print("\n======= 构建Stacking集成模型 =======")
    stacking_clf, base_estimators = build_stacking_ensemble(n_pos, n_neg)
    
    # 训练Stacking模型
    print("开始训练Stacking集成模型...")
    stacking_clf.fit(X_train, y_train)
    print("Stacking集成模型训练完成!")
    
    # 收集各模型预测概率
    proba_dict = {}
    # Stacking集成模型预测（此时元模型支持predict_proba）
    proba_dict['Stacking集成'] = stacking_clf.predict_proba(X_test)[:, 1]
    
    # 各基础模型预测
    base_model_names = [name for name, _ in base_estimators]
    for name, _ in base_estimators:
        proba_dict[name] = stacking_clf.named_estimators_[name].predict_proba(X_test)[:, 1]
    
    # 评估所有模型
    evaluator = EnsembleEvaluator()
    results = {}
    for name, proba in proba_dict.items():
        results[name] = evaluator.evaluate(y_test, proba, name, SAVE_DIR)
    
    # 可视化对比
    evaluator.plot_roc_comparison(y_test, proba_dict, SAVE_DIR)
    evaluator.plot_meta_model_weights(
        stacking_clf.final_estimator_, 
        base_model_names, 
        SAVE_DIR
    )
    
    # 保存模型与结果
    joblib.dump({
        'stacking_clf': stacking_clf,
        'preprocessor': preprocessor,
        'results': results
    }, os.path.join(SAVE_DIR, 'stacking_ensemble_model.pkl'))
    
    # 保存预测概率
    prob_df = pd.DataFrame({
        'true_label': y_test
    })
    for name, proba in proba_dict.items():
        prob_df[f'{name}_proba'] = proba
    
    prob_df.to_csv(os.path.join(SAVE_DIR, 'prediction_probabilities.csv'), index=False, encoding='utf-8-sig')
    
    print("\n✅ 所有结果已保存至:", SAVE_DIR)

if __name__ == "__main__":
    main()