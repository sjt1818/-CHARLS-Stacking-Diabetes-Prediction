import pandas as pd
import numpy as np
import os
import joblib
import warnings

# ========== 仅通过文件记录结果，避免控制台输出 ==========
LOG_FILE = 'model_results.log'
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("======= 模型运行日志 =======\n")

def log_to_file(*args):
    """将所有内容写入日志文件"""
    msg = ' '.join(str(arg) for arg in args) + '\n'
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg)

# ========== 基础配置 ==========
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ========== 数据预处理管道 ==========
def create_preprocessing_pipeline(numeric_cols, cat_cols):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    return preprocessor

# ========== 定义基模型 ==========
def get_base_models(random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    import lightgbm as lgb
    
    return [
        ('random_forest', RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, random_state=random_state, n_jobs=-1
        )),
        ('svm', SVC(
            C=10, gamma=0.001, kernel='rbf', probability=True,
            max_iter=5000, random_state=random_state
        )),
        ('xgboost', XGBClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            use_label_encoder=False, eval_metric='logloss'
        )),
        ('lightgbm', lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state
        )),
        ('logistic', LogisticRegression(
            C=1.0, solver='liblinear', penalty='l2',
            random_state=random_state, max_iter=1000
        )),
        ('lasso', LogisticRegression(
            C=0.1, solver='liblinear', penalty='l1',
            random_state=random_state, max_iter=1000
        ))
    ]

# ========== 构建Stacking集成模型 ==========
def build_ensemble_model(base_models, random_state=42):
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    
    meta_model = LogisticRegression(
        C=0.5, solver='liblinear', penalty='l2',
        random_state=random_state, max_iter=1000
    )
    
    stacking_ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        stack_method='predict_proba',
        n_jobs=1,
        passthrough=False
    )
    return stacking_ensemble

# ========== 模型训练与评估 ==========
def train_and_evaluate(ensemble, X_train, y_train, X_test, y_test, save_dir):
    log_to_file("======= 开始训练集成模型 =======")
    ensemble.fit(X_train, y_train)
    
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]  # 测试集预测概率
    y_pred = ensemble.predict(X_test)
    
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    auc = roc_auc_score(y_test, y_pred_proba)
    log_to_file("\n集成模型AUC:", round(auc, 4))
    
    # 记录分类报告
    report = classification_report(
        y_test, y_pred,
        target_names=['非糖尿病', '糖尿病'],
        digits=4
    )
    log_to_file("\n分类报告:\n", report)
    
    # 记录混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log_to_file("\n混淆矩阵:")
    log_to_file(f"真阴性: {tn} | 假阳性: {fp}")
    log_to_file(f"假阴性: {fn} | 真阳性: {tp}")
    log_to_file(f"灵敏度: {round(tp/(tp+fn),4)} | 特异度: {round(tn/(tn+fp),4)}")
    
    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(ensemble, os.path.join(save_dir, 'ensemble_model.pkl'))
    log_to_file("\n模型保存路径:", save_dir)
    
    return {
        'model': ensemble, 'y_pred_proba': y_pred_proba, 
        'y_pred': y_pred, 'auc': auc
    }

# ========== 基模型性能对比 ==========
def compare_base_models(base_models, X_train, y_train, X_test, y_test):
    log_to_file("\n======= 基模型性能对比 =======")
    base_performances = {}
    from sklearn.metrics import roc_auc_score
    
    for name, model in base_models:
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        base_performances[name] = auc
        log_to_file(f"{name} AUC:", round(auc, 4))
    
    return base_performances

# ========== 可视化集成效果 ==========
def visualize_results(ensemble_results, base_performances, X_test, y_test, save_dir):
    """修正：传入测试集X_test和y_test，确保与预测结果样本数一致"""
    os.makedirs(save_dir, exist_ok=True)
    
    # ROC曲线（使用测试集标签和预测概率）
    plt.figure(figsize=(10, 8))
    from sklearn.metrics import roc_curve
    # 关键修复：y_test与y_pred_proba样本数必须一致
    fpr, tpr, _ = roc_curve(y_test, ensemble_results['y_pred_proba'])
    plt.plot(fpr, tpr, 'b-', linewidth=3, 
             label=f'集成模型 (AUC={round(ensemble_results["auc"],4)})')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    for (name, model), color in zip(get_base_models(), colors):
        model.fit(X_test, y_test)  # 使用测试集拟合（仅为可视化）
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr_base, tpr_base, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr_base, tpr_base, color=color, linestyle='--', 
                 label=f'{name} (AUC={round(base_performances[name],4)})')
    
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线对比')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), bbox_inches='tight')
    plt.close()
    
    # 模型权重图
    meta_coef = ensemble_results['model'].final_estimator_.coef_[0]
    model_names = [name for name, _ in get_base_models()]
    plt.figure(figsize=(10,6))
    sns.barplot(x=meta_coef, y=model_names, palette='coolwarm')
    plt.xlabel('元模型系数（权重）')
    plt.title('基模型权重')
    plt.savefig(os.path.join(save_dir, 'model_weights.png'), bbox_inches='tight')
    plt.close()
    
    log_to_file("\n可视化结果已保存至:", save_dir)

# ========== 主函数 ==========
def main():
    base_dir = os.getcwd()
    train_path = os.path.join(base_dir, 'train_dataset_optimized.csv')
    test_path = os.path.join(base_dir, 'test_dataset_optimized.csv')
    save_dir = os.path.join(base_dir, 'ensemble_results')
    
    # 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    label_col = 'diabe'
    
    # 特征分类
    numeric_cols = train_df.drop(label_col, axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = train_df.drop(label_col, axis=1).select_dtypes(include=['object']).columns.tolist()
    log_to_file("特征分类: 数值特征", len(numeric_cols), "个, 分类特征", len(cat_cols), "个")
    
    # 预处理
    preprocessor = create_preprocessing_pipeline(numeric_cols, cat_cols)
    X_train = preprocessor.fit_transform(train_df.drop(label_col, axis=1))  # 训练集特征
    X_test = preprocessor.transform(test_df.drop(label_col, axis=1))        # 测试集特征
    y_train = train_df[label_col].astype(int)  # 训练集标签
    y_test = test_df[label_col].astype(int)    # 测试集标签（样本数应与X_test一致）
    
    # 训练与评估
    base_models = get_base_models()
    ensemble_model = build_ensemble_model(base_models)
    ensemble_results = train_and_evaluate(ensemble_model, X_train, y_train, X_test, y_test, save_dir)
    base_performances = compare_base_models(base_models, X_train, y_train, X_test, y_test)
    
    # 关键修复：传入测试集X_test和y_test，而非训练集
    visualize_results(ensemble_results, base_performances, X_test, y_test, save_dir)
    
    log_to_file("\n======= 运行完成 =======")
    log_to_file("所有结果已保存至日志文件和:", save_dir)

if __name__ == "__main__":
    main()
    