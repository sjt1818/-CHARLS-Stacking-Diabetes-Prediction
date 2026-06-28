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
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings

# 基础配置（解决编码、警告、中文字体问题）
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 全局参数
RANDOM_STATE = 42
BASE_DIR = os.getcwd()
SAVE_DIR = os.path.join(BASE_DIR, 'tuned_stacking_results')
os.makedirs(SAVE_DIR, exist_ok=True)

# 核心代谢特征（不变）
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']

# 基模型参数（不变，已优化）
XGBOOST_PARAMS = {
    'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100,
    'subsample': 0.85, 'scale_pos_weight': None, 'random_state': RANDOM_STATE,
    'use_label_encoder': False, 'eval_metric': 'logloss'
}
LIGHTGBM_PARAMS = {
    'learning_rate': 0.03, 'max_depth': 5, 'min_child_samples': 20,
    'n_estimators': 250, 'num_leaves': 25, 'class_weight': 'balanced',
    'random_state': RANDOM_STATE
}
LOGISTIC_PARAMS = {
    'C': 0.07, 'penalty': 'l2', 'solver': 'liblinear',
    'class_weight': 'balanced', 'max_iter': 5000, 'random_state': RANDOM_STATE
}
RANDOM_FOREST_PARAMS = {
    'bootstrap': True, 'class_weight': 'balanced', 'max_depth': 11,
    'max_features': 'sqrt', 'min_impurity_decrease': 0.0007143340896097039,
    'min_samples_leaf': 10, 'min_samples_split': 60, 'n_estimators': 180,
    'random_state': RANDOM_STATE, 'n_jobs': 1
}

# 自定义特异度计算（不变）
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# 数据预处理（删除hospital_level相关逻辑，恢复基础功能）
def preprocess_data(train_path, test_path, label_col='diabe'):
    # 安全读取数据（兼容utf-8/gbk编码）
    def safe_read(file):
        try:
            return pd.read_csv(file, encoding='utf-8')
        except:
            return pd.read_csv(file, encoding='gbk')
    
    # 加载数据
    train = safe_read(train_path)
    test = safe_read(test_path)
    print(f"✅ 数据加载完成：训练集{train.shape[0]}例，测试集{test.shape[0]}例")

    # 医学特征工程（不变，保留生理意义特征）
    def create_medical_features(df):
        df['glu_hbalc_ratio'] = df['bl_glu'] / (df['bl_hbalc'] + 1e-6)  # 糖代谢指数
        df['tyg_bmi_squared'] = df['tyg_bmi'] **2  # 胰岛素抵抗关联特征
        df['metabolic_abnormality'] = ((df['bl_glu'] > 7.0) | (df['bl_hbalc'] > 6.5)).astype(int)
        return df
    
    train = create_medical_features(train)
    test = create_medical_features(test)

    # 提取标签（统一编码为0/1）
    y_train = train[label_col].map({'是':1, '否':0, 1:1, 0:0}).astype(int)
    y_test = test[label_col].map({'是':1, '否':0, 1:1, 0:0}).astype(int)
    X_train = train.drop(label_col, axis=1)
    X_test = test.drop(label_col, axis=1)

    # 特征分类（数值/分类，适配预处理管道）
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    print(f"📊 特征分类：数值特征{len(num_cols)}个，分类特征{len(cat_cols)}个")

    # 数据预处理管道（缺失值+标准化/编码，医疗数据适配）
    num_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 数值特征用中位数填缺失（抗极端值）
        ('scaler', StandardScaler())  # 标准化消除量纲
    ])
    cat_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 分类特征用最频繁值填缺失
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))  # 独热编码（避免多重共线性）
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_preprocessor, num_cols),
            ('cat', cat_preprocessor, cat_cols)
        ])

    # 执行预处理（确保无缺失值）
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    if np.isnan(X_train_processed).any() or np.isnan(X_test_processed).any():
        print("⚠️ 警告：数据仍存在缺失值，请检查预处理步骤！")

    return X_train_processed, y_train, X_test_processed, y_test, preprocessor

# 构建Stacking集成模型（不变，保留最优基模型组合）
def build_stacking_ensemble(n_pos, n_neg):
    # 处理数据不平衡（XGBoost专用参数）
    XGBOOST_PARAMS['scale_pos_weight'] = n_neg / n_pos
    
    # 基模型（逻辑回归+随机森林+XGBoost+LightGBM，异构融合）
    base_estimators = [
        ('logistic', LogisticRegression(** LOGISTIC_PARAMS)),
        ('random_forest', RandomForestClassifier(**RANDOM_FOREST_PARAMS)),
        ('xgboost', xgb.XGBClassifier(** XGBOOST_PARAMS)),
        ('lightgbm', lgb.LGBMClassifier(**LIGHTGBM_PARAMS))
    ]
    
    # 元模型（用简单逻辑回归，避免过拟合，支持概率输出）
    meta_model = LogisticRegression(
        C=0.1, penalty='l2', solver='liblinear',
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
    )
    
    # 构建Stacking（5折交叉验证融合基模型输出）
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method='predict_proba',  # 用概率输出做元模型输入（更精准）
        n_jobs=1,  # 单进程避免编码冲突
        passthrough=False
    )
    return stacking_clf, base_estimators

# 核心：评估器（新增加权约登指数计算场景化阈值）
class Evaluator:
    @staticmethod
    def calculate_scene_thresholds(y_true, y_proba):
        """
        基于单一数据集，用加权约登指数计算3类阈值：
        1. 标准阈值：平衡灵敏度+特异度（无权重）
        2. 基层医院阈值：优先高灵敏度（权重：灵敏度1.2，特异度0.8）
        3. 三级医院阈值：优先高特异度（权重：灵敏度0.8，特异度1.2）
        """
        # 计算ROC曲线基础数据（fpr=1-特异度，tpr=灵敏度）
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # 1. 标准约登指数（平衡场景）
        standard_j = tpr - fpr  # 约登指数=灵敏度+特异度-1 = tpr + (1-fpr) -1 = tpr - fpr
        standard_idx = np.argmax(standard_j)
        standard_threshold = thresholds[standard_idx]
        
        # 2. 基层医院：加权约登指数（灵敏度权重1.2，特异度权重0.8）
        primary_j = 1.2 * tpr + 0.8 * (1 - fpr) - 1  # 加权后约登指数
        primary_idx = np.argmax(primary_j)
        primary_threshold = thresholds[primary_idx]
        
        # 3. 三级医院：加权约登指数（灵敏度权重0.8，特异度权重1.2）
        tertiary_j = 0.8 * tpr + 1.2 * (1 - fpr) - 1  # 加权后约登指数
        tertiary_idx = np.argmax(tertiary_j)
        tertiary_threshold = thresholds[tertiary_idx]
        
        # 返回阈值及对应ROC点（用于可视化）
        return {
            'standard': {'threshold': standard_threshold, 'fpr': fpr[standard_idx], 'tpr': tpr[standard_idx]},
            'primary': {'threshold': primary_threshold, 'fpr': fpr[primary_idx], 'tpr': tpr[primary_idx]},
            'tertiary': {'threshold': tertiary_threshold, 'fpr': fpr[tertiary_idx], 'tpr': tpr[tertiary_idx]}
        }
    
    @staticmethod
    def evaluate_scenario_perf(y_true, y_proba, threshold, scene_name):
        """评估某一场景阈值的性能（输出临床关心的指标）"""
        y_pred = (y_proba >= threshold).astype(int)
        sensitivity = recall_score(y_true, y_pred)  # 灵敏度=不漏诊率
        specificity = specificity_score(y_true, y_pred)  # 特异度=不误诊率
        auc = roc_auc_score(y_true, y_proba)  # 整体区分度
        
        # 打印场景性能（临床语言适配）
        print(f"\n----- {scene_name} 场景性能 -----")
        print(f"阈值: {threshold:.4f}")
        print(f"AUC: {auc:.4f} | 灵敏度: {sensitivity:.4f} (漏诊率: {1-sensitivity:.4f})")
        print(f"特异度: {specificity:.4f} (误诊率: {1-specificity:.4f})")
        print("分类报告（临床人群适配）:")
        print(classification_report(y_true, y_pred, target_names=['非糖尿病', '糖尿病'], digits=3))
        
        return {'auc': auc, 'sensitivity': sensitivity, 'specificity': specificity}
    
    @staticmethod
    def evaluate_model(y_true, y_proba, model_name):
        """完整评估：计算3类阈值+输出所有场景性能"""
        print(f"\n======= {model_name} 完整性能评估 =======")
        # 1. 计算3类场景阈值
        thresholds = Evaluator.calculate_scene_thresholds(y_true, y_proba)
        
        # 2. 评估各场景性能
        standard_perf = Evaluator.evaluate_scenario_perf(y_true, y_proba, thresholds['standard']['threshold'], "标准平衡")
        primary_perf = Evaluator.evaluate_scenario_perf(y_true, y_proba, thresholds['primary']['threshold'], "基层医院")
        tertiary_perf = Evaluator.evaluate_scenario_perf(y_true, y_proba, thresholds['tertiary']['threshold'], "三级医院")
        
        # 3. 整合结果（返回阈值+性能，用于后续保存/可视化）
        return {
            'thresholds': thresholds,
            'performances': {
                'standard': standard_perf,
                'primary': primary_perf,
                'tertiary': tertiary_perf
            },
            'y_proba': y_proba  # 保留预测概率，用于ROC图
        }
    
    @staticmethod
    def plot_roc_with_scenes(y_true, model_results):
        """绘制ROC曲线，标记3类场景阈值点（核心可视化，体现创新点）"""
        plt.figure(figsize=(10, 8))
        
        # 遍历所有模型绘制ROC曲线
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']  # 模型颜色区分
        for i, (model_name, result) in enumerate(model_results.items()):
            y_proba = result['y_proba']
            fpr_full, tpr_full, _ = roc_curve(y_true, y_proba)
            auc = result['performances']['standard']['auc']
            
            # 绘制ROC曲线
            plt.plot(fpr_full, tpr_full, color=colors[i], linewidth=2, 
                     label=f'{model_name} (AUC={auc:.4f})')
            
            # 标记3类场景阈值对应的ROC点
            thresholds = result['thresholds']
            # 标准阈值（黑色圆点）
            plt.scatter(thresholds['standard']['fpr'], thresholds['standard']['tpr'], 
                       color='black', s=80, zorder=5, marker='o', 
                       label=f'{model_name} 标准阈值' if i==0 else "")
            # 基层医院阈值（蓝色三角）
            plt.scatter(thresholds['primary']['fpr'], thresholds['primary']['tpr'], 
                       color='blue', s=80, zorder=5, marker='^', 
                       label=f'{model_name} 基层阈值' if i==0 else "")
            # 三级医院阈值（红色方形）
            plt.scatter(thresholds['tertiary']['fpr'], thresholds['tertiary']['tpr'], 
                       color='red', s=80, zorder=5, marker='s', 
                       label=f'{model_name} 三级阈值' if i==0 else "")
        
        # 辅助线与标签
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测', alpha=0.5)
        plt.xlabel('假阳性率（误诊率）', fontsize=12)
        plt.ylabel('真阳性率（灵敏度）', fontsize=12)
        plt.title('各模型ROC曲线及场景化阈值标记', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'roc_with_scene_thresholds.png'), bbox_inches='tight')
        plt.close()
        print(f"\n✅ ROC曲线（含场景阈值）已保存至 {SAVE_DIR}")
    
    @staticmethod
    def plot_threshold_comparison(model_results):
        """绘制各模型的3类场景阈值对比图（论文核心图表）"""
        plt.figure(figsize=(12, 6))
        
        # 整理数据：模型名 + 3类阈值
        model_names = list(model_results.keys())
        standard_thresholds = [model_results[m]['thresholds']['standard']['threshold'] for m in model_names]
        primary_thresholds = [model_results[m]['thresholds']['primary']['threshold'] for m in model_names]
        tertiary_thresholds = [model_results[m]['thresholds']['tertiary']['threshold'] for m in model_names]
        
        # 设置柱状图位置
        x = np.arange(len(model_names))
        width = 0.25  # 柱子宽度
        
        # 绘制3类阈值柱状图
        bars1 = plt.bar(x - width, standard_thresholds, width, label='标准平衡阈值', color='#9C27B0')
        bars2 = plt.bar(x, primary_thresholds, width, label='基层医院阈值', color='#2196F3')
        bars3 = plt.bar(x + width, tertiary_thresholds, width, label='三级医院阈值', color='#FF9800')
        
        # 柱子上标注阈值数值
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 图表美化（适配论文风格）
        plt.xlabel('模型类型', fontsize=12)
        plt.ylabel('约登指数最优阈值', fontsize=12)
        plt.title('各模型在不同临床场景下的阈值对比', fontsize=14)
        plt.xticks(x, model_names, rotation=0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'model_threshold_comparison.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ 模型阈值对比图已保存至 {SAVE_DIR}")
    
    @staticmethod
    def plot_meta_weights(meta_model, base_names):
        """绘制元模型对基模型的权重分配（体现Stacking融合逻辑）"""
        weights = np.abs(meta_model.coef_[0])  # 取绝对值（权重方向不影响贡献度）
        weights = weights / np.sum(weights)  # 归一化（便于对比）
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(base_names, weights, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
        
        # 标注权重数值
        for i, w in enumerate(weights):
            plt.text(i, w + 0.01, f'{w:.4f}', ha='center', fontsize=10)
        
        plt.title('Stacking元模型对基模型的归一化权重分配', fontsize=14)
        plt.xlabel('基模型', fontsize=12)
        plt.ylabel('归一化权重（贡献度）', fontsize=12)
        plt.ylim(0, max(weights) + 0.1)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'meta_model_weights.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ 元模型权重图已保存至 {SAVE_DIR}")

# 主函数（串联数据-模型-评估全流程）
def main():
    # 1. 数据预处理（无hospital_level，纯单一数据集）
    X_train, y_train, X_test, y_test, preprocessor = preprocess_data(
        os.path.join(BASE_DIR, 'train_dataset_optimized.csv'),
        os.path.join(BASE_DIR, 'test_dataset_optimized.csv')
    )
    
    # 2. 数据不平衡处理（打印信息，确认模型参数适配）
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    print(f"\n📈 训练集分布：阳性样本{n_pos}例，阴性样本{n_neg}例，占比{n_pos/len(y_train):.1%}")
    if n_pos / len(y_train) < 0.2:
        print("⚠️ 检测到数据不平衡，已通过class_weight/scale_pos_weight参数处理")
    
    # 3. 训练Stacking模型
    print("\n======= 训练Stacking集成模型 =======")
    stacking_clf, base_estimators = build_stacking_ensemble(n_pos, n_neg)
    stacking_clf.fit(X_train, y_train)
    print("✅ Stacking模型训练完成")
    
    # 4. 模型预测与评估（核心：计算场景化阈值）
    evaluator = Evaluator()
    model_results = {}  # 存储所有模型的“阈值+性能”结果
    base_names = [name for name, _ in base_estimators]
    
    # 4.1 Stacking集成模型评估
    stacking_proba = stacking_clf.predict_proba(X_test)[:, 1]
    model_results['Stacking集成'] = evaluator.evaluate_model(y_test, stacking_proba, 'Stacking集成')
    
    # 4.2 基模型单独评估（对比用）
    for name, _ in base_estimators:
        base_proba = stacking_clf.named_estimators_[name].predict_proba(X_test)[:, 1]
        model_results[name] = evaluator.evaluate_model(y_test, base_proba, name)
    
    # 5. 可视化核心结果（论文用图）
    evaluator.plot_roc_with_scenes(y_test, model_results)  # ROC+阈值标记
    evaluator.plot_threshold_comparison(model_results)    # 阈值对比
    evaluator.plot_meta_weights(stacking_clf.final_estimator_, base_names)  # 元模型权重
    
    # 6. 保存模型与结果（便于后续复用）
    joblib.dump({
        'stacking_clf': stacking_clf,
        'preprocessor': preprocessor,
        'model_results': model_results,  # 含所有场景的阈值+性能
        'y_test': y_test  # 保留测试集标签，便于后续复现
    }, os.path.join(SAVE_DIR, 'stacking_model_with_scene_thresholds.pkl'))
    print(f"\n✅ 模型及场景化结果已保存至 {SAVE_DIR}")

if __name__ == "__main__":
    main()