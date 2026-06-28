import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, make_scorer, recall_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel

# ========== 环境配置与常量定义 ==========
os.environ["PYTHONUTF8"] = "1"
warnings.filterwarnings('ignore')

# 可视化配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 核心代谢指标与最佳参数
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']
BEST_PARAMS = {
    'bootstrap': True,
    'class_weight': 'balanced',
    'max_depth': 9,
    'max_features': 'sqrt',
    'min_impurity_decrease': 0.0007143340896097039,
    'min_samples_leaf': 10,
    'min_samples_split': 51,
    'n_estimators': 202,
    'random_state': 42,
    'n_jobs': 1
}

# 临床相关参数（可根据指南调整）
CLINICAL_PARAMS = {
    '漏诊权重': 2.0,  # 漏诊危害是误诊的2倍
    '初筛阈值': 0.3,   # 基层初筛推荐阈值
    '确诊阈值': 0.6    # 确诊前筛查推荐阈值
}


# ========== 自定义评分函数 ==========
def specificity_score(y_true, y_pred):
    """计算特异度（真阴性率）"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def clinical_net_benefit(y_true, y_proba, threshold):
    """临床校准的净获益计算（考虑漏诊危害更大）"""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 应用临床权重：漏诊危害 > 误诊危害
    return (tp - fp * (1/CLINICAL_PARAMS['漏诊权重'])) / len(y_true)


# ========== 数据处理模块 ==========
def load_and_preprocess_data(train_path, test_path, label_col='diabe'):
    """加载并预处理数据，增加特征选择步骤"""
    def safe_read(file):
        """兼容多种编码读取CSV"""
        for encoding in ['utf-8', 'gbk']:
            try:
                return pd.read_csv(file, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解析文件：{file}")
    
    # 读取数据
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
        X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    # 特征选择（基于重要性筛选）
    selector = SelectFromModel(
        RandomForestClassifier(**BEST_PARAMS),
        threshold='mean'  # 保留重要性高于均值的特征
    )
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    selected_features = np.array(selector.get_feature_names_out())
    
    print(f"\n======= 数据验证 =======")
    print(f"训练集样本：{len(X_train)} | 患病率：{y_train.mean():.2%}（{y_train.sum()}例）")
    print(f"测试集样本：{len(X_test)} | 患病率：{y_test.mean():.2%}（{y_test.sum()}例）")
    print(f"筛选后特征数：{X_train.shape[1]}（原始{len(num_cols)+len(cat_cols)}个）")

    return X_train, y_train, X_test, y_test, train_df, test_df, selected_features


# ========== 模型训练与评估模块 ==========
def train_model(X_train, y_train):
    """使用最佳参数训练模型"""
    print("\n📊 使用最佳参数训练模型...")
    model = RandomForestClassifier(** BEST_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, save_dir, selected_features):
    """全面评估模型性能，增加临床场景指标"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # 多阈值评估（默认+临床场景）
    thresholds_to_eval = [
        thresholds[np.argmax(tpr - fpr)],  # 约登指数最佳阈值
        CLINICAL_PARAMS['初筛阈值'],       # 基层初筛阈值
        CLINICAL_PARAMS['确诊阈值']        # 确诊前阈值
    ]
    
    # 计算核心指标
    metrics = {}
    for threshold in thresholds_to_eval:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics[threshold] = {
            '灵敏度': tp / (tp + fn) if (tp + fn) > 0 else 0,
            '特异度': tn / (tn + fp) if (tn + fp) > 0 else 0,
            '阳性预测值': tp / (tp + fp) if (tp + fp) > 0 else 0,
            '阴性预测值': tn / (tn + fn) if (tn + fn) > 0 else 0,
            '临床净获益': clinical_net_benefit(y_test, y_pred_proba, threshold)
        }
    
    # 整体AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 输出评估结果
    print("\n======= 模型评估结果 =======")
    print(f"AUC：{auc:.4f} | 最佳阈值：{thresholds_to_eval[0]:.3f}")
    print("\n📊 多场景阈值指标：")
    for i, threshold in enumerate(thresholds_to_eval):
        scenario = ["约登最佳", "基层初筛", "确诊前筛查"][i]
        print(f"\n【{scenario}（阈值={threshold:.3f}）】")
        print(f"   灵敏度：{metrics[threshold]['灵敏度']:.4f} | 特异度：{metrics[threshold]['特异度']:.4f}")
        print(f"   阳性预测值：{metrics[threshold]['阳性预测值']:.4f} | 阴性预测值：{metrics[threshold]['阴性预测值']:.4f}")
        print(f"   临床净获益：{metrics[threshold]['临床净获益']:.4f}")
    
    # 详细混淆矩阵（最佳阈值）
    best_y_pred = (y_pred_proba >= thresholds_to_eval[0]).astype(int)
    cm = confusion_matrix(y_test, best_y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n📊 最佳阈值混淆矩阵：")
    print(f"非糖尿病正确：{tn} | 误诊：{fp}")
    print(f"糖尿病漏诊：{fn} | 正确识别：{tp}")
    print("\n📋 分类报告：")
    print(classification_report(
        y_test, best_y_pred, target_names=['非糖尿病', '糖尿病'], digits=4
    ))

    return {
        'auc': auc,
        'thresholds': thresholds_to_eval,
        'metrics': metrics,
        'y_pred_proba': y_pred_proba,
        'y_pred': best_y_pred,
        'fpr': fpr,
        'tpr': tpr
    }


# ========== 可视化模块 ==========
def plot_roc_curve(results, save_dir):
    """绘制ROC曲线，标注多阈值点"""
    plt.figure(figsize=(8, 6))
    plt.plot(
        results['fpr'], results['tpr'], 
        color='#E63946', linewidth=3, 
        label=f'随机森林 (AUC={results["auc"]:.4f})'
    )
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='随机猜测')
    
    # 标注多个阈值点
    thresholds = results['thresholds']
    scenarios = ["约登最佳", "基层初筛", "确诊前筛查"]
    colors = ['#A23B72', '#457B9D', '#1D3557']
    
    for threshold, scenario, color in zip(thresholds, scenarios, colors):
        # 找到最接近该阈值的点
        idx = np.argmin(np.abs(results['fpr'] + results['tpr'] - 1 - (1 - 2*threshold)))
        plt.scatter(
            results['fpr'][idx], results['tpr'][idx], 
            color=color, s=100, zorder=5, label=f'{scenario} ({threshold:.3f})'
        )
    
    plt.xlabel('假阳性率（1-特异度）')
    plt.ylabel('真阳性率（灵敏度）')
    plt.title('随机森林ROC曲线（多场景阈值）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rf_roc_curve.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存：{os.path.join(save_dir, 'rf_roc_curve.png')}")


def plot_feature_importance(model, selected_features, save_dir):
    """绘制特征重要性，突出代谢指标"""
    importance = pd.DataFrame({
        '特征': selected_features,
        '重要性': model.feature_importances_,
        '是否代谢指标': [col in CORE_METABOLIC for col in selected_features]
    }).sort_values('重要性', ascending=False)

    # 可视化Top15特征
    top15 = importance.head(15)
    plt.figure(figsize=(10, 6))
    colors = ['#DC2626' if meta else '#64748B' for meta in top15['是否代谢指标']]
    sns.barplot(x='重要性', y='特征', data=top15, palette=colors)
    
    # 标记代谢指标
    for i, (_, row) in enumerate(top15.iterrows()):
        if row['是否代谢指标']:
            plt.text(row['重要性']+0.001, i, '✓ 代谢指标', va='center', color='#DC2626')
    
    plt.xlabel('特征重要性')
    plt.ylabel('特征名称')
    plt.title('随机森林核心特征重要性（Top15）')
    plt.grid(axis='x', alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rf_feature_importance.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 特征重要性图已保存：{os.path.join(save_dir, 'rf_feature_importance.png')}")

    return importance


def plot_clinical_dca(y_true, y_proba, save_dir):
    """绘制临床校准的决策曲线"""
    thresholds = np.linspace(0, 0.99, 100)
    
    # 计算三种策略的净获益
    nb_model = [clinical_net_benefit(y_true, y_proba, t) for t in thresholds]
    nb_all = [(sum(y_true) - len(y_true)*t/(CLINICAL_PARAMS['漏诊权重']))/len(y_true) for t in thresholds]
    nb_none = [0 for _ in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb_model, label='随机森林模型', color='#E63946', linewidth=3)
    plt.plot(thresholds, nb_all, label='全部筛查', color='gray', linestyle='--')
    plt.plot(thresholds, nb_none, label='不筛查', color='black', linestyle=':')
    
    # 标注临床推荐阈值
    for threshold, label in zip(
        [CLINICAL_PARAMS['初筛阈值'], CLINICAL_PARAMS['确诊阈值']],
        ['基层初筛阈值', '确诊前阈值']
    ):
        idx = np.argmin(np.abs(thresholds - threshold))
        plt.axvline(x=threshold, color='#457B9D', linestyle=':', alpha=0.7)
        plt.annotate(
            label, (threshold, nb_model[idx]),
            xytext=(5, 5), textcoords='offset points',
            color='#457B9D', fontweight='bold'
        )
    
    plt.xlabel('风险阈值（预测患病概率）')
    plt.ylabel('临床净获益（考虑漏诊危害更大）')
    plt.title('临床校准的决策曲线分析')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rf_clinical_dca.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 临床决策曲线已保存：{os.path.join(save_dir, 'rf_clinical_dca.png')}")


def plot_stratified_auc(stratified_results, save_dir):
    """可视化分层AUC结果（修复调色板错误）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 年龄分层
    age_data = pd.DataFrame(stratified_results['age'].items(), columns=['年龄组', 'AUC'])
    # 使用颜色列表（修复Seaborn调色板错误）
    sns.barplot(x='年龄组', y='AUC', data=age_data, ax=ax1, palette=['#A8DADC'])
    ax1.axhline(y=stratified_results['overall_auc'], color='r', linestyle='--', label=f'整体AUC: {stratified_results["overall_auc"]:.4f}')
    ax1.set_title('年龄分层AUC对比')
    ax1.set_ylim(0.5, 1.0)
    ax1.legend()
    
    # BMI分层
    bmi_data = pd.DataFrame(stratified_results['bmi'].items(), columns=['BMI组', 'AUC'])
    # 使用颜色列表（修复Seaborn调色板错误）
    sns.barplot(x='BMI组', y='AUC', data=bmi_data, ax=ax2, palette=['#F1FAEE'])
    ax2.axhline(y=stratified_results['overall_auc'], color='r', linestyle='--', label=f'整体AUC: {stratified_results["overall_auc"]:.4f}')
    ax2.set_title('BMI分层AUC对比')
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stratified_auc.png'), bbox_inches='tight')
    plt.close()
    print(f"✅ 分层AUC对比图已保存：{os.path.join(save_dir, 'stratified_auc.png')}")


# ========== 分层验证模块 ==========
def stratified_validation(model, X_test, y_test, original_test, overall_auc, save_dir):
    """增强版分层验证，增加可视化输出"""
    y_proba = model.predict_proba(X_test)[:, 1]
    original_test = original_test.reset_index(drop=True)
    
    # 年龄分层验证
    age_groups = pd.cut(original_test['age'], bins=[0, 60, 120], labels=['<60岁', '≥60岁'])
    age_auc = {}
    for g in age_groups.unique():
        mask = (age_groups == g)
        if sum(mask) > 0:
            age_auc[g] = roc_auc_score(y_test[mask], y_proba[mask])
    
    # BMI分层验证
    bmi_groups = pd.cut(original_test['bmi'], bins=[0, 18.5, 24, 28, 100], labels=['偏瘦', '正常', '超重', '肥胖'])
    bmi_auc = {}
    for g in bmi_groups.unique():
        if pd.isna(g):
            continue
        mask = (bmi_groups == g)
        if sum(mask) > 0:
            bmi_auc[g] = roc_auc_score(y_test[mask], y_proba[mask])
    
    # 输出结果
    print("\n======= 分层验证结果 =======")
    print("【年龄分层AUC】")
    for g, auc in age_auc.items():
        print(f"   {g}：{auc:.4f}")
    print("\n【BMI分层AUC】")
    for g, auc in bmi_auc.items():
        print(f"   {g}：{auc:.4f}")
    
    # 保存结果与可视化
    results = {
        'age': age_auc,
        'bmi': bmi_auc,
        'overall_auc': overall_auc
    }
    plot_stratified_auc(results, save_dir)
    
    pd.DataFrame({'年龄分组': age_auc.keys(), 'AUC': age_auc.values()}).to_csv(
        os.path.join(save_dir, 'stratified_age.csv'), index=False, encoding='utf-8-sig'
    )
    pd.DataFrame({'BMI分组': bmi_auc.keys(), 'AUC': bmi_auc.values()}).to_csv(
        os.path.join(save_dir, 'stratified_bmi.csv'), index=False, encoding='utf-8-sig'
    )
    
    return results


# ========== 主函数 ==========
def main():
    # 路径配置
    base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, 'rf_optimized_results')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据预处理（含特征选择）
    X_train, y_train, X_test, y_test, train_df, test_df, selected_features = load_and_preprocess_data(
        os.path.join(base_dir, 'train_dataset_optimized.csv'),
        os.path.join(base_dir, 'test_dataset_optimized.csv')
    )

    # 2. 模型训练
    print(f"\n🔍 使用最佳参数：{BEST_PARAMS}")
    model = train_model(X_train, y_train)

    # 3. 模型评估
    results = evaluate_model(model, X_test, y_test, save_dir, selected_features)

    # 4. 交叉验证
    scoring = {
        'roc_auc': 'roc_auc',
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(specificity_score)
    }
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=5, scoring=scoring,
        n_jobs=1
    )
    print("\n======= 交叉验证稳定性 =======")
    print(f"AUC均值±标准差：{cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"灵敏度均值±标准差：{cv_results['test_sensitivity'].mean():.4f} ± {cv_results['test_sensitivity'].std():.4f}")
    print(f"特异度均值±标准差：{cv_results['test_specificity'].mean():.4f} ± {cv_results['test_specificity'].std():.4f}")

    # 5. 可视化
    plot_roc_curve(results, save_dir)
    feature_importance = plot_feature_importance(model, selected_features, save_dir)
    plot_clinical_dca(y_test, results['y_pred_proba'], save_dir)

    # 6. 分层验证
    stratified_results = stratified_validation(
        model, X_test, y_test, test_df, results['auc'], save_dir
    )

    # 7. 保存结果
    joblib.dump(model, os.path.join(save_dir, 'rf_best_model.pkl'))
    result_summary = {
        '整体AUC': results['auc'],
        '交叉验证AUC': f"{cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}",
        '最佳阈值': results['thresholds'][0],
        '临床推荐阈值': CLINICAL_PARAMS,
        '分层验证结果': stratified_results
    }
    pd.DataFrame([result_summary]).to_csv(
        os.path.join(save_dir, 'result_summary.csv'), index=False, encoding='utf-8-sig'
    )
    print("\n✅ 所有结果已保存至：", save_dir)


if __name__ == "__main__":
    main()