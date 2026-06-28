import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import warnings
import joblib
import os

# 全局配置：解决中文编码与显示问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


# ========================
# 1. 数据加载与预处理（与其他模型保持一致）
# ========================
def load_and_preprocess_data(train_path, test_path, label_col='diabe'):
    # 处理中文路径与编码
    train_path = str(train_path)
    test_path = str(test_path)
    try:
        train_df = pd.read_csv(train_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
    except UnicodeDecodeError:
        train_df = pd.read_csv(train_path, encoding='gbk')
        test_df = pd.read_csv(test_path, encoding='gbk')
    
    print(f"✅ 成功读取优化数据集：")
    print(f"   - 训练集：{train_path}（{train_df.shape[0]}行 × {train_df.shape[1]}列）")
    print(f"   - 测试集：{test_path}（{test_df.shape[0]}行 × {test_df.shape[1]}列）")

    # 验证核心代谢指标
    core_metabolic_cols = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']
    missing_metabolic = [col for col in core_metabolic_cols if col not in train_df.columns]
    if missing_metabolic:
        print(f"⚠️  警告：核心代谢指标 {missing_metabolic} 缺失")
    else:
        print(f"✅ 核心代谢指标 {core_metabolic_cols} 均已保留")

    # 分离特征与标签
    X_train = train_df.drop(label_col, axis=1)
    y_train = train_df[label_col].copy()
    X_test = test_df.drop(label_col, axis=1)
    y_test = test_df[label_col].copy()

    # 标签处理（确保0-1编码）
    if y_train.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        print(f"✅ 标签编码完成：{le.classes_} → [0, 1]")
    else:
        y_train = y_train.astype(int).clip(0, 1)
        y_test = y_test.astype(int).clip(0, 1)
        print(f"✅ 标签验证完成：数值型（0=非糖尿病，1=糖尿病）")

    # 特征分类（Lasso对类别特征需One-Hot编码）
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [col for col in X_train.select_dtypes(include=['object', 'category']).columns 
                if col not in num_cols]
    
    print(f"\n✅ 特征分类完成：")
    print(f"   - 数值特征：{len(num_cols)}个（含代谢指标）")
    print(f"   - 分类特征：{len(cat_cols)}个（示例：{cat_cols[:3]}...）")

    # 缺失值处理
    from sklearn.impute import SimpleImputer
    # 数值特征用中位数填充（医疗数据抗极端值）
    num_imputer = SimpleImputer(strategy='median')
    X_train_num = pd.DataFrame(
        num_imputer.fit_transform(X_train[num_cols]),
        columns=num_cols,
        index=X_train.index
    )
    X_test_num = pd.DataFrame(
        num_imputer.transform(X_test[num_cols]),
        columns=num_cols,
        index=X_test.index
    )

    # 分类特征One-Hot编码（Lasso需处理类别特征）
    X_train_cat, X_test_cat = pd.DataFrame(), pd.DataFrame()
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train_cat_raw = pd.DataFrame(
            cat_imputer.fit_transform(X_train[cat_cols]),
            columns=cat_cols,
            index=X_train.index
        )
        X_test_cat_raw = pd.DataFrame(
            cat_imputer.transform(X_test[cat_cols]),
            columns=cat_cols,
            index=X_test.index
        )

        # One-Hot编码（删除首列避免多重共线性）
        X_train_cat = pd.get_dummies(X_train_cat_raw, drop_first=True)
        X_test_cat = pd.get_dummies(X_test_cat_raw, drop_first=True)

        # 确保测试集与训练集特征一致
        missing_cols = set(X_train_cat.columns) - set(X_test_cat.columns)
        for col in missing_cols:
            X_test_cat[col] = 0
        X_test_cat = X_test_cat[X_train_cat.columns]
        print(f"✅ 分类特征One-Hot编码完成：{len(cat_cols)}个 → {X_train_cat.shape[1]}个哑变量")

    # 合并特征矩阵
    X_train = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test = pd.concat([X_test_num, X_test_cat], axis=1)

    # 特征标准化（Lasso对特征尺度敏感，必须执行）
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print(f"✅ 特征标准化完成（Lasso专用）：均值=0，标准差=1")

    # 数据验证
    print(f"\n======= 数据验证信息 =======")
    print(f"训练集样本数: {len(X_train)} | 患病率: {y_train.mean():.2%}（{y_train.sum()}例患病）")
    print(f"测试集样本数: {len(X_test)} | 患病率: {y_test.mean():.2%}（{y_test.sum()}例患病）")
    print(f"最终特征数: {len(X_train.columns)} | 代谢指标特征: {[col for col in core_metabolic_cols if col in X_train.columns]}")
    print(f"是否存在缺失值: {X_train.isna().sum().sum() > 0}")
    print(f"分类特征哑变量数: {X_train_cat.shape[1]}")

    return X_train_scaled, y_train, X_test_scaled, y_test, core_metabolic_cols, scaler


# ========================
# 2. 加载数据（指定中文路径）
# ========================
train_path = r'train_dataset_optimized.csv'
test_path = r'test_dataset_optimized.csv'

X_train, y_train, X_test, y_test, core_metabolic_cols, scaler = load_and_preprocess_data(
    train_path=train_path,
    test_path=test_path,
    label_col='diabe'
)


# ========================
# 3. Lasso模型训练（L1正则化逻辑回归）
# ========================
def train_lasso_model(X_train, y_train, X_test, y_test):
    # 计算类别权重（处理不平衡数据）
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1, 1: n_neg / n_pos}  # 与其他模型保持一致
    print(f"\n======= 训练Lasso模型（L1正则化逻辑回归） =======")
    print(f"类别平衡权重：负样本={class_weight[0]}, 正样本={class_weight[1]:.2f}")

    # Lasso关键参数：C是正则化强度的倒数（C越小，正则化越强）
    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # 重点测试不同正则化强度
        'penalty': ['l1'],        # L1正则化（Lasso核心）
        'solver': ['liblinear'],  # 仅liblinear支持L1正则化
        'class_weight': [class_weight]
    }

    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 初始化Lasso模型（逻辑回归+L1正则化）
    base_lasso = LogisticRegression(
        max_iter=1000,  # 确保收敛
        random_state=42,
        verbose=0
    )

    # 网格搜索（单线程避免编码问题）
    print(f"📊 开始参数优化（L1正则化强度搜索）...")
    grid_search = GridSearchCV(
        estimator=base_lasso,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1,  # 单线程运行
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 最佳参数（关注C值，体现L1正则化强度）
    print(f"✅ 参数优化完成，最佳参数：{grid_search.best_params_}")
    print(f"   - 最佳交叉验证AUC：{grid_search.best_score_:.4f}")

    # 最佳模型预测
    best_lasso = grid_search.best_estimator_
    y_pred_proba = best_lasso.predict_proba(X_test)[:, 1]  # 正类概率
    y_pred = best_lasso.predict(X_test)

    # 测试集性能
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"   - 测试集AUC：{test_auc:.4f}")

    # Lasso特征选择结果（核心差异：系数为0的特征被剔除）
    coefs = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': best_lasso.coef_[0]
    })
    selected_features = coefs[coefs['Coefficient'] != 0].shape[0]
    removed_features = coefs[coefs['Coefficient'] == 0].shape[0]
    print(f"   - L1正则化特征选择：保留{selected_features}个特征，剔除{removed_features}个冗余特征")

    return best_lasso, y_pred_proba, y_pred, test_auc, coefs

# 训练Lasso模型
lasso_model, y_pred_proba, y_pred, lasso_auc, coefs = train_lasso_model(X_train, y_train, X_test, y_test)


# ========================
# 4. 模型评估（与其他模型对比）
# ========================
print("\n======= 模型评估结果（含代谢指标） =======")
# 核心指标计算
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # 灵敏度（不漏诊）
specificity = tn / (tn + fp)  # 特异度（不误诊）
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值

# 与其他模型对比（普通逻辑回归/LightGBM等）
print(f"准确率（Accuracy）：{accuracy:.4f}（普通逻辑回归：0.8215，Lasso：{accuracy:.4f}，LightGBM：0.8290）")
print(f"AUC值（核心指标）：{lasso_auc:.4f}（普通逻辑回归：0.8765，Lasso：{lasso_auc:.4f}，XGBoost：0.8946）")

print("\n📊 混淆矩阵（行=真实，列=预测）：")
print(f"非糖尿病正确：{tn:>4} | 误诊为糖尿病：{fp:>4}")
print(f"糖尿病漏诊：   {fn:>4} | 正确识别：   {tp:>4}")

print(f"\n🏥 医学关键指标：")
print(f"灵敏度（不漏诊）：{sensitivity:.4f}（普通逻辑回归：0.7425，Lasso：{sensitivity:.4f}）")
print(f"特异度（不误诊）：{specificity:.4f}（普通逻辑回归：0.8302，Lasso：{specificity:.4f}）")
print(f"阳性预测值（PPV）：{ppv:.4f} | 阴性预测值（NPV）：{npv:.4f}")

# 分类报告
print("\n📋 分类报告：")
print(classification_report(
    y_test, y_pred,
    target_names=['非糖尿病', '糖尿病'],
    digits=4
))


# ========================
# 5. 特征重要性分析（Lasso系数可视化）
# ========================
def analyze_lasso_importance(coefs, core_metabolic_cols, top_n=15):
    # 按系数绝对值排序（Lasso系数直接反映特征重要性）
    importance = coefs.copy()
    importance['Abs_Coefficient'] = importance['Coefficient'].abs()
    importance['Is_Metabolic'] = [col in core_metabolic_cols for col in importance['Feature']]
    importance = importance.sort_values('Abs_Coefficient', ascending=False)

    # Top N特征
    top_features = importance.head(top_n).copy()
    print(f"\n======= Top 15 特征重要性（Lasso系数绝对值） =======")
    top_features['Metabolic_Label'] = top_features['Is_Metabolic'].map({True: '✓ 代谢指标', False: ''})
    print(top_features[['Feature', 'Coefficient', 'Abs_Coefficient', 'Metabolic_Label']].round(4))

    # 可视化Lasso系数（正负表示影响方向）
    plt.figure(figsize=(12, 7))
    colors = ['#1E3A8A' if is_meta else '#64748B' for is_meta in top_features['Is_Metabolic']]
    sns.barplot(
        x='Abs_Coefficient', 
        y='Feature', 
        data=top_features,
        palette=colors,
        edgecolor='black',
        linewidth=0.5
    )
    # 标注系数正负（区分保护因素/风险因素）
    for i, (_, row) in enumerate(top_features.iterrows()):
        sign = '+' if row['Coefficient'] > 0 else '-'
        plt.text(
            row['Abs_Coefficient'] + max(top_features['Abs_Coefficient']) * 0.01,
            i, 
            f"{sign} {row['Coefficient']:.3f}", 
            va='center', fontsize=9, fontweight='bold'
        )
        if row['Is_Metabolic']:
            plt.text(
                -max(top_features['Abs_Coefficient']) * 0.1,
                i, 
                '✓ 代谢指标', 
                va='center', fontsize=9, color='#1E3A8A', fontweight='bold'
            )
    plt.xlabel('系数绝对值（Lasso重要性）', fontsize=12, fontweight='bold')
    plt.ylabel('特征名称', fontsize=12, fontweight='bold')
    plt.title('Lasso模型 - 核心特征重要性（含代谢指标方向）', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('lasso_feature_importance_with_direction.png', bbox_inches='tight')
    plt.close()
    print("\n✅ 特征重要性图已保存：lasso_feature_importance_with_direction.png")

    # 代谢指标系数对比（与普通逻辑回归的核心差异）
    metabolic_coefs = importance[importance['Is_Metabolic']].sort_values('Abs_Coefficient', ascending=False)
    print("\n======= 代谢指标Lasso系数（与普通逻辑回归对比） =======")
    print(f"{'特征':<10} {'Lasso系数':<12} {'普通逻辑回归系数':<18} {'是否被Lasso保留':<15}")
    print("-" * 70)
    # 普通逻辑回归参考系数（示例值，需替换为实际结果）
    lr_ref_coefs = {
        'bl_hbalc': 0.523, 'bl_glu': 0.412, 'tyg_bmi': 0.187,
        'bl_tg': 0.093, 'tyg': 0.076
    }
    for _, row in metabolic_coefs.iterrows():
        feat = row['Feature']
        lasso_coef = row['Coefficient']
        lr_coef = lr_ref_coefs.get(feat, 0.0)
        is_kept = '保留' if lasso_coef != 0 else '剔除'
        print(f"{feat.ljust(10)} {str(round(lasso_coef,4)).ljust(12)} {str(round(lr_coef,4)).ljust(18)} {is_kept.ljust(15)}")
    
    return importance

feature_importance = analyze_lasso_importance(coefs, core_metabolic_cols)


# ========================
# 6. 绘制多模型ROC对比图（加入Lasso）
# ========================
def plot_roc_with_lasso(y_test, y_pred_proba_lasso):
    # 加载其他模型预测概率
    try:
        xgb_proba = pd.read_csv(r'xgb_pred_proba.csv')['xgb_pred_proba']
        lgb_proba = pd.read_csv(r'lgb_pred_proba.csv')['lgb_pred_proba']
        rf_proba = pd.read_csv(r'rf_pred_proba.csv')['rf_pred_proba']
        lr_proba = pd.read_csv(r'lr_pred_proba.csv')['lr_pred_proba']
        svm_proba = pd.read_csv(r'svm_pred_proba.csv')['svm_pred_proba']
        have_other_models = True
    except:
        print("\n⚠️  警告：未找到其他模型预测概率文件，仅绘制Lasso的ROC曲线")
        have_other_models = False

    plt.figure(figsize=(12, 8))
    # Lasso ROC（绿色实线，突出显示）
    fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_pred_proba_lasso)
    plt.plot(
        fpr_lasso, tpr_lasso, 
        color='#10B981', linewidth=3, 
        label=f'Lasso (AUC = {lasso_auc:.4f})',
        zorder=6
    )
    # 其他模型ROC
    if have_other_models:
        # 普通逻辑回归
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
        plt.plot(
            fpr_lr, tpr_lr, 
            color='#2E86AB', linewidth=2.5, linestyle='-.', label=f'普通逻辑回归 (AUC ≈ 0.8765)', zorder=2
        )
        # XGBoost
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
        plt.plot(
            fpr_xgb, tpr_xgb, 
            color='#DC2626', linewidth=2.5, label=f'XGBoost (AUC ≈ 0.8946)', zorder=5
        )
        # LightGBM
        fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_proba)
        plt.plot(
            fpr_lgb, tpr_lgb, 
            color='#065F46', linewidth=2.5, label=f'LightGBM (AUC ≈ 0.8906)', zorder=4
        )
        # SVM
        fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_proba)
        plt.plot(
            fpr_svm, tpr_svm, 
            color='#800080', linewidth=2.5, label=f'SVM (AUC ≈ {svm_proba.mean():.4f})', zorder=3
        )
    # 随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='随机猜测', zorder=1)

    # 标注Lasso最佳临界点
    youden_idx = np.argmax(tpr_lasso - fpr_lasso)
    plt.scatter(fpr_lasso[youden_idx], tpr_lasso[youden_idx], color='#FF6347', s=120, zorder=7)
    plt.annotate(
        f'Lasso最佳临界点\n(灵敏度={tpr_lasso[youden_idx]:.3f},\n特异度={1-fpr_lasso[youden_idx]:.3f})',
        xy=(fpr_lasso[youden_idx], tpr_lasso[youden_idx]),
        xytext=(fpr_lasso[youden_idx]+0.1, tpr_lasso[youden_idx]-0.2),
        arrowprops=dict(arrowstyle='->', color='#FF6347', lw=2),
        fontsize=10, fontweight='bold'
    )

    plt.xlabel('假阳性率（1-特异度）', fontsize=12, fontweight='bold')
    plt.ylabel('真阳性率（灵敏度）', fontsize=12, fontweight='bold')
    plt.title('糖尿病预测模型 ROC曲线对比（含Lasso）', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('roc_with_lasso_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"\n✅ ROC对比图已保存：roc_with_lasso_comparison.png")

plot_roc_with_lasso(y_test, y_pred_proba)


# ========================
# 7. 保存模型与结果
# ========================
def save_lasso_results(model, scaler, y_pred_proba, y_test, feature_importance, coefs):
    save_dir = r''
    
    # 保存Lasso模型（含标准化器）
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist(),
        'coefficients': coefs  # 保存系数用于特征分析
    }, os.path.join(save_dir, 'lasso_diabetes_model.pkl'))

    # 保存预测概率（用于集成学习）
    lasso_proba_df = pd.DataFrame({
        'lasso_pred_proba': y_pred_proba,
        'true_label': y_test
    })
    lasso_proba_df.to_csv(os.path.join(save_dir, 'lasso_pred_proba.csv'), index=False, encoding='utf-8-sig')

    # 保存特征重要性（含系数）
    feature_importance.to_csv(
        os.path.join(save_dir, 'lasso_feature_importance_full.csv'), 
        index=False, 
        encoding='utf-8-sig'
    )

    # 保存Lasso特征选择结果（论文重点）
    selected_features = coefs[coefs['Coefficient'] != 0]['Feature'].tolist()
    with open(os.path.join(save_dir, 'lasso_selected_features.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Lasso模型特征选择结果（总特征数：{len(coefs)}）\n")
        f.write(f"保留特征数：{len(selected_features)} | 剔除特征数：{len(coefs) - len(selected_features)}\n\n")
        f.write("保留的核心代谢指标：\n")
        for feat in core_metabolic_cols:
            if feat in selected_features:
                f.write(f"- {feat}（系数：{coefs[coefs['Feature']==feat]['Coefficient'].values[0]:.4f}）\n")
        f.write("\n保留的其他重要特征：\n")
        for feat in selected_features[:10]:
            if feat not in core_metabolic_cols:
                f.write(f"- {feat}（系数：{coefs[coefs['Feature']==feat]['Coefficient'].values[0]:.4f}）\n")

    print("\n✅ 模型与结果保存完成：")
    print(f"   - Lasso模型：lasso_diabetes_model.pkl")
    print(f"   - 预测概率：lasso_pred_proba.csv（集成学习用）")
    print(f"   - 特征选择结果：lasso_selected_features.txt（论文重点）")

save_lasso_results(lasso_model, scaler, y_pred_proba, y_test, feature_importance, coefs)


# ========================
# 8. Lasso与普通逻辑回归核心差异总结
# ========================
print("\n" + "="*90)
print("🔍 Lasso vs 普通逻辑回归 核心差异")
print("="*90)
print(f"{'对比维度':<20} {'Lasso回归（L1正则化）':<40} {'普通逻辑回归':<30}")
print("-" * 90)
print(f"正则化机制      {'L1正则化，系数可压缩至0':<40} {'无正则化或L2正则化':<30}")
print(f"特征选择能力    {'自动剔除冗余特征（系数=0）':<40} {'保留所有特征，系数非0':<30}")
print(f"高维数据表现    {'抗过拟合，适合代谢指标等多特征场景':<40} {'易过拟合，特征冗余时性能下降':<30}")
print(f"可解释性        {'系数直接反映特征重要性及方向':<40} {'系数受多重共线性影响':<30}")
print(f"本次任务AUC     {f'{lasso_auc:.4f}（可能略低但更稳健）':<40} {'0.8765（可能过拟合）':<30}")
print(f"核心优势        {'筛选出关键代谢指标（如bl_hbalc）':<40} {'训练速度快，基线模型参考':<30}")
print("="*90)
print("💡 论文应用建议：用Lasso的特征选择结果解释核心代谢指标的预测价值，支持临床决策")
