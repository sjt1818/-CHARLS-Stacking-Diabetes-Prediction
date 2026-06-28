import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, make_scorer, recall_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

# ========== 环境与常量配置 ==========
def setup_environment():
    os.environ["PYTHONUTF8"] = "1"
    warnings.filterwarnings('ignore')
    plt.rcParams.update({
        'font.sans-serif': ['SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'figure.dpi': 150,
        'savefig.dpi': 300
    })

CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']
CLINICAL_PARAMS = {
    '漏诊权重': 2.0,
    '初筛阈值': 0.3,
    '确诊阈值': 0.6
}


# ========== 自定义评估函数 ==========
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def clinical_net_benefit(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp - fp * (1/CLINICAL_PARAMS['漏诊权重'])) / len(y_true)


# ========== 数据处理类 ==========
class DataProcessor:
    def __init__(self, label_col='diabe'):
        self.label_col = label_col
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.label_encoder = None

    def load_and_preprocess(self, train_path, test_path):
        train_df = self._safe_read(train_path)
        test_df = self._safe_read(test_path)
        print(f"✅ 数据集读取成功：")
        print(f"   - 训练集：{os.path.basename(train_path)}（{train_df.shape}）")
        print(f"   - 测试集：{os.path.basename(test_path)}（{test_df.shape}）")

        missing_metabolic = [col for col in CORE_METABOLIC if col not in train_df.columns]
        print(f"✅ 核心代谢指标：{'完整' if not missing_metabolic else f'缺失{missing_metabolic}'}")

        X_train = train_df.drop(self.label_col, axis=1)
        y_train = train_df[self.label_col].copy()
        X_test = test_df.drop(self.label_col, axis=1)
        y_test = test_df[self.label_col].copy()

        y_train, y_test = self._process_labels(y_train, y_test)

        num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
        print(f"\n✅ 特征分类：数值特征{len(num_cols)}个，分类特征{len(cat_cols)}个")

        X_train[num_cols] = self.num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = self.num_imputer.transform(X_test[num_cols])

        if cat_cols:
            X_train[cat_cols] = self.cat_imputer.fit_transform(X_train[cat_cols])
            X_test[cat_cols] = self.cat_imputer.transform(X_test[cat_cols])
            X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
            X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)

        print(f"\n======= 数据验证 =======")
        print(f"训练集：{len(X_train_scaled)}样本 | 患病率{y_train.mean():.2%}")
        print(f"测试集：{len(X_test_scaled)}样本 | 患病率{y_test.mean():.2%}")
        print(f"特征数：{X_train_scaled.shape[1]} | 无缺失值")

        return X_train_scaled, y_train, X_test_scaled, y_test, train_df, test_df, X_train.columns.tolist()

    def _safe_read(self, file_path):
        for encoding in ['utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解析文件编码: {file_path}")

    def _process_labels(self, y_train, y_test):
        if y_train.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)
            print(f"✅ 标签编码：{self.label_encoder.classes_} → [0, 1]")
        else:
            y_train = y_train.astype(int).clip(0, 1)
            y_test = y_test.astype(int).clip(0, 1)
            print(f"✅ 标签验证：数值型（0=非糖尿病，1=糖尿病）")
        return y_train, y_test


# ========== 模型训练类（嵌套CV+网格搜索） ==========
class SVMModel:
    def __init__(self, random_state=42):
        self.model = None
        self.best_params = None
        self.random_state = random_state
        self.outer_cv_results = []

    def train_with_nested_cv(self, X_train, y_train, n_outer=5, n_inner=3):
        print("\n======= 嵌套交叉验证训练SVM =======")
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        class_weight = {0: 1, 1: n_neg / n_pos}
        print(f"类别权重：负样本=1，正样本={class_weight[1]:.2f}")

        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear'],
            'class_weight': [class_weight]
        }

        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=self.random_state)
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=self.random_state)

        base_model = SVC(probability=True, max_iter=5000, random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=1,
            verbose=1
        )

        for train_idx, val_idx in outer_cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            grid_search.fit(X_tr, y_tr)
            y_pred_proba = grid_search.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            self.outer_cv_results.append(auc)
            print(f"外层折叠 AUC: {auc:.4f} (最佳参数: {grid_search.best_params_['C']}, {grid_search.best_params_['kernel']})")

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.model.fit(X_train, y_train)

        print(f"\n嵌套CV结果：AUC均值±标准差 {np.mean(self.outer_cv_results):.4f} ± {np.std(self.outer_cv_results):.4f}")
        print(f"最佳参数：{self.best_params}")
        return self


# ========== 模型评估类 ==========
class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.thresholds = {}
        self.y_pred_proba = None

    def evaluate(self, model, X_test, y_test):
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, self.y_pred_proba)

        self.thresholds = {
            '约登最佳': thresholds[np.argmax(tpr - fpr)],
            '基层初筛': CLINICAL_PARAMS['初筛阈值'],
            '确诊前筛查': CLINICAL_PARAMS['确诊阈值']
        }

        self.metrics = {'整体': {'auc': roc_auc_score(y_test, self.y_pred_proba)}}
        for name, threshold in self.thresholds.items():
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            self.metrics[name] = {
                '灵敏度': tp / (tp + fn) if (tp + fn) > 0 else 0,
                '特异度': tn / (tn + fp) if (tn + fp) > 0 else 0,
                '阳性预测值': tp / (tp + fp) if (tp + fp) > 0 else 0,
                '阴性预测值': tn / (tn + fn) if (tn + fn) > 0 else 0,
                '临床净获益': clinical_net_benefit(y_test, self.y_pred_proba, threshold)
            }

        self._print_evaluation(y_test)
        return {
            'metrics': self.metrics,
            'thresholds': self.thresholds,
            'y_pred_proba': self.y_pred_proba,
            'fpr': fpr,
            'tpr': tpr
        }

    def _print_evaluation(self, y_test):
        print("\n======= 模型评估结果 =======")
        print(f"AUC：{self.metrics['整体']['auc']:.4f} | 最佳阈值：{self.thresholds['约登最佳']:.3f}")

        for name, threshold in self.thresholds.items():
            print(f"\n【{name}（阈值={threshold:.3f}）】")
            print(f"   灵敏度：{self.metrics[name]['灵敏度']:.4f} | 特异度：{self.metrics[name]['特异度']:.4f}")
            print(f"   阳性预测值：{self.metrics[name]['阳性预测值']:.4f} | 阴性预测值：{self.metrics[name]['阴性预测值']:.4f}")
            print(f"   临床净获益：{self.metrics[name]['临床净获益']:.4f}")

        best_y_pred = (self.y_pred_proba >= self.thresholds['约登最佳']).astype(int)
        y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
        y_test = y_test.astype(int)

        cm = confusion_matrix(y_test, best_y_pred)
        tn, fp, fn, tp = cm.ravel()
        print("\n📊 最佳阈值混淆矩阵：")
        print(f"非糖尿病正确：{tn} | 误诊：{fp}")
        print(f"糖尿病漏诊：{fn} | 正确识别：{tp}")

        print("\n📋 分类报告：")
        print(classification_report(
            y_test, best_y_pred, 
            target_names=['非糖尿病', '糖尿病'], 
            digits=4
        ))


# ========== 可视化类 ==========
class Visualizer:
    @staticmethod
    def plot_roc_curve(results, save_dir):
        plt.figure(figsize=(8, 6))
        plt.plot(
            results['fpr'], results['tpr'], 
            color='#800080', linewidth=3, 
            label=f'SVM (AUC={results["metrics"]["整体"]["auc"]:.4f})'
        )
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='随机猜测')

        thresholds = results['thresholds']
        colors = ['#A23B72', '#457B9D', '#1D3557']
        for (name, threshold), color in zip(thresholds.items(), colors):
            idx = np.argmin(np.abs(results['fpr'] + results['tpr'] - 1 - (1 - 2*threshold)))
            plt.scatter(
                results['fpr'][idx], results['tpr'][idx], 
                color=color, s=100, zorder=5, label=f'{name} ({threshold:.3f})'
            )

        plt.xlabel('假阳性率（1-特异度）')
        plt.ylabel('真阳性率（灵敏度）')
        plt.title('SVM ROC曲线（多场景阈值）')
        plt.legend()
        plt.grid(alpha=0.3)
        save_path = os.path.join(save_dir, 'svm_roc_curve.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC曲线已保存：{save_path}")

    @staticmethod
    def plot_feature_importance(model, X_train, y_train, feature_names, save_dir):
        result = permutation_importance(
            model, X_train, y_train,
            n_repeats=10,
            random_state=42,
            n_jobs=1
        )

        importance = pd.DataFrame({
            '特征': feature_names,
            '重要性': result.importances_mean,
            '是否代谢指标': [col in CORE_METABOLIC for col in feature_names]
        }).sort_values('重要性', ascending=False)

        top15 = importance.head(15)
        plt.figure(figsize=(10, 6))
        colors = ['#DC2626' if meta else '#64748B' for meta in top15['是否代谢指标']]
        sns.barplot(x='重要性', y='特征', data=top15, palette=colors)

        for i, (_, row) in enumerate(top15.iterrows()):
            if row['是否代谢指标']:
                plt.text(row['重要性']+0.001, i, '✓ 代谢指标', va='center', color='#DC2626')

        plt.xlabel('特征重要性（排列重要性）')
        plt.ylabel('特征名称')
        plt.title('SVM核心特征重要性（Top15）')
        plt.grid(axis='x', alpha=0.3)
        save_path = os.path.join(save_dir, 'svm_feature_importance.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 特征重要性图已保存：{save_path}")
        return importance

    @staticmethod
    def plot_clinical_dca(y_true, y_proba, save_dir):
        thresholds = np.linspace(0, 0.99, 100)
        nb_model = [clinical_net_benefit(y_true, y_proba, t) for t in thresholds]
        nb_all = [(sum(y_true) - len(y_true)*t/(CLINICAL_PARAMS['漏诊权重']))/len(y_true) for t in thresholds]
        nb_none = [0 for _ in thresholds]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, nb_model, label='SVM模型', color='#800080', linewidth=3)
        plt.plot(thresholds, nb_all, label='全部筛查', color='gray', linestyle='--')
        plt.plot(thresholds, nb_none, label='不筛查', color='black', linestyle=':')

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
        plt.title('SVM临床校准决策曲线分析（DCA）')
        plt.legend()
        plt.grid(alpha=0.3)
        save_path = os.path.join(save_dir, 'svm_clinical_dca.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 临床决策曲线已保存：{save_path}")

    @staticmethod
    def plot_stratified_auc(model, X_test, y_test, original_test, overall_auc, save_dir):
        y_proba = model.predict_proba(X_test)[:, 1]
        original_test = original_test.reset_index(drop=True)

        age_groups = pd.cut(original_test['age'], bins=[0, 60, 120], labels=['<60岁', '≥60岁'])
        age_auc = {g: roc_auc_score(y_test[age_groups==g], y_proba[age_groups==g]) 
                   for g in age_groups.unique() if (age_groups==g).sum() > 0}

        bmi_groups = pd.cut(original_test['bmi'], bins=[0, 18.5, 24, 28, 100], labels=['偏瘦', '正常', '超重', '肥胖'])
        bmi_auc = {g: roc_auc_score(y_test[bmi_groups==g], y_proba[bmi_groups==g]) 
                   for g in bmi_groups.unique() if not pd.isna(g) and (bmi_groups==g).sum() > 0}

        print("\n======= 分层验证结果 =======")
        print("【年龄分层AUC】")
        for g, auc in age_auc.items():
            print(f"   {g}：{auc:.4f}")
        print("\n【BMI分层AUC】")
        for g, auc in bmi_auc.items():
            print(f"   {g}：{auc:.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        age_data = pd.DataFrame(age_auc.items(), columns=['年龄组', 'AUC'])
        sns.barplot(x='年龄组', y='AUC', data=age_data, ax=ax1, palette=['#A8DADC'])
        ax1.axhline(y=overall_auc, color='r', linestyle='--', label=f'整体AUC: {overall_auc:.4f}')
        ax1.set_title('年龄分层AUC对比')
        ax1.set_ylim(0.5, 1.0)
        ax1.legend()

        bmi_data = pd.DataFrame(bmi_auc.items(), columns=['BMI组', 'AUC'])
        sns.barplot(x='BMI组', y='AUC', data=bmi_data, ax=ax2, palette=['#F1FAEE'])
        ax2.axhline(y=overall_auc, color='r', linestyle='--', label=f'整体AUC: {overall_auc:.4f}')
        ax2.set_title('BMI分层AUC对比')
        ax2.set_ylim(0.5, 1.0)
        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'svm_stratified_auc.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 分层AUC对比图已保存：{save_path}")
        return {'age': age_auc, 'bmi': bmi_auc}


# ========== 结果保存类 ==========
class ResultSaver:
    @staticmethod
    def save_results(model, processor, results, importance, stratified_results, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        joblib.dump({
            'model': model.model,
            'scaler': processor.scaler,
            'best_params': model.best_params
        }, os.path.join(save_dir, 'svm_best_model.pkl'))

        pd.DataFrame({
            'svm_pred_proba': results['y_pred_proba'],
            'true_label': y_test
        }).to_csv(os.path.join(save_dir, 'svm_pred_proba.csv'), index=False, encoding='utf-8-sig')

        importance.to_csv(
            os.path.join(save_dir, 'svm_feature_importance.csv'),
            index=False, encoding='utf-8-sig'
        )

        metrics_df = pd.DataFrame()
        for name, metric in results['metrics'].items():
            if name == '整体':
                continue
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                '场景': [name],
                '阈值': [results['thresholds'][name]],
                'AUC': [results['metrics']['整体']['auc']],
                '灵敏度': [metric['灵敏度']],
                '特异度': [metric['特异度']],
                '阳性预测值': [metric['阳性预测值']],
                '阴性预测值': [metric['阴性预测值']],
                '临床净获益': [metric['临床净获益']]
            })], ignore_index=True)

        metrics_df.to_csv(
            os.path.join(save_dir, 'svm_metrics.csv'),
            index=False, encoding='utf-8-sig'
        )

        pd.DataFrame({
            '嵌套折叠': range(1, len(model.outer_cv_results)+1),
            'AUC': model.outer_cv_results
        }).to_csv(
            os.path.join(save_dir, 'svm_nested_cv_results.csv'),
            index=False, encoding='utf-8-sig'
        )

        print("\n✅ 所有结果已保存至：", save_dir)


# ========== 主函数 ==========
def main():
    setup_environment()
    base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, 'svm_optimized_results')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 数据预处理
    processor = DataProcessor(label_col='diabe')
    X_train, y_train, X_test, y_test, train_df, test_df, feature_names = processor.load_and_preprocess(
        os.path.join(base_dir, 'train_dataset_optimized.csv'),
        os.path.join(base_dir, 'test_dataset_optimized.csv')
    )

    # 2. 模型训练
    svm_model = SVMModel(random_state=42)
    svm_model.train_with_nested_cv(X_train, y_train, n_outer=5, n_inner=3)

    # 3. 模型评估
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(svm_model.model, X_test, y_test)

    # 4. 交叉验证稳定性
    scoring = {
        'roc_auc': 'roc_auc',
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(specificity_score)
    }
    cv_results = cross_validate(
        svm_model.model, X_train, y_train,
        cv=5, scoring=scoring,
        n_jobs=1
    )
    print("\n======= 交叉验证稳定性 =======")
    print(f"AUC均值±标准差：{cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"灵敏度均值±标准差：{cv_results['test_sensitivity'].mean():.4f} ± {cv_results['test_sensitivity'].std():.4f}")
    print(f"特异度均值±标准差：{cv_results['test_specificity'].mean():.4f} ± {cv_results['test_specificity'].std():.4f}")

    # 5. 可视化
    visualizer = Visualizer()
    visualizer.plot_roc_curve(results, save_dir)
    importance = visualizer.plot_feature_importance(svm_model.model, X_train, y_train, feature_names, save_dir)
    visualizer.plot_clinical_dca(y_test, results['y_pred_proba'], save_dir)

    # 6. 分层验证
    stratified_results = visualizer.plot_stratified_auc(
        svm_model.model, X_test, y_test, test_df, results['metrics']['整体']['auc'], save_dir
    )

    # 7. 保存结果
    ResultSaver.save_results(
        svm_model, processor, results, importance, stratified_results, save_dir
    )


if __name__ == "__main__":
    main()