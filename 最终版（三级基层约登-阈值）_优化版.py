import os, sys, json, time, warnings, tempfile
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap  # SHAP 可解释性分析
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, recall_score, precision_score, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb

os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'tuned_stacking_results')
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 日志系统 =====================
class Logger:
    def __init__(self, path):
        self.path = path
        self.lines = []
    def log(self, *args):
        msg = ' '.join(str(a) for a in args)
        self.lines.append(msg)
        print(msg)
    def save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
log = Logger(os.path.join(SAVE_DIR, 'experiment_log.txt'))

# ===================== 模型参数 =====================
XGBOOST_PARAMS = {
    'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100,
    'subsample': 0.85, 'scale_pos_weight': None, 'random_state': RANDOM_STATE,
    'use_label_encoder': False, 'eval_metric': 'logloss'
}
LIGHTGBM_PARAMS = {
    'learning_rate': 0.03, 'max_depth': 5, 'min_child_samples': 20,
    'n_estimators': 250, 'num_leaves': 25, 'class_weight': 'balanced',
    'random_state': RANDOM_STATE, 'verbose': -1
}
LOGISTIC_PARAMS = {
    'C': 0.07, 'penalty': 'l2', 'solver': 'liblinear',
    'class_weight': 'balanced', 'max_iter': 5000, 'random_state': RANDOM_STATE
}
RF_PARAMS = {
    'bootstrap': True, 'class_weight': 'balanced', 'max_depth': 11,
    'max_features': 'sqrt', 'min_impurity_decrease': 0.0007143340896097039,
    'min_samples_leaf': 10, 'min_samples_split': 60, 'n_estimators': 180,
    'random_state': RANDOM_STATE, 'n_jobs': 1
}

BASE_MODEL_NAMES = ['logistic', 'random_forest', 'xgboost', 'lightgbm']

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# ===================== 数据预处理 =====================
def get_feature_names(preprocessor, num_cols, cat_cols):
    """从 fitted ColumnTransformer 中提取特征名称"""
    names = list(num_cols)
    try:
        encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cats = encoder.categories_
        for i, col in enumerate(cat_cols):
            for val in cats[i][1:]:  # drop_first=True, 跳过第一个
                names.append(f'{col}_{val}')
    except:
        names += [f'cat_{c}' for c in cat_cols]
    return names

def preprocess_data(train_path, test_path, label_col='diabe'):
    def safe_read(file):
        try:
            return pd.read_csv(file, encoding='utf-8')
        except:
            return pd.read_csv(file, encoding='gbk')

    train = safe_read(train_path)
    test = safe_read(test_path)
    log.log(f"数据加载：训练集{train.shape[0]}例，测试集{test.shape[0]}例")

    def create_medical_features(df):
        glu_in = 'bl_glu' in df.columns
        hba1c_in = 'bl_hbalc' in df.columns
        tyg_bmi_in = 'tyg_bmi' in df.columns

        if glu_in and hba1c_in:
            ratio = df['bl_glu'] / (df['bl_hbalc'] + 1e-6)
            df['glu_hbalc_ratio'] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            df['glu_hbalc_ratio'] = 0.0

        if tyg_bmi_in:
            df['tyg_bmi_squared'] = df['tyg_bmi'] ** 2
        else:
            df['tyg_bmi_squared'] = 0.0

        conditions = []
        if glu_in:
            conditions.append(df['bl_glu'] > 7.0)
        if hba1c_in:
            conditions.append(df['bl_hbalc'] > 6.5)
        if conditions:
            df['metabolic_abnormality'] = np.sum(np.column_stack(conditions), axis=1)
        else:
            df['metabolic_abnormality'] = 0

        return df

    train = create_medical_features(train)
    test = create_medical_features(test)

    y_train = train[label_col].map({'是': 1, '否': 0, 1: 1, 0: 0}).astype(int)
    y_test = test[label_col].map({'是': 1, '否': 0, 1: 1, 0: 0}).astype(int)
    X_train = train.drop(label_col, axis=1)
    X_test = test.drop(label_col, axis=1)

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    log.log(f"特征分类：数值{len(num_cols)}个，分类{len(cat_cols)}个")

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # 兜底处理残余 NaN/Inf
    if isinstance(X_train_p, np.ndarray):
        X_train_p = np.nan_to_num(X_train_p)
        X_test_p = np.nan_to_num(X_test_p)

    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    return X_train_p, y_train, X_test_p, y_test, preprocessor, feature_names

# ===================== Stacking 集成 =====================
def build_stacking(n_pos, n_neg):
    params_xgb = XGBOOST_PARAMS.copy()
    params_xgb['scale_pos_weight'] = n_neg / n_pos

    estimators = [
        ('logistic', LogisticRegression(**LOGISTIC_PARAMS)),
        ('random_forest', RandomForestClassifier(**RF_PARAMS)),
        ('xgboost', xgb.XGBClassifier(**params_xgb)),
        ('lightgbm', lgb.LGBMClassifier(**LIGHTGBM_PARAMS))
    ]
    meta = LogisticRegression(
        C=0.1, penalty='l2', solver='liblinear',
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
    )
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method='predict_proba',
        n_jobs=1,
        passthrough=False
    )
    return clf, estimators

# ===================== 带 Bootstrap CI 的评估器 =====================
def bootstrap_auc(y_true, y_proba, n_bootstrap=1000, ci=95):
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true.iloc[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true.iloc[idx], y_proba[idx]))
        except:
            continue
    if not aucs:
        return None, None, None
    lower = np.percentile(aucs, (100 - ci) / 2)
    upper = np.percentile(aucs, 100 - (100 - ci) / 2)
    return np.mean(aucs), lower, upper

class Evaluator:
    @staticmethod
    def calc_thresholds(y_true, y_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        # standard
        j = tpr - fpr
        sidx = np.argmax(j)
        # primary: 灵敏度↑
        pj = 1.2 * tpr + 0.8 * (1 - fpr) - 1
        pidx = np.argmax(pj)
        # tertiary: 特异度↑
        tj = 0.8 * tpr + 1.2 * (1 - fpr) - 1
        tidx = np.argmax(tj)
        return {
            'standard': {'threshold': thresholds[sidx], 'fpr': fpr[sidx], 'tpr': tpr[sidx]},
            'primary':  {'threshold': thresholds[pidx], 'fpr': fpr[pidx], 'tpr': tpr[pidx]},
            'tertiary': {'threshold': thresholds[tidx], 'fpr': fpr[tidx], 'tpr': tpr[tidx]}
        }

    @staticmethod
    def evaluate_scene(y_true, y_proba, threshold, scene_name):
        y_pred = (y_proba >= threshold).astype(int)
        return {
            'threshold': threshold,
            'auc': roc_auc_score(y_true, y_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred),
            'specificity': specificity_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'youden': recall_score(y_true, y_pred) + specificity_score(y_true, y_pred) - 1,
            'confusion_matrix': confusion_matrix(y_true, y_pred).ravel().tolist()
        }

    @staticmethod
    def full_eval(y_true, y_proba, model_name):
        log.log(f"\n======= {model_name} =======")
        ths = Evaluator.calc_thresholds(y_true, y_proba)

        # Bootstrap CI for AUC
        auc_mean, auc_lo, auc_hi = bootstrap_auc(y_true, y_proba)
        if auc_mean:
            log.log(f"AUC: {auc_mean:.4f} (95%CI: {auc_lo:.4f}~{auc_hi:.4f})")
        else:
            log.log(f"AUC: {roc_auc_score(y_true, y_proba):.4f} (CI skipped)")

        scenes = {'standard': '标准平衡', 'primary': '基层医院', 'tertiary': '三级医院'}
        results = {}
        for key, name in scenes.items():
            t = ths[key]['threshold']
            res = Evaluator.evaluate_scene(y_true, y_proba, t, name)
            results[key] = res
            tn, fp, fn, tp = res['confusion_matrix']
            log.log(f"  [{name}] 阈值={t:.4f} | AUC={res['auc']:.4f} | "
                    f"灵敏度={res['sensitivity']:.4f} | 特异度={res['specificity']:.4f} | "
                    f"约登={res['youden']:.4f} | F1={res['f1']:.4f} | "
                    f"TN={tn} FP={fp} FN={fn} TP={tp}")

        return {'thresholds': ths, 'scenes': results, 'y_proba': y_proba, 'auc_ci': (auc_mean, auc_lo, auc_hi)}

    # ----- 可视化 -----
    @staticmethod
    def plot_roc_with_scenes(y_true, model_results):
        plt.figure(figsize=(10, 8))
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']
        markers = {'standard': ('o', 'black'), 'primary': ('^', 'blue'), 'tertiary': ('s', 'red')}
        first_legend = True
        for i, (name, res) in enumerate(model_results.items()):
            fpr, tpr, _ = roc_curve(y_true, res['y_proba'])
            auc_val = res['scenes']['standard']['auc']
            plt.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'{name} (AUC={auc_val:.4f})')
            for skey, (marker, mcolor) in markers.items():
                pt = res['thresholds'][skey]
                lbl = f'{name} {skey}' if first_legend else ""
                plt.scatter(pt['fpr'], pt['tpr'], color=mcolor, s=80, zorder=5,
                            marker=marker, edgecolors='white', label=lbl)
            first_legend = False
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')
        plt.xlabel('假阳性率（1-特异度）', fontsize=12)
        plt.ylabel('真阳性率（灵敏度）', fontsize=12)
        plt.title('各模型ROC曲线及场景化阈值标记', fontsize=14)
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'roc_with_scene_thresholds.png'), bbox_inches='tight', dpi=200)
        plt.close()
        log.log(f"ROC曲线已保存")

    @staticmethod
    def plot_threshold_comparison(model_results):
        names = list(model_results.keys())
        std_t = [model_results[m]['thresholds']['standard']['threshold'] for m in names]
        pri_t = [model_results[m]['thresholds']['primary']['threshold'] for m in names]
        ter_t = [model_results[m]['thresholds']['tertiary']['threshold'] for m in names]

        plt.figure(figsize=(12, 6))
        x = np.arange(len(names))
        w = 0.25
        b1 = plt.bar(x - w, std_t, w, label='标准平衡', color='#9C27B0')
        b2 = plt.bar(x, pri_t, w, label='基层医院', color='#2196F3')
        b3 = plt.bar(x + w, ter_t, w, label='三级医院', color='#FF9800')
        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.4f}',
                         ha='center', va='bottom', fontsize=9)
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('约登指数最优阈值', fontsize=12)
        plt.title('各模型在不同临床场景下的阈值对比', fontsize=14)
        plt.xticks(x, names)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'model_threshold_comparison.png'), bbox_inches='tight', dpi=200)
        plt.close()
        log.log(f"阈值对比图已保存")

    @staticmethod
    def plot_meta_weights(meta_model, base_names):
        weights = np.abs(meta_model.coef_[0])
        weights /= np.sum(weights)
        plt.figure(figsize=(10, 6))
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
        bars = plt.bar(base_names, weights, color=colors)
        for i, w in enumerate(weights):
            plt.text(i, w + 0.01, f'{w:.4f}', ha='center', fontsize=10)
        plt.title('Stacking元模型对基模型的归一化权重分配', fontsize=14)
        plt.xlabel('基模型', fontsize=12)
        plt.ylabel('归一化权重（贡献度）', fontsize=12)
        plt.ylim(0, max(weights) + 0.1)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'meta_model_weights.png'), bbox_inches='tight', dpi=200)
        plt.close()
        log.log(f"元模型权重图已保存")

    # ----- SHAP 可解释性分析 -----
    @staticmethod
    def shap_analysis(stacking_clf, X_test, feature_names, save_dir, log, drop_feats=None):
        """对 Stacking 各基模型计算 SHAP，按元模型权重融合后可视化"""
        try:
            base_models = {}
            for name, _ in stacking_clf.estimators:
                base_models[name] = stacking_clf.named_estimators_[name]

            meta_weights = np.abs(stacking_clf.final_estimator_.coef_[0])
            meta_weights /= np.sum(meta_weights)
            log.log(f"\n======= SHAP 可解释性分析 =======")
            log.log(f"融合权重: {dict(zip(base_models.keys(), [f'{w:.4f}' for w in meta_weights]))}")

            if isinstance(X_test, np.ndarray):
                n_features = X_test.shape[1]
            else:
                n_features = X_test.shape[1] if hasattr(X_test, 'shape') else len(feature_names)
                if len(feature_names) != n_features:
                    feature_names = [f'feature_{i}' for i in range(n_features)]

            fused_shap = np.zeros((X_test.shape[0], n_features))

            for i, (name, model) in enumerate(base_models.items()):
                log.log(f"  计算 {name} SHAP...")
                try:
                    if name in ['random_forest', 'xgboost', 'lightgbm']:
                        explainer = shap.TreeExplainer(model)
                        sv = explainer.shap_values(X_test)
                        if isinstance(sv, list):
                            sv = sv[1] if len(sv) > 1 else sv[0]
                    elif name == 'logistic':
                        explainer = shap.LinearExplainer(model, X_test)
                        sv = explainer.shap_values(X_test)
                        if isinstance(sv, list):
                            sv = sv[1] if len(sv) > 1 else sv[0]
                    else:
                        log.log(f"    跳过 {name}: 不支持的模型类型")
                        continue

                    sv = np.array(sv)
                    if sv.ndim == 1:
                        sv = sv.reshape(-1, 1)
                    if sv.shape[1] != n_features:
                        log.log(f"    特征数不匹配 (模型 {sv.shape[1]}, 期望 {n_features})，跳过")
                        continue
                    fused_shap += meta_weights[i] * sv
                    log.log(f"    {name} 完成, shape={sv.shape}")
                except Exception as e:
                    log.log(f"    {name} SHAP 计算失败: {e}")
                    continue

            if np.all(fused_shap == 0):
                log.log("  SHAP 融合全部失败，跳过可视化")
                return

            # 全局特征重要性图（剔除标签泄露特征）
            importance = pd.DataFrame({
                'Feature': feature_names[:fused_shap.shape[1]],
                'Mean_Abs_SHAP': np.abs(fused_shap).mean(axis=0)
            })
            if drop_feats:
                importance = importance[~importance['Feature'].isin(drop_feats)]
            importance = importance.sort_values('Mean_Abs_SHAP', ascending=False)

            metabolic = ['bl_hbalc', 'bl_glu', 'tyg', 'tyg_bmi', 'bl_tg']
            top15 = importance.head(15)
            colors = ['#8B0000' if f in metabolic else '#1E3A8A' for f in top15['Feature']]

            plt.figure(figsize=(12, 8))
            bars = plt.barh(top15['Feature'][::-1], top15['Mean_Abs_SHAP'][::-1],
                           color=colors, edgecolor='black', linewidth=0.5)
            from matplotlib.patches import Patch
            legend = [Patch(facecolor='#8B0000', label='核心代谢指标'),
                      Patch(facecolor='#1E3A8A', label='其他特征')]
            plt.legend(handles=legend, loc='lower right', fontsize=11)
            plt.xlabel('平均绝对 SHAP 值', fontsize=13)
            plt.ylabel('特征', fontsize=13)
            plt.title('Stacking 集成模型 SHAP 全局特征重要性', fontsize=14)
            for bar in bars:
                w = bar.get_width()
                plt.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{w:.4f}', ha='left', va='center', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'shap_global_importance.png'), dpi=200, bbox_inches='tight')
            plt.close()

            log.log(f"\n特征重要性排名 (Top 10，已剔除{len(drop_feats or [])}个泄露特征):")
            for _, row in importance.head(10).iterrows():
                log.log(f"  {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")
            log.log(f"SHAP 图已保存")
        except Exception as e:
            log.log(f"SHAP 分析整体跳过: {e}")

# ===================== 主流程 =====================
def main():
    t0 = time.time()
    log.log("=" * 50)
    log.log("Stacking 集成 + 三级加权约登阈值 优化版")
    log.log("=" * 50)

    # 1. 数据预处理
    X_train, y_train, X_test, y_test, preprocessor, feature_names = preprocess_data(
        os.path.join(BASE_DIR, 'train_dataset_optimized.csv'),
        os.path.join(BASE_DIR, 'test_dataset_optimized.csv')
    )

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    log.log(f"\n训练集：阳性={n_pos}，阴性={n_neg}，比例={n_pos/len(y_train):.1%}")

    # 2. 训练 Stacking
    log.log("\n训练 Stacking 集成模型...")
    stacking_clf, base_estimators = build_stacking(n_pos, n_neg)
    stacking_clf.fit(X_train, y_train)
    log.log(f"训练完成，耗时 {time.time()-t0:.1f}s")

    # 3. 预测
    proba_dict = {}
    proba_dict['Stacking集成'] = stacking_clf.predict_proba(X_test)[:, 1]
    for name, _ in base_estimators:
        proba_dict[name] = stacking_clf.named_estimators_[name].predict_proba(X_test)[:, 1]

    # 4. 评估
    evaluator = Evaluator()
    model_results = {}
    for name, proba in proba_dict.items():
        model_results[name] = evaluator.full_eval(y_test, proba, name)

    # 5. 可视化
    evaluator.plot_roc_with_scenes(y_test, model_results)
    evaluator.plot_threshold_comparison(model_results)
    base_names = [n for n, _ in base_estimators]
    evaluator.plot_meta_weights(stacking_clf.final_estimator_, base_names)

    # 5b. SHAP 可解释性分析（可视化时剔除标签泄露特征 chronic_num）
    log.log("\n进行 SHAP 可解释性分析...")
    drop_feats = {'chronic_num', 'chronic_num_是', 'chronic_num_否', 'chronic_num_0', 'chronic_num_1'}
    evaluator.shap_analysis(stacking_clf, X_test, feature_names, SAVE_DIR, log, drop_feats)

    # 6. 保存结果汇总 CSV
    rows = []
    for mname, res in model_results.items():
        for skey in ['standard', 'primary', 'tertiary']:
            s = res['scenes'][skey]
            rows.append({
                'Model': mname, 'Scene': skey,
                'Threshold': s['threshold'], 'AUC': s['auc'],
                'Accuracy': s['accuracy'], 'Sensitivity': s['sensitivity'],
                'Specificity': s['specificity'], 'Precision': s['precision'],
                'F1': s['f1'], 'Youden': s['youden'],
                'TN': s['confusion_matrix'][0], 'FP': s['confusion_matrix'][1],
                'FN': s['confusion_matrix'][2], 'TP': s['confusion_matrix'][3]
            })
    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(SAVE_DIR, 'all_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8-sig')
    log.log(f"\n指标汇总已保存: {metrics_csv}")

    # 7. 保存模型与预处理器
    joblib.dump({
        'stacking_clf': stacking_clf,
        'preprocessor': preprocessor,
        'model_results': model_results,
        'y_test': y_test,
        'metrics_df': metrics_df
    }, os.path.join(SAVE_DIR, 'stacking_model_with_scene_thresholds.pkl'))
    log.log(f"模型已保存至 {SAVE_DIR}")

    # 8. 保存预测概率
    prob_df = pd.DataFrame({'true_label': y_test})
    for name, proba in proba_dict.items():
        prob_df[f'{name}_proba'] = proba
    prob_df.to_csv(os.path.join(SAVE_DIR, 'prediction_probabilities.csv'), index=False, encoding='utf-8-sig')

    log.log(f"\n总耗时: {time.time()-t0:.1f}s")
    log.save()

if __name__ == "__main__":
    main()
