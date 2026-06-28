import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tempfile
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report, recall_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# 基础配置
os.environ["JOBLIB_TEMP_FOLDER"] = tempfile.gettempdir()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 全局参数
RANDOM_STATE = 42
BASE_DIR = os.getcwd()
SAVE_DIR = os.path.join(BASE_DIR, 'diabetes_nn_optimized_results')
os.makedirs(SAVE_DIR, exist_ok=True)

# 设置随机种子
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# 核心代谢特征（强化权重）
CORE_METABOLIC = ['tyg', 'tyg_bmi', 'bl_glu', 'bl_hbalc', 'bl_tg']

# 自定义特异度计算
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# ========== 数据预处理（优化特征保留） ==========
def preprocess_data(train_path, test_path, label_col='diabe'):
    # 安全读取数据
    def safe_read(file):
        try:
            return pd.read_csv(file, encoding='utf-8')
        except:
            return pd.read_csv(file, encoding='gbk')
    
    # 加载数据
    train = safe_read(train_path)
    test = safe_read(test_path)
    print(f"✅ 数据加载完成：训练集{train.shape[0]}例，测试集{test.shape[0]}例")
    print(f"\n📋 核心代谢特征：{[col for col in CORE_METABOLIC if col in train.columns]}")

    # 特征工程（强化核心代谢特征）
    def create_medical_features(df):
        # 核心代谢特征衍生（增强区分度）
        if 'bl_glu' in df.columns and 'bl_hbalc' in df.columns:
            df['glu_hbalc_ratio'] = df['bl_glu'] / (df['bl_hbalc'] + 1e-6)
            df['glu_hbalc_product'] = df['bl_glu'] * df['bl_hbalc']  # 新增乘积项
        else:
            df['glu_hbalc_ratio'] = 0
            df['glu_hbalc_product'] = 0
        
        if 'tyg_bmi' in df.columns:
            df['tyg_bmi_squared'] = df['tyg_bmi'] **2
            df['tyg_bmi_log'] = np.log1p(df['tyg_bmi'])  # 新增对数项
        else:
            df['tyg_bmi_squared'] = 0
            df['tyg_bmi_log'] = 0
        
        # 代谢异常标记（多维度）
        metabolic_conditions = []
        if 'bl_glu' in df.columns:
            metabolic_conditions.append(df['bl_glu'] > 7.0)
        if 'bl_hbalc' in df.columns:
            metabolic_conditions.append(df['bl_hbalc'] > 6.5)
        if 'tyg' in df.columns:
            metabolic_conditions.append(df['tyg'] > 1.7)  # 甘油三酯异常阈值
        
        if metabolic_conditions:
            df['metabolic_abnormality'] = np.sum(np.column_stack(metabolic_conditions), axis=1)  # 计数型标记
        else:
            df['metabolic_abnormality'] = 0
        
        # 异常值处理（温和裁剪，保留更多信息）
        for col in CORE_METABOLIC:
            if col in df.columns:
                q05 = df[col].quantile(0.05)
                q95 = df[col].quantile(0.95)
                df[col] = np.clip(df[col], q05, q95)  # 从1%/99%改为5%/95%
        
        return df
    
    train = create_medical_features(train)
    test = create_medical_features(test)

    # 提取标签
    y_train = train[label_col].map({'是':1, '否':0, 1:1, 0:0}).fillna(0).astype(int)
    y_test = test[label_col].map({'是':1, '否':0, 1:1, 0:0}).fillna(0).astype(int)
    X_train = train.drop(label_col, axis=1)
    X_test = test.drop(label_col, axis=1)

    # 特征分类
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    print(f"\n📊 特征分类：数值特征{len(num_cols)}个，分类特征{len(cat_cols)}个")

    # 预处理管道（优化特征选择：保留更多特征）
    num_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        # 关键优化：特征选择k值从15改为min(30, len(num_cols))，保留更多特征
        ('feature_selection', SelectKBest(f_classif, k=min(30, len(num_cols))))
    ])
    
    cat_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_preprocessor, num_cols),
            ('cat', cat_preprocessor, cat_cols)
        ],
        remainder='drop'
    )

    # 执行预处理
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 缺失值兜底
    X_train_processed = np.nan_to_num(X_train_processed)
    X_test_processed = np.nan_to_num(X_test_processed)

    return X_train_processed, y_train, X_test_processed, y_test, preprocessor

# ========== 增强型神经网络（提升拟合能力） ==========
def build_diabetes_nn(input_dim):
    """增强型模型：适度增加复杂度，提升AUC，仍适配CPU"""
    model = Sequential([
        # 输入层 + 隐藏层1（适度增加节点数）
        Dense(256, input_dim=input_dim, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),  # 降低dropout率，减少信息丢失
        
        # 隐藏层2
        Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.15),
        
        # 隐藏层3（新增，提升拟合能力）
        Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.1),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ])
    
    # 优化器：调整学习率，适配更长训练
    optimizer = Nadam(learning_rate=0.0008, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# ========== 模型评估与可视化（保留毕业设计格式） ==========
class DiabetesNNAnalyzer:
    @staticmethod
    def calculate_youdens_threshold(y_true, y_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        youden_j = tpr - fpr
        standard_idx = np.argmax(youden_j)
        standard_threshold = thresholds[standard_idx]
        
        primary_j = 1.2 * tpr + 0.8 * (1 - fpr) - 1
        primary_idx = np.argmax(primary_j)
        primary_threshold = thresholds[primary_idx]
        
        tertiary_j = 0.8 * tpr + 1.2 * (1 - fpr) - 1
        tertiary_idx = np.argmax(tertiary_j)
        tertiary_threshold = thresholds[tertiary_idx]
        
        return {
            'standard': {'threshold': standard_threshold, 'fpr': fpr[standard_idx], 'tpr': tpr[standard_idx]},
            'primary': {'threshold': primary_threshold, 'fpr': fpr[primary_idx], 'tpr': tpr[primary_idx]},
            'tertiary': {'threshold': tertiary_threshold, 'fpr': fpr[tertiary_idx], 'tpr': tpr[tertiary_idx]}
        }
    
    @staticmethod
    def evaluate_model(y_true, y_proba, thresholds, model_name):
        print(f"\n======= {model_name} 性能评估（高AUC版） =======")
        scenes = {
            'standard': '标准平衡场景（约登指数最优）',
            'primary': '基层医院场景（优先减少漏诊）',
            'tertiary': '三级医院场景（优先减少误诊）'
        }
        
        results = {}
        for scene_key, scene_name in scenes.items():
            threshold = thresholds[scene_key]['threshold']
            y_pred = (y_proba >= threshold).astype(int)
            
            sensitivity = recall_score(y_true, y_pred)
            specificity = specificity_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba)
            
            print(f"\n📌 {scene_name}")
            print(f"   阈值：{threshold:.4f}")
            print(f"   AUC：{auc:.4f} | 灵敏度：{sensitivity:.4f}（漏诊率：{1-sensitivity:.4f}）")
            print(f"   特异度：{specificity:.4f}（误诊率：{1-specificity:.4f}）")
            print("   分类报告：")
            print(classification_report(y_true, y_pred, target_names=['非糖尿病', '糖尿病'], digits=3))
            
            results[scene_key] = {
                'auc': auc, 'sensitivity': sensitivity, 'specificity': specificity,
                'threshold': threshold, 'y_pred': y_pred
            }
        
        return results
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba, thresholds, save_path):
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, color='#2196F3', linewidth=2.5, label=f'神经网络模型 (AUC={auc:.4f})')
        
        plt.scatter(thresholds['standard']['fpr'], thresholds['standard']['tpr'], 
                   color='black', s=100, marker='o', edgecolors='white', label='标准阈值')
        plt.scatter(thresholds['primary']['fpr'], thresholds['primary']['tpr'], 
                   color='green', s=100, marker='^', edgecolors='white', label='基层阈值')
        plt.scatter(thresholds['tertiary']['fpr'], thresholds['tertiary']['tpr'], 
                   color='red', s=100, marker='s', edgecolors='white', label='三级阈值')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')
        plt.xlabel('假阳性率（误诊率）', fontsize=12)
        plt.ylabel('真阳性率（灵敏度）', fontsize=12)
        plt.title('糖尿病预测神经网络ROC曲线及最优阈值（高AUC版）', fontsize=14, pad=15)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'nn_roc_curve_high_auc.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n✅ ROC曲线已保存至：{os.path.join(save_path, 'nn_roc_curve_high_auc.png')}")
    
    @staticmethod
    def plot_training_history(history, save_path):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='训练损失', linewidth=2)
        plt.plot(history['val_loss'], label='验证损失', linewidth=2)
        plt.title('模型损失变化（高AUC版）', fontsize=12)
        plt.xlabel('训练轮数', fontsize=10)
        plt.ylabel('损失值', fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['auc'], label='训练AUC', linewidth=2)
        plt.plot(history['val_auc'], label='验证AUC', linewidth=2)
        plt.title('模型AUC变化（高AUC版）', fontsize=12)
        plt.xlabel('训练轮数', fontsize=10)
        plt.ylabel('AUC值', fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'nn_training_history_high_auc.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✅ 训练过程图已保存至：{os.path.join(save_path, 'nn_training_history_high_auc.png')}")

# ========== 主函数（优化训练策略） ==========
def main():
    print("======= 糖尿病预测神经网络（高AUC版） =======")
    print(f"📌 TensorFlow版本：{tf.__version__}")
    print(f"📌 GPU支持：{'是' if tf.config.list_physical_devices('GPU') else '否（使用CPU训练）'}\n")
    
    # 1. 数据预处理
    X_train, y_train, X_test, y_test, preprocessor = preprocess_data(
        os.path.join(BASE_DIR, 'train_dataset_optimized.csv'),
        os.path.join(BASE_DIR, 'test_dataset_optimized.csv'),
        label_col='diabe'
    )
    
    # 2. 数据分布
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    print(f"\n📈 训练集分布：")
    print(f"   糖尿病样本：{n_pos}例 | 非糖尿病样本：{n_neg}例 | 均衡度：{n_pos/len(y_train):.1%}")
    
    # 3. 构建模型
    input_dim = X_train.shape[1]
    model = build_diabetes_nn(input_dim)
    print(f"\n✅ 增强型模型构建完成，输入维度：{input_dim}")
    
    # 4. 训练策略优化（提升AUC关键）
    print("\n======= 开始训练模型（高AUC版） =======")
    callbacks = [
        # 调整早停策略：patience=8，给模型更多训练时间
        EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True, verbose=1),
        # 学习率衰减：训练后期降低学习率，提升收敛效果
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        ModelCheckpoint(os.path.join(SAVE_DIR, 'best_diabetes_model_high_auc.keras'), 
                       monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    ]
    
    # 训练参数优化：增加epochs，适配CPU的batch_size
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=50,  # 增加训练轮数，充分学习
        batch_size=24,  # 适度调整批次，平衡训练速度和效果
        class_weight={0:1.0, 1:1.0},
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. 加载最优模型
    best_model = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'best_diabetes_model_high_auc.keras'))
    print(f"\n✅ 最优高AUC模型已保存至：{os.path.join(SAVE_DIR, 'best_diabetes_model_high_auc.keras')}")
    
    # 6. 预测与评估
    y_proba = best_model.predict(X_test, batch_size=24).ravel()
    thresholds = DiabetesNNAnalyzer.calculate_youdens_threshold(y_test, y_proba)
    eval_results = DiabetesNNAnalyzer.evaluate_model(y_test, y_proba, thresholds, '糖尿病预测神经网络（高AUC版）')
    
    # 7. 可视化
    DiabetesNNAnalyzer.plot_roc_curve(y_test, y_proba, thresholds, SAVE_DIR)
    DiabetesNNAnalyzer.plot_training_history(history.history, SAVE_DIR)
    
    # 8. 保存结果
    results = {
        'preprocessor': preprocessor,
        'thresholds': thresholds,
        'evaluation': eval_results,
        'y_test': y_test,
        'y_proba': y_proba,
        'training_history': history.history
    }
    joblib.dump(results, os.path.join(SAVE_DIR, 'diabetes_nn_high_auc_results.pkl'))
    print(f"\n✅ 所有高AUC版结果已保存至：{os.path.join(SAVE_DIR, 'diabetes_nn_high_auc_results.pkl')}")
    
    print("\n======= 高AUC版模型运行完成！=======")
    print(f"📁 输出文件路径：{SAVE_DIR}")
    print("📋 核心优化点：增强模型结构、保留更多特征、优化训练策略")

if __name__ == "__main__":
    main()