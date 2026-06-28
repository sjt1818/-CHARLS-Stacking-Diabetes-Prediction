import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')

# ========================
# 全局配置（集中管理核心参数）
# ========================
CORE_METABOLIC_COLS = ["tyg", "tyg_bmi"]  # 核心代谢指标
LABEL_COL = "diabe"  # 标签列
MEDICAL_BOUNDS = {
    "bmi": (15, 50),
    "systo": (80, 200),
    "diasto": (50, 120),
    "bl_glu": (30, 400),
    "bl_hbalc": (3, 15),
    "bl_tg": (20, 1000),
    "tyg": (3, 8),
    "tyg_bmi": (50, 400),
}
DROP_FULL_MISSING_COLS = ["vgactx_c"]  # 初始完全缺失特征
STR_TO_NUM_MAP = {
    "mets": {"否": 0, "无": 0, "是": 1, "有": 1},
    "joga": {"否": 0, "无限制": 0, "是": 1, "有限制": 1},
    "lifta": {"否": 0, "无限制": 0, "是": 1, "有限制": 1},
    "work": {"否": 0, "无工作": 0, "是": 1, "有工作": 1},
    "retire": {"否": 0, "未退休": 0, "是": 1, "已退休": 1},
}
CATEGORICAL_COLS = [
    "ragender", "hrural", "rural2", "hukou", "marry", "bmicata",
    "smokev", "drinkev", "social4", "eyesight_distance", "hear"
]
LIFESTYLE_COLS = ["smoken", "drinkl"]
NUMERIC_COLS = [
    "age", "bmi", "mwaist", "systo", "diasto", "bl_glu", "bl_hbalc", "bl_tg"
] + CORE_METABOLIC_COLS
ONE_HOT_COLS = ["ragender", "hrural", "age_group", "bmicata", "marry"]


# ========================
# 1. 数据读取（含异常处理）
# ========================
def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        print(f"✅ 成功读取数据：{data_path}（{df.shape[0]}行 × {df.shape[1]}列）")
        return df
    except Exception as e:
        print(f"❌ 数据读取失败：{e}")
        exit()


# ========================
# 2. 字符串特征转数值（容错处理）
# ========================
def str_to_numeric(df):
    print("\n======= 字符串特征转数值 =======")
    for col, map_dict in STR_TO_NUM_MAP.items():
        if col in df.columns:
            original_missing = df[col].isnull().sum()
            try:
                df[col] = df[col].map(map_dict).astype(float)
                print(f"✅ {col}：缺失值 {df[col].isnull().sum()} 个（原始缺失：{original_missing}）")
            except Exception as e:
                print(f"⚠️ {col} 转换异常：{str(e)}，保留原始值")
    return df


# ========================
# 3. 检测并删除完全缺失特征
# ========================
def drop_full_missing_features(df):
    total_samples = len(df)
    full_missing_cols = [
        col for col in df.columns if df[col].isnull().sum() == total_samples
    ]
    if full_missing_cols:
        df = df.drop(full_missing_cols, axis=1)
        print(f"\n✅ 删除完全缺失特征：{full_missing_cols}（共{len(full_missing_cols)}个）")
    else:
        print("\n✅ 无完全缺失特征")
    return df, full_missing_cols


# ========================
# 4. 缺失值处理（分类+连续变量）
# ========================
def handle_missing_values(df):
    print("\n======= 缺失值处理 =======")
    # 分类变量：众数/默认值填充
    cat_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()
            fill_val = mode[0] if not mode.empty else ("未知" if df[col].dtype == "object" else 0)
            df[col] = df[col].fillna(fill_val)
            print(f"✅ {col}：填充值 {fill_val}，剩余缺失值 {df[col].isnull().sum()}")

    # 年龄及分组处理
    if "age" in df.columns:
        df["age"] = df["age"].fillna(df["age"].median())
        df["age_group"] = pd.cut(
            df["age"], bins=[45, 60, 80, 120], labels=["45-59", "60-79", "≥80"], right=False
        ).fillna("45-59")
        print("✅ 年龄及分组：已填充，无缺失值")

    # 核心生理指标：分组中位数/全局中位数填充
    core_cols = ["bmi", "systo", "diasto", "bl_glu", "bl_hbalc", "bl_tg"] + CORE_METABOLIC_COLS
    core_cols = [col for col in core_cols if col in df.columns]
    for col in core_cols:
        if df[col].isnull().sum() > 0:
            if "ragender" in df.columns and "age_group" in df.columns:
                df[col] = df.groupby(["ragender", "age_group"])[col].transform(
                    lambda x: x.fillna(x.median())
                )
            df[col] = df[col].fillna(df[col].median())
            print(f"✅ {col}：全局中位数填充，剩余缺失值 {df[col].isnull().sum()}")

    # 生活习惯变量：默认值填充
    lifestyle_cols = [col for col in LIFESTYLE_COLS if col in df.columns]
    for col in lifestyle_cols:
        if df[col].isnull().sum() > 0:
            fill_val = "不" if df[col].dtype == "object" else 0
            df[col] = df[col].fillna(fill_val)
            print(f"✅ {col}：填充值 {fill_val}，剩余缺失值 {df[col].isnull().sum()}")

    # 最终缺失值检查
    remaining = df.isnull().sum()[df.isnull().sum() > 0]
    if len(remaining) == 0:
        print("✅ 所有缺失值已处理")
    else:
        print("⚠️ 剩余非核心缺失值：\n", remaining)
    return df


# ========================
# 5. 异常值处理（医学范围/IQR）
# ========================
def handle_outliers(df):
    print("\n======= 异常值处理 =======")
    # 连续变量：医学范围或IQR裁剪
    num_cols = [col for col in NUMERIC_COLS if col in df.columns]
    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            print(f"🔄 {col}：强制转为数值型")
        if col in MEDICAL_BOUNDS:
            lower, upper = MEDICAL_BOUNDS[col]
            bound_type = "医学范围"
        else:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            bound_type = "IQR"
        before = len(df[(df[col] < lower) | (df[col] > upper)])
        df[col] = df[col].clip(lower, upper)
        print(f"✅ {col}：{bound_type}裁剪 {before} 个异常值 → 范围 [{lower:.1f}, {upper:.1f}]")

    # 分类变量：过滤无效值
    cat_filter_cols = ["ragender", "hrural", "bmicata"]
    cat_filter_cols = [col for col in cat_filter_cols if col in df.columns]
    for col in cat_filter_cols:
        valid = df[col].dropna().unique()
        invalid = len(df[~df[col].isin(valid)])
        df = df[df[col].isin(valid)]
        print(f"✅ {col}：过滤 {invalid} 个无效值 → 剩余样本 {len(df)}")
    return df


# ========================
# 6. 特征工程（衍生+去重）
# ========================
def feature_engineering(df):
    print("\n======= 特征工程 ========")
    # 删除重复特征
    if "tyg_index" in df.columns:
        df = df.drop("tyg_index", axis=1)
        print("✅ 删除重复特征 tyg_index")

    # 衍生身体活动强度
    if all(col in df.columns for col in ["mdact_c", "ltact_c"]):
        df["mdact_c_num"] = pd.factorize(df["mdact_c"])[0]
        df["ltact_c_num"] = pd.factorize(df["ltact_c"])[0]
        df["activity_level"] = df["mdact_c_num"] + df["ltact_c_num"]
        df = df.drop(["mdact_c_num", "ltact_c_num"], axis=1)
        print("✅ 衍生特征 activity_level（身体活动强度）")
    return df


# ========================
# 7. 编码与标准化（One-Hot+StandardScaler）
# ========================
def encode_and_scale(df, label_col):
    # 先记录原始列数（无论是否One-Hot，都需先保存）
    original_cols = df.columns.tolist()  

    # 分类特征One-Hot编码
    cat_cols = [col for col in ONE_HOT_COLS if col in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"✅ One-Hot编码：{cat_cols} → 新增 {len(df.columns)-len(original_cols)} 列")
    else:
        print("✅ 无分类特征需要One-Hot编码")

    # 连续特征标准化
    num_cols = [col for col in NUMERIC_COLS if col in df.columns]
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"✅ 标准化：{num_cols[:5]}...（共{len(num_cols)}个特征）")

    # 分离特征与标签
    X = df.drop(label_col, axis=1)
    y = df[label_col].astype(int)  # 确保标签为整数
    return X, y


# ========================
# 8. 数据集划分与欠采样（解决不平衡）
# ========================
def split_and_resample(X, y, test_size=0.3, random_state=42):
    # 划分训练集/测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print("\n======= 划分前数据集 =======")
    print(f"总样本：{len(X)} | 患病比例：{y.mean():.2%}")
    print(f"训练集：{len(X_train)} | 患病比例：{y_train.mean():.2%}")
    print(f"测试集：{len(X_test)} | 患病比例：{y_test.mean():.2%}")

    # 训练集欠采样（平衡正负样本）
    rus = RandomUnderSampler(random_state=random_state)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    print("\n======= 欠采样后训练集 =======")
    print(f"欠采样后：{len(X_train_res)} 样本 | 患病比例：{y_train_res.mean():.2%}")

    # 重组数据框
    train_df = pd.concat([X_train_res, y_train_res.rename(LABEL_COL)], axis=1)
    test_df = pd.concat([X_test, y_test.rename(LABEL_COL)], axis=1)
    return train_df, test_df, X_train_res, X_test, y_train_res, y_test


# ========================
# 主函数：整合所有流程
# ========================
def main():
    data_path = "final_charls_diabetes_data_with_new_features.csv"
    df = load_data(data_path)

    # 初步删除已知完全缺失特征
    initial_cols = df.shape[1]
    df = df.drop([col for col in DROP_FULL_MISSING_COLS if col in df.columns], axis=1)
    print(f"✅ 初步删除 {initial_cols-df.shape[1]} 个完全缺失特征")

    df = str_to_numeric(df)
    df, full_missing = drop_full_missing_features(df)

    # 数据探查
    print("\n======= 数据探查 =======")
    print("前5行：\n", df.head())
    key_cols = CORE_METABOLIC_COLS + ["bl_glu", "bl_hbalc", "bl_tg", "bmi", LABEL_COL]
    key_cols = [col for col in key_cols if col in df.columns]
    print("核心特征缺失值：\n", df[key_cols].isnull().sum()[df[key_cols].isnull().sum() > 0])
    df.info()

    # 核心预处理流程
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = feature_engineering(df)

    # 标签处理
    if LABEL_COL not in df.columns:
        print(f"❌ 无标签列 {LABEL_COL}")
        exit()
    if df[LABEL_COL].dtype == "object":
        df[LABEL_COL] = df[LABEL_COL].map({"是": 1, "否": 0}).fillna(0).astype(int)
        print("✅ 标签列转为数值：1=患病，0=未患病")

    # 编码、标准化、划分
    X, y = encode_and_scale(df, LABEL_COL)
    train_df, test_df, X_train, X_test, y_train, y_test = split_and_resample(X, y)

    # 保存数据
    train_df.to_csv("train_dataset_optimized.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv("test_dataset_optimized.csv", index=False, encoding="utf-8-sig")

    print("\n======= 预处理完成 =======")
    print(f"训练集路径：train_dataset_optimized.csv（{len(train_df)} 样本）")
    print(f"测试集路径：test_dataset_optimized.csv（{len(test_df)} 样本）")
    print(f"最终特征数：{X.shape[1]}（含 {len(CORE_METABOLIC_COLS)} 个核心代谢指标）")


if __name__ == "__main__":
    main()