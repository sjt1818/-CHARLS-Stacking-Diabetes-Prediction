# ==============================================
#  CHARLS 原始数据集（建模前）特征统计代码
#  输入：charls_wave3.csv（原始元数据）
#  输出：CHARLS原始数据集特征统计.xlsx（论文可用）
# ==============================================
import pandas as pd
import numpy as np

# ---------------------- 1. 读取原始数据 ----------------------
# 注意：如果编码报错，将encoding改为"gbk"或"gb2312"
df_raw = pd.read_csv(
    "charls_wave3.csv",
    encoding="utf-8"
)

# 读取最终建模用的中间数据集列名（用于关联原始→最终特征）
df_final_cols = pd.read_csv(
    "final_charls_diabetes_data_with_new_features.csv",
    nrows=0  # 仅读取列名，不读数据
)
final_feature_list = df_final_cols.columns.tolist()

# ---------------------- 2. 基础信息统计（论文表1用） ----------------------
print("="*60)
print("【1. 原始数据集基础信息】")
print(f"总样本量：{df_raw.shape[0]} 例")
print(f"总特征数：{df_raw.shape[1]} 个")
print(f"数值型特征数：{df_raw.select_dtypes(include=[np.number]).shape[1]} 个")
print(f"分类型特征数：{df_raw.select_dtypes(exclude=[np.number]).shape[1]} 个")
print(f"总缺失值数量：{df_raw.isnull().sum().sum()} 个")

# 保存基础信息
basic_info = pd.DataFrame({
    "统计指标": ["总样本量", "总特征数", "数值型特征数", "分类型特征数", "总缺失值数"],
    "数值": [
        df_raw.shape[0],
        df_raw.shape[1],
        df_raw.select_dtypes(include=[np.number]).shape[1],
        df_raw.select_dtypes(exclude=[np.number]).shape[1],
        df_raw.isnull().sum().sum()
    ]
})

# ---------------------- 3. 特征分类（按糖尿病研究维度，可自行调整列名） ----------------------
print("\n" + "="*60)
print("【2. 特征分类统计（按研究维度）】")

# 【关键】根据CHARLS 2015变量表，按你的研究维度分类
# 请根据你原始数据的实际列名，修改下面的列表！
feature_categories = {
    "标签特征": ["dc007"],  # 糖尿病患病标签（1=患病/0=未患病）
    "核心代谢指标": ["bd010", "bd011", "bd012", "bd013", "bd014"],  # 甘油三酯、血糖、糖化血红蛋白等
    "人口统计学特征": ["ba002", "ba004", "ba005", "ba006", "ba007"],  # 性别、年龄、户籍、城乡、婚姻
    "体格检查特征": ["bd001", "bd002", "bd004", "bd005", "bd006"],  # 身高、体重、腰围、血压
    "生活方式特征": ["be001", "be002", "be003", "be005", "bf001"],  # 吸烟、饮酒、运动、睡眠
    "健康与共病特征": ["bc001", "bc002", "bc003", "dc001"],  # 高血压、血脂异常、慢性病数量
    "其他辅助特征": ["ca001", "cb001", "cb002", "cc001"]  # 认知、视力、听力、社会活动
}

# 生成分类统计（仅保留数据集中存在的列）
category_stats = []
for cat, cols in feature_categories.items():
    valid_cols = [col for col in cols if col in df_raw.columns]
    category_stats.append({
        "特征类别": cat,
        "特征数量": len(valid_cols),
        "特征列表（原始列名）": ", ".join(valid_cols)
    })
category_df = pd.DataFrame(category_stats)
print(category_df)

# ---------------------- 4. 缺失值统计（论文表2用） ----------------------
print("\n" + "="*60)
print("【3. 缺失值统计（按缺失率降序，仅显示缺失率>0的特征）】")
missing_stats = df_raw.isnull().sum().reset_index()
missing_stats.columns = ["特征名称（原始列名）", "缺失数量"]
missing_stats["缺失率(%)"] = round(missing_stats["缺失数量"] / df_raw.shape[0] * 100, 2)
missing_stats = missing_stats.sort_values(by="缺失率(%)", ascending=False).reset_index(drop=True)
missing_stats = missing_stats[missing_stats["缺失率(%)"] > 0]  # 仅保留有缺失的特征
print(missing_stats.head(20))  # 打印前20个缺失率最高的特征

# ---------------------- 5. 数值型特征描述性统计（论文表3用） ----------------------
print("\n" + "="*60)
print("【4. 数值型特征描述性统计】")
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
numeric_stats = df_raw[numeric_cols].describe().T  # 转置，方便阅读
numeric_stats = numeric_stats[["count", "mean", "50%", "std", "min", "max"]]
numeric_stats.columns = ["有效样本量", "均值", "中位数", "标准差", "最小值", "最大值"]
print(numeric_stats.head(20))

# ---------------------- 6. 分类型特征频数统计 ----------------------
print("\n" + "="*60)
print("【5. 分类型特征频数统计】")
categorical_cols = df_raw.select_dtypes(exclude=[np.number]).columns
cat_stats_list = []
for col in categorical_cols:
    freq = df_raw[col].value_counts(dropna=False)
    for val, cnt in freq.items():
        cat_stats_list.append({
            "特征名称（原始列名）": col,
            "类别取值": val,
            "频数": cnt,
            "频率(%)": round(cnt / df_raw.shape[0] * 100, 2)
        })
cat_stats_df = pd.DataFrame(cat_stats_list)
print(cat_stats_df.head(20))

# ---------------------- 7. 最终建模特征的原始缺失统计（关联原始→final） ----------------------
print("\n" + "="*60)
print("【6. 最终建模特征的原始缺失情况】")
# 筛选出最终建模用的特征，统计其在原始数据中的缺失情况
final_missing = df_raw[final_feature_list].isnull().sum().reset_index()
final_missing.columns = ["最终特征名", "原始缺失数量"]
final_missing["原始缺失率(%)"] = round(final_missing["原始缺失数量"] / df_raw.shape[0] * 100, 2)
final_missing = final_missing.sort_values(by="原始缺失率(%)", ascending=False).reset_index(drop=True)
print(final_missing.head(20))

# ---------------------- 8. 导出所有结果到Excel（论文直接用） ----------------------
output_path = "CHARLS原始数据集特征统计.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    basic_info.to_excel(writer, sheet_name="1.基础信息", index=False)
    category_df.to_excel(writer, sheet_name="2.特征分类", index=False)
    missing_stats.to_excel(writer, sheet_name="3.缺失值统计", index=False)
    numeric_stats.to_excel(writer, sheet_name="4.数值特征统计", index=True)
    cat_stats_df.to_excel(writer, sheet_name="5.分类特征统计", index=False)
    final_missing.to_excel(writer, sheet_name="6.最终建模特征原始缺失", index=False)

print(f"\n✅ 所有特征统计结果已导出至：{output_path}")
print("="*60)