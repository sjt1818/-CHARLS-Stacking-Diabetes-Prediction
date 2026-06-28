# ==============================================
# Pandas手动实现tableone基线特征表（无需装包）
# ==============================================
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

# 读取数据
df = pd.read_csv("final_charls_diabetes_data_with_new_features.csv", encoding="utf-8")
group0 = df[df["diabe"] == 0]  # 未患病
group1 = df[df["diabe"] == 1]  # 患病

# 分类变量统计函数
def cat_stats(col):
    total = df[col].value_counts(dropna=False)
    g0 = group0[col].value_counts(dropna=False)
    g1 = group1[col].value_counts(dropna=False)
    # 卡方检验
    contingency = pd.crosstab(df["diabe"], df[col])
    p = chi2_contingency(contingency)[1]
    return {
        "特征": col,
        "总体(n=16877)": f"{total.sum()} ({total.sum()/len(df)*100:.2f}%)",
        "未患病(n=15207)": f"{g0.sum()} ({g0.sum()/len(group0)*100:.2f}%)",
        "患病(n=1670)": f"{g1.sum()} ({g1.sum()/len(group1)*100:.2f}%)",
        "p值": f"{p:.4f}"
    }

# 连续变量统计函数（中位数IQR）
def cont_stats(col):
    total = df[col].describe(percentiles=[0.25, 0.75])
    g0 = group0[col].describe(percentiles=[0.25, 0.75])
    g1 = group1[col].describe(percentiles=[0.25, 0.75])
    # 秩和检验
    p = mannwhitneyu(group0[col].dropna(), group1[col].dropna())[1]
    return {
        "特征": col,
        "总体(n=16877)": f"{total['50%']:.2f} ({total['25%']:.2f}, {total['75%']:.2f})",
        "未患病(n=15207)": f"{g0['50%']:.2f} ({g0['25%']:.2f}, {g0['75%']:.2f})",
        "患病(n=1670)": f"{g1['50%']:.2f} ({g1['25%']:.2f}, {g1['75%']:.2f})",
        "p值": f"{p:.4f}"
    }

# 生成表格
categorical_cols = ["ragender", "hrural", "marry", "smokev", "drinkev"]  # 核心分类变量
continuous_cols = ["age", "tyg", "tyg_bmi", "bl_glu", "bl_hbalc", "bmi"]  # 核心连续变量

cat_table = pd.DataFrame([cat_stats(col) for col in categorical_cols])
cont_table = pd.DataFrame([cont_stats(col) for col in continuous_cols])
final_table = pd.concat([cont_table, cat_table], ignore_index=True)

# 导出Excel
final_table.to_excel("手动版基线特征表.xlsx", index=False)
print("✅ 手动版基线特征表已导出")