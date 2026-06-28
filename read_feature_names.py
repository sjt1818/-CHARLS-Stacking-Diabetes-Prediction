import pandas as pd

def read_feature_names(train_path, test_path):
    # 读取训练集和测试集
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 显示数据集基本信息
    print(f"训练集形状: {train_df.shape} (行 × 列)")
    print(f"测试集形状: {test_df.shape} (行 × 列)")
    
    # 检查特征是否一致
    train_features = set(train_df.columns)
    test_features = set(test_df.columns)
    
    if train_features == test_features:
        print("\n✅ 训练集和测试集特征完全一致")
    else:
        print("\n⚠️ 训练集和测试集特征存在差异:")
        print(f"仅训练集有的特征: {train_features - test_features}")
        print(f"仅测试集有的特征: {test_features - train_features}")
    
    # 显示所有特征列名
    all_features = sorted(train_df.columns.tolist())
    print("\n======= 所有特征列名列表 =======")
    for i, feature in enumerate(all_features, 1):
        print(f"{i:3d}. {feature}")
    
    # 统计不同类型的特征数量
    numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\n特征类型统计:")
    print(f"数值型特征: {len(numeric_features)} 个")
    print(f"分类型特征: {len(categorical_features)} 个")
    
    return all_features, numeric_features, categorical_features

# 执行函数
if __name__ == "__main__":
    # 请确保文件路径与实际情况一致
    all_features, numeric_features, categorical_features = read_feature_names(
        train_path='train_dataset_optimized.csv',
        test_path='test_dataset_optimized.csv'
    )
    