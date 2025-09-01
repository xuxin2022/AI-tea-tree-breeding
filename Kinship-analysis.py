import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rogers_distance(train_data, test_data):
    """
    计算测试样本与训练群体之间的Rogers距离
    
    参数:
    train_data -- 训练集SNP数据 (DataFrame, 样本为行，SNP位点为列)
    test_data -- 测试集SNP数据 (DataFrame, 样本为行，SNP位点为列)
    
    返回:
    distances -- 每个测试样本到训练群体中心的Rogers距离
    """
    # 确保训练集和测试集有相同的SNP位点
    common_columns = train_data.columns.intersection(test_data.columns)
    if len(common_columns) == 0:
        raise ValueError("训练集和测试集没有共同的SNP位点")
    
    train_common = train_data[common_columns]
    test_common = test_data[common_columns]
    
    # 计算Rogers距离
    distances = []
    
    # 计算训练集的等位基因频率
    train_allele_freq = train_common.mean(axis=0) / 2  # 假设编码为0,1,2，所以除以2得到等位基因频率
    
    for _, test_sample in test_common.iterrows():
        # 计算每个位点的Rogers距离分量
        rogers_components = []
        
        for locus in common_columns:
            p = train_allele_freq[locus]  # 训练群体中该位点的等位基因频率
            q = 1 - p
            
            # 测试样本的基因型编码（假设为0,1,2）
            genotype = test_sample[locus]
            
            # 计算该位点的Rogers距离分量
            if genotype == 0:
                component = (0 - p)**2 + (1 - q)**2
            elif genotype == 1:
                component = (0.5 - p)**2 + (0.5 - q)**2
            elif genotype == 2:
                component = (1 - p)**2 + (0 - q)**2
            else:  # 处理缺失值或其他情况
                component = 0
            
            rogers_components.append(component)
        
        # 计算Rogers距离
        rogers_distance = np.sqrt(np.sum(rogers_components) / len(common_columns))
        distances.append(rogers_distance)
    
    return np.array(distances)

def preprocess_data(file_path, missing_threshold=0.2):
    """自适应读取并过滤高缺失率位点"""
    logger.info(f"读取数据文件: {file_path}")
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path, index_col=0)
    else:
        data = pd.read_csv(file_path, sep='\t', index_col=0)
    
    # 转置为样本行为特征列 
    samples = data.T 
    
    # 过滤缺失率超过阈值的列 
    missing_rates = samples.isnull().mean()
    filtered_columns = missing_rates[missing_rates <= missing_threshold].index
    filtered_data = samples[filtered_columns]
    
    logger.info(f"原始位点数: {samples.shape[1]}, 过滤后位点数: {filtered_data.shape[1]}")
    
    return filtered_data

def encode_genotypes(df):
    """将基因型字符串转换为数值编码"""
    label_encoders = {}
    encoded_data = df.copy() 
    
    # 统一数据类型并处理空值 
    encoded_data = encoded_data.apply(
        lambda x: x.astype(str).str.strip().replace('nan', 'NA')
    )
    
    for col in encoded_data.columns: 
        # 创建标签编码器 
        le = LabelEncoder()
        # 只对非NA值进行编码
        non_na_mask = encoded_data[col] != 'NA'
        
        # 获取唯一值并排序以确保一致性
        unique_vals = encoded_data.loc[non_na_mask, col].unique()
        unique_vals.sort()
        
        # 拟合编码器
        le.fit(unique_vals)
        
        # 转换数据
        encoded_data.loc[non_na_mask, col] = le.transform(encoded_data.loc[non_na_mask, col])
        encoded_data.loc[~non_na_mask, col] = np.nan  # 将NA值设为NaN
        
        label_encoders[col] = le.classes_.tolist() 
    
    return encoded_data.astype(float), label_encoders

def impute_data(train_df, test_df=None, n_neighbors=5):
    """使用KNN填充缺失值"""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # 使用训练集拟合imputer
    train_filled = imputer.fit_transform(train_df)
    train_result = pd.DataFrame(
        train_filled,
        index=train_df.index, 
        columns=train_df.columns  
    )
    
    # 如果有测试集，使用训练集拟合的imputer进行转换
    if test_df is not None:
        test_filled = imputer.transform(test_df)
        test_result = pd.DataFrame(
            test_filled,
            index=test_df.index,
            columns=test_df.columns
        )
        return train_result, test_result
    
    return train_result

def preprocess_and_calculate_distance(train_file, test_file, purple_samples=None, missing_threshold=0.2):
    """
    完整流程：预处理数据并计算遗传距离
    
    参数:
    train_file -- 训练集文件路径
    test_file -- 测试集文件路径
    purple_samples -- 紫化品种的样本ID列表
    missing_threshold -- 缺失率阈值
    
    返回:
    distance_results -- 包含距离结果的DataFrame
    """
    # 1. 预处理训练数据
    logger.info("预处理训练数据...")
    train_raw = preprocess_data(train_file, missing_threshold)
    
    logger.info("预处理测试数据...")
    test_raw = preprocess_data(test_file, missing_threshold)
    
    # 获取训练集和测试集共有的SNP位点
    common_columns = train_raw.columns.intersection(test_raw.columns)
    logger.info(f"共有位点数量: {len(common_columns)}")
    
    if len(common_columns) < 100:
        logger.warning("共有位点数量较少，可能影响分析结果")
    
    # 只保留共有的SNP位点
    train_raw = train_raw[common_columns]
    test_raw = test_raw[common_columns]
    
    # 2. 编码训练基因型
    logger.info("编码训练基因型...")
    train_encoded, train_encoders = encode_genotypes(train_raw)
    
    # 3. 预处理测试数据（编码）
    logger.info("编码测试基因型...")
    # 使用训练集的编码器确保一致编码
    test_encoded = test_raw.copy()
    for col in test_encoded.columns:
        if col in train_encoders:
            le = LabelEncoder()
            le.classes_ = np.array(train_encoders[col])
            
            # 处理测试集中出现的新基因型
            test_col = test_encoded[col].astype(str).str.strip().replace('nan', 'NA')
            unknown = set(test_col.unique()) - set(le.classes_)
            if unknown:
                logger.warning(f"位点 {col} 发现新基因型: {unknown}，将被编码为缺失")
                test_col = test_col.apply(lambda x: 'NA' if x in unknown else x)
            
            # 编码并处理缺失值
            non_na_mask = test_col != 'NA'
            test_encoded.loc[non_na_mask, col] = test_col.loc[non_na_mask].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else np.nan
            )
            test_encoded.loc[~non_na_mask, col] = np.nan
    
    # 4. 填充缺失值
    logger.info("填充缺失值...")
    train_filled, test_filled = impute_data(train_encoded, test_encoded)
    
    # 5. 计算遗传距离
    logger.info("计算Rogers距离...")
    distances = calculate_rogers_distance(train_filled, test_filled)
    
    # 6. 创建结果DataFrame
    distance_results = pd.DataFrame({
        'Sample_ID': test_filled.index,
        'Rogers_Distance': distances
    })
    
    # 添加品种类型信息
    if purple_samples is not None:
        # 假设purple_samples是紫化品种的样本ID列表
        distance_results['Variety_Type'] = distance_results['Sample_ID'].apply(
            lambda x: 'Purple' if x in purple_samples else 'Green'
        )
    else:
        # 如果没有提供品种信息，标记为未知
        distance_results['Variety_Type'] = 'Unknown'
    
    return distance_results

def visualize_results(results, output_prefix):
    """
    可视化分析结果
    
    参数:
    results -- 包含距离结果的DataFrame
    output_prefix -- 输出文件前缀
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # 1. 绘制距离分布图
    plt.figure(figsize=(10, 6))
    if 'Variety_Type' in results.columns and results['Variety_Type'].nunique() > 1:
        sns.histplot(data=results, x='Rogers_Distance', hue='Variety_Type', 
                    kde=True, alpha=0.5, element="step")
    else:
        sns.histplot(data=results, x='Rogers_Distance', kde=True)
    
    plt.title('Rogers Distance Distribution')
    plt.xlabel('Rogers Distance')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_distance_distribution.png", dpi=300)
    plt.close()
    
    # 2. 绘制箱线图（如果有品种信息）
    if 'Variety_Type' in results.columns and results['Variety_Type'].nunique() > 1:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=results, x='Variety_Type', y='Rogers_Distance')
        plt.title('Rogers Distance by Variety Type')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_distance_by_variety.png", dpi=300)
        plt.close()
    
    # 3. 输出统计信息
    stats_file = f"{output_prefix}_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Rogers Distance Statistics\n")
        f.write("=" * 30 + "\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"Mean: {results['Rogers_Distance'].mean():.4f}\n")
        f.write(f"Std: {results['Rogers_Distance'].std():.4f}\n")
        f.write(f"Min: {results['Rogers_Distance'].min():.4f}\n")
        f.write(f"Max: {results['Rogers_Distance'].max():.4f}\n\n")
        
        if 'Variety_Type' in results.columns and results['Variety_Type'].nunique() > 1:
            f.write("Statistics by Variety Type:\n")
            for variety in results['Variety_Type'].unique():
                variety_data = results[results['Variety_Type'] == variety]
                f.write(f"\n{variety}:\n")
                f.write(f"  Count: {len(variety_data)}\n")
                f.write(f"  Mean: {variety_data['Rogers_Distance'].mean():.4f}\n")
                f.write(f"  Std: {variety_data['Rogers_Distance'].std():.4f}\n")
    
    logger.info(f"可视化结果已保存至: {output_prefix}_*.png")
    logger.info(f"统计信息已保存至: {stats_file}")

# 主程序
if __name__ == "__main__":
    # 配置路径
    train_file = "cs.xlsx"  # 训练集文件路径
    test_file = "cs1.xlsx"  # 包含紫化和长绿品种的测试集
    
    # 紫化品种的样本ID列表（根据实际情况修改）
    purple_samples = ["sample1", "sample2", "sample3"]  # 示例
    
    # 执行分析
    logger.info("开始计算Rogers距离...")
    results = preprocess_and_calculate_distance(
        train_file, 
        test_file,
        purple_samples=purple_samples,  # 紫化品种的样本ID列表
        missing_threshold=0.2
    )
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"rogers_distance_results_{timestamp}"
    output_file = f"{output_prefix}.csv"
    results.to_csv(output_file, index=False)
    
    # 可视化结果
    visualize_results(results, output_prefix)
    
    logger.info(f"\n分析完成！结果已保存至: {output_file}")
    print("\n结果包含以下列:")
    print("1. Sample_ID: 样本名称")
    print("2. Rogers_Distance: 该样本到训练群体中心的Rogers距离")
    
    # 打印简要统计信息
    print("\n简要统计信息:")
    print(f"总样本数: {len(results)}")
    if 'Variety_Type' in results.columns:
        for variety in results['Variety_Type'].unique():
            count = len(results[results['Variety_Type'] == variety])
            mean_dist = results[results['Variety_Type'] == variety]['Rogers_Distance'].mean()
            print(f"{variety}品种: {count}个样本, 平均距离: {mean_dist:.4f}")
