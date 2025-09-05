import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置随机种子保证可复现性
np.random.seed(42)

# Step 1: 读取数据
file_path = 'sj.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path, index_col=0)  # 假设第一列是基因位点编号

# Step 2: 数据整理
data_transposed = data.transpose()

# 提取标签数据
labels = data_transposed['label_column_name'].values  # 替换为实际的标签列名

# 使用 LabelEncoder 对标签进行编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 移除标签列，剩下的就是特征数据
features = data_transposed.drop(columns=['label_column_name'])

# 处理缺失值 - 用特殊字符串 'missing' 替换 NaN
features = features.fillna('missing')

# 对每一列进行标签编码
label_encoders = {}
for column in features.columns:
    le = LabelEncoder()
    # 将整列转换为字符串，确保缺失值被编码
    features[column] = le.fit_transform(features[column].astype(str))
    label_encoders[column] = le

# 确保所有列名为字符串
features.columns = features.columns.astype(str)

# ====================== 方差阈值法（群体遗传学标准实现） ======================
def apply_variance_threshold(features, threshold_percent):
    """
    应用真正的方差阈值法（基于百分位数）
    
    参数:
    features -- 特征DataFrame (samples x SNPs)
    threshold_percent -- 方差异百分点阈值 (0-100), 例如10表示保留方差>第10百分位数的特征
    
    返回:
    筛选后的特征DataFrame和方差分布数据
    """
    print(f"\n{'='*60}")
    print(f"应用方差阈值法 (阈值={threshold_percent}%)")
    print('='*60)
    
    # 1. 计算每个SNP位点的方差
    variances = features.var()
    
    # 2. 计算指定百分位数的方差阈值
    threshold_value = np.percentile(variances, threshold_percent)
    print(f"方差阈值(第{threshold_percent}百分位数): {threshold_value:.4f}")
    
    # 3. 筛选方差超过阈值的特征
    selected_snps = variances[variances > threshold_value].index
    filtered_features = features[selected_snps]
    
    # 4. 计算筛选统计
    total_snps = features.shape[1]
    selected_count = len(selected_snps)
    filtered_percent = 100 * selected_count / total_snps
    
    print(f"原始SNP数量: {total_snps}")
    print(f"筛选后保留SNP数量: {selected_count} ({filtered_percent:.2f}%)")
    
    # 保存方差分布数据用于分析
    var_dist = pd.DataFrame({
        'snp': variances.index,
        'variance': variances.values,
        'threshold': threshold_value,
        'selected': variances.index.isin(selected_snps)
    })
    
    return filtered_features, var_dist

# 创建结果目录
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 定义方差阈值测试范围 (10%, 15%, 20%, 25%)
threshold_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
results = {}
best_threshold = None
best_accuracy = 0
variance_distributions = []

# 方差分布可视化
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# 使用 StratifiedKFold 以确保每一折具有相似的类分布
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 测试每个阈值
for thresh in threshold_percentages:
    # 应用方差阈值法
    filtered_features, var_dist = apply_variance_threshold(features, thresh)
    variance_distributions.append(var_dist)
    
    # 保存方差分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=var_dist, x='variance', hue='selected', bins=50, 
                 kde=True, palette={True: 'green', False: 'red'})
    plt.axvline(var_dist['threshold'].iloc[0], color='blue', linestyle='--')
    plt.title(f'SNP方差分布 (阈值={thresh}%)')
    plt.xlabel('方差')
    plt.ylabel('SNP数量')
    plt.savefig(f'{output_dir}/variance_distribution_{thresh}percent.png', dpi=300)
    plt.close()
    
    # =================== 数据标准化和模型 ============================
    # 构建 Pipeline 确保每折的处理方法是一致的
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补缺失值
        ('scaler', StandardScaler()),  # 标准化
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))  # 随机森林分类器
    ])
    
    # 执行交叉验证（10折）
    accuracy_scores = cross_val_score(pipeline, filtered_features, encoded_labels, cv=kf, scoring='accuracy')
    precision_scores = cross_val_score(pipeline, filtered_features, encoded_labels, cv=kf, scoring='precision_weighted')
    recall_scores = cross_val_score(pipeline, filtered_features, encoded_labels, cv=kf, scoring='recall_weighted')
    
    # 计算统计量
    accuracy_mean = np.mean(accuracy_scores)
    accuracy_std = np.std(accuracy_scores)
    precision_mean = np.mean(precision_scores)
    precision_std = np.std(precision_scores)
    recall_mean = np.mean(recall_scores)
    recall_std = np.std(recall_scores)
    
    # 存储结果
    results[thresh] = {
        'num_features': filtered_features.shape[1],
        'accuracy': {'mean': accuracy_mean, 'std': accuracy_std, 'scores': accuracy_scores},
        'precision': {'mean': precision_mean, 'std': precision_std, 'scores': precision_scores},
        'recall': {'mean': recall_mean, 'std': recall_std, 'scores': recall_scores}
    }
    
    print("\n交叉验证结果:")
    print(f"准确率 (Accuracy): {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"精确度 (Precision): {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"召回率 (Recall): {recall_mean:.4f} ± {recall_std:.4f}")
    print("="*60)
    
    # 检查是否是最佳结果
    if accuracy_mean > best_accuracy:
        best_accuracy = accuracy_mean
        best_threshold = thresh

# ====================== 结果分析 ======================
if best_threshold is not None:
    best_result = results[best_threshold]
    
    # 打印总结报告
    print("\n\n" + "="*80)
    print("方差阈值法特征选择结果汇总")
    print("="*80)
    print(f"{'阈值(%)':<10}{'保留特征数':<15}{'准确率(均值)':<15}{'精确度(均值)':<15}{'召回率(均值)':<15}")
    
    for thresh in threshold_percentages:
        res = results[thresh]
        print(f"{thresh:<10}{res['num_features']:<15}{res['accuracy']['mean']:.4f}{'':<8}{res['precision']['mean']:.4f}{'':<8}{res['recall']['mean']:.4f}")
    
    print("\n最佳模型:")
    print(f"  阈值: {best_threshold}% (第{best_threshold}百分位数)")
    print(f"  保留SNP数量: {best_result['num_features']}")
    print(f"  准确率: {best_result['accuracy']['mean']:.4f} ± {best_result['accuracy']['std']:.4f}")
    print("="*80)
    
    # 保存详细结果
    result_details = []
    for thresh in threshold_percentages:
        res = results[thresh]
        for cv_idx, (acc, prec, rec) in enumerate(zip(res['accuracy']['scores'], 
                                                      res['precision']['scores'], 
                                                      res['recall']['scores'])):
            result_details.append({
                'threshold_percent': thresh,
                'cv_fold': cv_idx + 1,
                'accuracy': acc,
                'precision': prec,
                'recall': rec
            })
    
    result_df = pd.DataFrame(result_details)
    result_df.to_csv(f'{output_dir}/variance_threshold_results.csv', index=False)
    
    # 特征数量与准确率可视化
    plt.figure(figsize=(12, 8))
    feature_counts = [results[t]['num_features'] for t in threshold_percentages]
    acc_means = [results[t]['accuracy']['mean'] for t in threshold_percentages]
    acc_stds = [results[t]['accuracy']['std'] for t in threshold_percentages]
    
    # 双轴图 (左侧：准确率，右侧：特征数量)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 准确率（左轴）
    ax1.set_xlabel('方差阈值 (%)')
    ax1.set_ylabel('准确率', color='b')
    ax1.errorbar(threshold_percentages, acc_means, yerr=acc_stds, 
                 fmt='o-', capsize=5, color='b', label='准确率')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 特征数量（右轴）
    ax2 = ax1.twinx()
    ax2.set_ylabel('保留特征数量', color='r')
    ax2.plot(threshold_percentages, feature_counts, 's--', color='r', label='特征数量')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 标题和图例
    plt.title('不同方差阈值下的模型性能与特征保留情况')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'{output_dir}/feature_performance_comparison.png', dpi=300)
    plt.close()
    
    # ====================== 使用最佳阈值训练最终模型 ======================
    print("\n使用最佳阈值训练最终模型...")
    best_features, _ = apply_variance_threshold(features, best_threshold)
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(best_features)
    
    # 训练最终模型
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(data_scaled, encoded_labels)
    
    # 保存预处理器和模型
    joblib.dump(scaler, f'{output_dir}/standard_scaler.pkl')
    joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
    joblib.dump(label_encoder, f'{output_dir}/label_encoder_for_labels.pkl')
    joblib.dump(final_model, f'{output_dir}/random_forest_model.pkl')
    
    # 保存选择的特征名
    selected_feature_names = list(best_features.columns)
    with open(f'{output_dir}/selected_features_{best_threshold}percent.txt', 'w') as f:
        f.write('\n'.join(selected_feature_names))
    
    # 输出总结
    print("\n" + "="*60)
    print("模型训练完成! 文件已保存:")
    print("="*60)
    print(f"最佳方差阈值: {best_threshold}% (第{best_threshold}百分位数)")
    print(f"最终保留SNP数量: {best_features.shape[1]}")
    print(f"验证准确率: {best_accuracy:.4f}")
    print(f"输出目录: {output_dir}/")
    print("\n保存的文件:")
    print(f"  - standard_scaler.pkl (标准化器)")
    print(f"  - label_encoders.pkl (特征编码器)")
    print(f"  - label_encoder_for_labels.pkl (标签编码器)")
    print(f"  - random_forest_model.pkl (随机森林模型)")
    print(f"  - selected_features_{best_threshold}percent.txt (选择的特征名称)")
    print(f"  - variance_threshold_results.csv (所有测试结果的详细数据)")
    print(f"  - variance_distribution_*.png (方差分布图)")
    print(f"  - feature_performance_comparison.png (特征性能比较图)")
    print("="*60)
else:
    print("错误: 没有找到最佳阈值")

# 保存方差分布数据用于后续分析
if variance_distributions:
    combined_var_dist = pd.concat(variance_distributions)
    combined_var_dist.to_csv(f'{output_dir}/variance_distributions.csv', index=False)
