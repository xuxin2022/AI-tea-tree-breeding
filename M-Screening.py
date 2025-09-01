import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 设置随机种子确保结果可复现
np.random.seed(42)

# Step 1: 读取数据
file_path = 'cs.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path, index_col=0)  # 假设第一列是基因位点编号

# Step 2: 数据整理
data_transposed = data.transpose()

# 提取标签数据
labels = data_transposed['label_column_name'].values  # 替换为实际的标签列名

# 使用LabelEncoder对标签进行编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 移除标签列，剩下的就是特征数据
features = data_transposed.drop(columns=['label_column_name'])

# 处理缺失值 - 用特殊字符串'missing'替换NaN
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

# 创建输出目录
output_dir = "rf_importance_results"
os.makedirs(output_dir, exist_ok=True)

# ====================== 随机森林特征重要性筛选 ======================

# 使用全部特征训练初始随机森林模型
print("\n" + "="*80)
print("步骤1: 训练初始随机森林模型计算特征重要性")
print("="*80)
scaler = StandardScaler()
full_data_scaled = scaler.fit_transform(features)
rf_full = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)  # 增加树的数量提高重要性估计精度
rf_full.fit(full_data_scaled, encoded_labels)

# 获取特征重要性
importances = rf_full.feature_importances_
feature_importance_df = pd.DataFrame({
    'SNP': features.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# 保存特征重要性结果
feature_importance_df.to_csv(f'{output_dir}/full_feature_importances.csv', index=False)
print(f"特征重要性计算完成! 已保存至: {output_dir}/full_feature_importances.csv")

# 可视化特征重要性
plt.figure(figsize=(15, 10))
sns.barplot(x='Importance', y='SNP', data=feature_importance_df.head(50))
plt.title('Top 50 最重要的SNP特征 (随机森林)')
plt.tight_layout()
plt.savefig(f'{output_dir}/top_50_feature_importances.png', dpi=300)
plt.close()

# 计算重要性阈值 - 基于均值的百分数
print("\n" + "="*80)
print("步骤2: 计算基于重要性的筛选阈值")
print("="*80)
avg_importance = importances.mean()
print(f"平均特征重要性: {avg_importance:.6f}")

# 定义高于平均值的百分比阈值
threshold_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # 高于均值的5%-35%
threshold_values = [avg_importance * (1 + p/100) for p in threshold_percentages]

# 显示不同阈值对应的特征数量
for p, thresh in zip(threshold_percentages, threshold_values):
    count = sum(importances > thresh)
    print(f"阈值: +{p}% (绝对值: {thresh:.6f}) | 保留特征数: {count}")

# ====================== 不同阈值下的模型性能评估 ======================
print("\n" + "="*80)
print("步骤3: 不同阈值下的交叉验证评估")
print("="*80)

results = {}
best_threshold = None
best_accuracy = 0

for p, thresh in zip(threshold_percentages, threshold_values):
    print(f"\n正在测试阈值: +{p}% 高于平均值 (绝对值: {thresh:.6f})")
    
    # 选择高于阈值的特征
    selected_features = feature_importance_df[feature_importance_df['Importance'] > thresh]['SNP']
    selected_feature_names = selected_features.tolist()
    print(f"保留特征数: {len(selected_feature_names)}")
    
    # 准备选定特征的数据
    X_selected = features[selected_feature_names]
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X_selected)
    
    # 初始化模型
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 执行交叉验证
    accuracy_scores = cross_val_score(model, data_scaled, encoded_labels, cv=10, scoring='accuracy')
    precision_scores = cross_val_score(model, data_scaled, encoded_labels, cv=10, scoring='precision_weighted')
    recall_scores = cross_val_score(model, data_scaled, encoded_labels, cv=10, scoring='recall_weighted')
    
    # 计算统计量
    accuracy_mean = np.mean(accuracy_scores)
    accuracy_std = np.std(accuracy_scores)
    precision_mean = np.mean(precision_scores)
    precision_std = np.std(precision_scores)
    recall_mean = np.mean(recall_scores)
    recall_std = np.std(recall_scores)
    
    # 存储结果
    results[p] = {
        'threshold_percent': p,
        'threshold_value': thresh,
        'num_features': len(selected_feature_names),
        'features': selected_feature_names,
        'accuracy': accuracy_mean,
        'accuracy_std': accuracy_std,
        'accuracy_scores': accuracy_scores,
        'precision': precision_mean,
        'precision_std': precision_std,
        'recall': recall_mean,
        'recall_std': recall_std,
    }
    
    print(f"交叉验证结果:")
    print(f"  准确率: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"  精确度: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"  召回率: {recall_mean:.4f} ± {recall_std:.4f}")
    
    # 检查是否是最佳结果
    if accuracy_mean > best_accuracy:
        best_accuracy = accuracy_mean
        best_threshold = p
        best_features = selected_feature_names.copy()

# ====================== 结果分析与可视化 ======================
print("\n" + "="*80)
print("步骤4: 结果分析与可视化")
print("="*80)

# 创建结果DataFrame
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df = results_df.sort_values('threshold_percent')

# 保存详细结果
results_df.to_csv(f'{output_dir}/rf_importance_threshold_results.csv')

# 特征数量与准确率关系图
plt.figure(figsize=(12, 8))
plt.errorbar(
    results_df['threshold_percent'], 
    results_df['accuracy'], 
    yerr=results_df['accuracy_std'], 
    fmt='o-', 
    capsize=5,
    label='准确率'
)
plt.plot(
    results_df['threshold_percent'], 
    results_df['num_features']/max(results_df['num_features']), 
    's--', 
    label='保留特征比例'
)
plt.xlabel('阈值 (%)')
plt.ylabel('性能 / 特征比例')
plt.title('随机森林特征重要性筛选性能比较')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/rf_performance_comparison.png', dpi=300)
plt.close()

# 重要性密度与阈值图
plt.figure(figsize=(12, 8))
sns.kdeplot(importances, fill=True, label='特征重要性分布')
for p in threshold_percentages:
    if p == best_threshold:
        color = 'red'
        alpha = 1.0
        label = f'最佳阈值 (+{p}%)'
    else:
        color = 'gray'
        alpha = 0.5
        label = f'+{p}%'
    
    plt.axvline(
        avg_importance * (1 + p/100), 
        color=color, 
        linestyle='--', 
        alpha=alpha,
        label=label
    )

plt.axvline(avg_importance, color='blue', linestyle='-', label='平均重要性')
plt.xlabel('特征重要性')
plt.ylabel('密度')
plt.title('特征重要性分布与筛选阈值')
plt.legend()
plt.savefig(f'{output_dir}/feature_importance_thresholds.png', dpi=300)
plt.close()

# ====================== 最佳模型训练与保存 ======================
print("\n" + "="*80)
print("步骤5: 使用最佳阈值训练最终模型")
print("="*80)

if best_threshold is not None:
    print(f"选择的最佳阈值: +{best_threshold}% 高于平均值")
    print(f"保留SNP数量: {len(best_features)}")
    print(f"交叉验证准确率: {best_accuracy:.4f}")
    
    # 准备选定特征的数据
    X_best = features[best_features]
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X_best)
    
    # 训练最终模型
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(data_scaled, encoded_labels)
    
    # 保存预处理器和模型
    joblib.dump(scaler, f'{output_dir}/standard_scaler.pkl')
    joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
    joblib.dump(label_encoder, f'{output_dir}/label_encoder_for_labels.pkl')
    joblib.dump(final_model, f'{output_dir}/random_forest_model.pkl')
    
    # 保存选择的特征名
    with open(f'{output_dir}/selected_features_{best_threshold}percent.txt', 'w') as f:
        f.write('\n'.join(best_features))
    
    # 保存特征重要性
    best_importances = feature_importance_df[feature_importance_df['SNP'].isin(best_features)]
    best_importances.to_csv(f'{output_dir}/selected_features_importances.csv', index=False)
    
    print("\n" + "="*60)
    print("模型训练完成! 文件已保存:")
    print("="*60)
    print(f"最佳阈值: +{best_threshold}% 高于平均值")
    print(f"最终保留SNP数量: {len(best_features)}")
    print(f"验证准确率: {best_accuracy:.4f}")
    print(f"输出目录: {output_dir}/")
    print("\n保存的文件:")
    print(f"  - standard_scaler.pkl (标准化器)")
    print(f"  - label_encoders.pkl (特征编码器)")
    print(f"  - label_encoder_for_labels.pkl (标签编码器)")
    print(f"  - random_forest_model.pkl (随机森林模型)")
    print(f"  - selected_features_{best_threshold}percent.txt (选择的特征名称)")
    print(f"  - selected_features_importances.csv (选择特征的详细重要性)")
    print(f"  - rf_importance_threshold_results.csv (所有测试结果)")
    print(f"  - full_feature_importances.csv (所有特征的重要性)")
    print(f"  - top_50_feature_importances.png (最重要的50个特征)")
    print(f"  - rf_performance_comparison.png (性能比较图)")
    print(f"  - feature_importance_thresholds.png (特征重要性分布图)")
    print("="*60)
    
    # 可视化选定的重要特征
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Importance', y='SNP', data=best_importances.head(20))
    plt.title(f'Top 20 最重要的选定SNP (阈值: +{best_threshold}%)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/selected_top_20_important_features.png', dpi=300)
    plt.close()
else:
    print("错误: 未能确定最佳阈值")

print("\n" + "="*80)
print("随机森林重要性筛选分析完成!")
print("="*80)
