from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

# ====================== 不同阈值下的模型性能评估 ======================
print("\n" + "="*80)
print("步骤2: 不同阈值下的交叉验证评估")
print("="*80)

results = {}
best_threshold = None
best_accuracy = 0

# 使用 StratifiedKFold 以确保每一折具有相似的类分布
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for p, thresh in zip(threshold_percentages, threshold_values):
    print(f"\n正在测试阈值: +{p}% 高于平均值 (绝对值: {thresh:.6f})")
    
    # 选择高于阈值的特征
    selected_features = feature_importance_df[feature_importance_df['Importance'] > thresh]['SNP']
    selected_feature_names = selected_features.tolist()
    print(f"保留特征数: {len(selected_feature_names)}")
    
    # 准备选定特征的数据
    X_selected = features[selected_feature_names]
    
    # =================== 使用Pipeline避免数据泄漏 ====================
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 填补缺失值
        ('scaler', StandardScaler()),  # 标准化
        ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))  # 随机森林分类器
    ])
    
    # 执行交叉验证
    accuracy_scores = cross_val_score(pipeline, X_selected, encoded_labels, cv=kf, scoring='accuracy')
    precision_scores = cross_val_score(pipeline, X_selected, encoded_labels, cv=kf, scoring='precision_weighted')
    recall_scores = cross_val_score(pipeline, X_selected, encoded_labels, cv=kf, scoring='recall_weighted')
    f1_scores = cross_val_score(pipeline, X_selected, encoded_labels, cv=kf, scoring='f1_weighted')
    
    # 计算统计量
    accuracy_mean = np.mean(accuracy_scores)
    accuracy_std = np.std(accuracy_scores)
    precision_mean = np.mean(precision_scores)
    precision_std = np.std(precision_scores)
    recall_mean = np.mean(recall_scores)
    recall_std = np.std(recall_scores)
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    
    # 存储结果
    results[p] = {
        'threshold_percent': p,
        'threshold_value': thresh,
        'num_features': len(selected_feature_names),
        'features': selected_feature_names,
        'accuracy': accuracy_mean,
        'accuracy_std': accuracy_std,
        'precision': precision_mean,
        'precision_std': precision_std,
        'recall': recall_mean,
        'recall_std': recall_std,
        'f1_score': f1_mean,
        'f1_score_std': f1_std,
    }
    
    print(f"交叉验证结果: ")
    print(f"  准确率: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"  精确度: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"  召回率: {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"  F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
    
    # 检查是否是最佳结果
    if accuracy_mean > best_accuracy:
        best_accuracy = accuracy_mean
        best_threshold = p
        best_features = selected_feature_names.copy()

# 进一步分析并保存结果
print("\n" + "="*80)
print("步骤3: 结果分析与可视化")
print("="*80)

# 创建结果DataFrame
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df = results_df.sort_values('threshold_percent')

# 保存详细结果
results_df.to_csv(f'{output_dir}/rf_importance_threshold_results.csv')

# 生成可视化图
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
