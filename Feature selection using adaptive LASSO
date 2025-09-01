import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectFromModel 
from sklearn.linear_model import Lasso 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score 
from sklearn.ensemble import RandomForestClassifier 
import joblib 
import os  # 添加os模块用于处理路径

# Step 1: 读取数据 
file_path = 'sj.xlsx' 
data = pd.read_excel(file_path, index_col=0)

# Step 2: 数据整理 
data_transposed = data.transpose() 

# 提取标签数据 
label_col = 'label_column_name'  # 替换为实际标签列名
labels = data_transposed[label_col].values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels) 

# 移除标签列 
features = data_transposed.drop(columns=[label_col]) 

# 对每一列进行标签编码 
label_encoders = {}
for column in features.columns: 
    if features[column].dtypes == 'object':
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column].astype(str)) 
        label_encoders[column] = le 

features.columns = features.columns.astype(str) 

# Step 3: 数据预处理 
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(features) 

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed) 

# 定义λ值和存储结果的字典
lasso_alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
results = {}
best_accuracy = 0
best_alpha = None
best_features = None
best_model = None
best_selector = None  # 初始化best_selector

# 十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 确保输出目录存在
output_dir = 'lasso_results'
os.makedirs(output_dir, exist_ok=True)

for alpha in lasso_alphas:
    try:
        print(f"\n处理 λ = {alpha}")
        
        # Lasso特征选择 - 增加max_iter确保收敛
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=50000)
        selector = SelectFromModel(lasso)
        data_selected = selector.fit_transform(data_scaled, encoded_labels)
        
        # 确保模型已拟合后再访问系数
        if hasattr(selector.estimator_, 'coef_'):
            n_selected = data_selected.shape[1]
            print(f"λ = {alpha}: 原始特征 {data_scaled.shape[1]} -> 筛选后特征 {n_selected}")
            
            # 初始化模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # 存储每折的指标
            fold_accuracies = []
            fold_precisions = []
            fold_recalls = []
            
            for train_index, test_index in kf.split(data_selected):
                X_train, X_test = data_selected[train_index], data_selected[test_index]
                y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                fold_accuracies.append(accuracy_score(y_test, y_pred))
                fold_precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_recalls.append(recall_score(y_test, y_pred, average='weighted'))
            
            # 计算平均指标
            mean_accuracy = np.mean(fold_accuracies)
            mean_precision = np.mean(fold_precisions)
            mean_recall = np.mean(fold_recalls)
            
            # 存储结果
            results[alpha] = {
                'n_features': n_selected,
                'accuracy': mean_accuracy,
                'precision': mean_precision,
                'recall': mean_recall,
                'fold_accuracies': fold_accuracies
            }
            
            print(f"平均准确率: {mean_accuracy:.4f} ± {np.std(fold_accuracies):.4f}")
            
            # 更新最佳结果
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_alpha = alpha
                best_features = data_selected
                best_selector = selector
                best_model = model
                
        else:
            print(f"警告: λ = {alpha} 时Lasso模型未生成系数，跳过")
            
    except Exception as e:
        print(f"处理 λ = {alpha} 时发生错误: {str(e)}")
        continue

# 检查是否有最佳结果
if best_alpha is None:
    print("警告: 没有找到有效的λ值，请检查数据和参数设置")
    exit()

# 输出最佳结果
print("\n" + "="*50)
print(f"最佳 λ: {best_alpha}")
print(f"最佳准确率: {best_accuracy:.4f}")
print(f"筛选特征数: {best_features.shape[1]}")

# ===== 提取被选中的SNP名称 =====
try:
    # 获取特征选择掩码
    selected_features_mask = best_selector.get_support()
    # 获取被选中的SNP名称
    selected_snp_names = features.columns[selected_features_mask]
    
    print(f"\n找到 {len(selected_snp_names)} 个选中的SNP")
    
    # ===== 构建包含所有样本的选中SNP数据集 =====
    # 使用原始数据获取选中SNP的数据
    selected_snp_data = data_transposed[selected_snp_names]
    
    # 添加标签列回数据集
    selected_snp_data = pd.concat([
        selected_snp_data,
        data_transposed[label_col].rename('Label')
    ], axis=1)
    
    # 保存选中的SNP数据到Excel
    output_path = os.path.join(output_dir, 'selected_snp_data.xlsx')
    selected_snp_data.to_excel(output_path)
    print(f"已保存选中的SNP数据到 {output_path}")
    
    # ===== 保存选中SNP的详细信息 =====
    snp_info_df = pd.DataFrame({
        'SNP_Name': selected_snp_names,
        'Lasso_Coefficient': best_selector.estimator_.coef_[selected_features_mask]
    })
    
    # 添加在原始数据中的位置信息
    snp_info_df['Original_Column_Index'] = np.where(selected_features_mask)[0]
    info_path = os.path.join(output_dir, 'selected_snp_info.xlsx')
    snp_info_df.to_excel(info_path, index=False)
    print(f"已保存SNP详细信息到 {info_path}")
    
except Exception as e:
    print(f"提取SNP数据时发生错误: {str(e)}")

# 重新训练最佳模型（使用全数据集）
try:
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(best_features, encoded_labels)
    
    # Step 5: 保存完整pipeline 
    joblib.dump(imputer, 'simple_imputer.pkl') 
    joblib.dump(scaler, 'standard_scaler.pkl') 
    joblib.dump(best_selector, 'best_lasso_selector.pkl') 
    joblib.dump(label_encoders, 'label_encoders.pkl') 
    joblib.dump(label_encoder, 'label_encoder_for_labels.pkl') 
    joblib.dump(final_model, 'best_random_forest_model.pkl') 
    print("已保存模型和预处理对象")
except Exception as e:
    print(f"保存模型时发生错误: {str(e)}")

# 保存结果报告
try:
    result_df = pd.DataFrame.from_dict(results, orient='index')
    result_df.index.name = 'Lambda'
    result_df.reset_index(inplace=True)
    result_path = os.path.join(output_dir, 'feature_selection_results.xlsx')
    result_df.to_excel(result_path, index=False)
    print(f"结果已保存到 {result_path}")
    
    # 输出所有λ的结果
    print("\n所有λ值测试结果:")
    print(result_df[['Lambda', 'n_features', 'accuracy']])
except Exception as e:
    print(f"保存结果报告时发生错误: {str(e)}")
