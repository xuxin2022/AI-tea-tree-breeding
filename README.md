Breeding Decision Kit - 使用说明
(English Version below)
概述
这是一个用于多年生作物分子育种的两阶段分析流程：

使用自适应LASSO回归筛选关键SNP标记

从SNP标记中提取可解释的决策规则

系统要求
操作系统：Ubuntu 18.04+

Python 3.8+

内存：8GB以上

存储：1GB可用空间

安装步骤
下载代码：
git clone https://github.com/xuxin2022/AI-tea-tree-breeding.git
cd AI-tea-tree-breeding

安装依赖：
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm openpyxl

数据准备
需要准备两个Excel文件：

train.xlsx - 训练数据
格式：

第一列：SNP标记名称

第一行：样本ID和表型标签

数据：基因型（AA、AB、BB等）

必须包含名为"label_column_name"的表型列，用"1"和"0"表示不同表型

verification.xlsx - 验证数据
格式与训练数据相同

示例：
Sample1 Sample2 ... label_column_name
SNP1 AA AB ... 1
SNP2 BB AA ... 0

配置修改
打开代码文件，找到CONFIG部分，修改以下参数：
label_column: 'label_column_name' # 改为你的表型列名
train_data_path: 'train.xlsx' # 训练数据路径
val_data_path: 'verification.xlsx' # 验证数据路径

运行程序
直接运行：
python BDK.py

输出结果
运行后在lasso_rf_results目录生成：

lasso_selection_summary.xlsx - 特征选择结果

processed_snp_subset.xlsx - 筛选的SNP数据

rule_extraction_results/extracted_rules.xlsx - 提取的决策规则

结果解读
规则示例：
如果 SNP001 是 AA 且 SNP002 是 AB 那么 预测：1 (置信度: 92%)

核心规则：置信度≥80%

补充规则：置信度≥60%

应用
对新样本检测核心SNP（通常2-4个）

应用提取的规则进行选择

成本：约1-2美元/样本

常见问题
如果报错"Label column not found"，检查Excel文件中表型列名是否匹配

如果内存不足，减少rf_n_estimators参数

确保训练和验证数据来自同一母系群体

联系我们
问题反馈：xuxin@jsscyyjs.cn

============================================

Breeding Decision Kit - User Manual
Overview
A two-stage analysis pipeline for molecular breeding in perennial crops:

Select key SNP markers using adaptive LASSO regression

Extract interpretable decision rules from SNP markers

System Requirements
OS: Ubuntu 18.04+

Python 3.8+

Memory: 8GB+

Storage: 1GB free space

Installation
Download code:
git clone https://github.com/xuxin2022/AI-tea-tree-breeding.git
cd AI-tea-tree-breeding

Install dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm openpyxl

Data Preparation
Prepare two Excel files:

train.xlsx - Training data
Format:

First column: SNP marker names

First row: Sample IDs and phenotype label

Data: Genotypes (AA, AB, BB, etc.)

Must contain phenotype column named "label_column_name" with "1" and "0"

verification.xlsx - Validation data
Same format as training data

Example:
Sample1 Sample2 ... label_column_name
SNP1 AA AB ... 1
SNP2 BB AA ... 0

Configuration
Open code file, find CONFIG section, modify:
label_column: 'label_column_name' # Your phenotype column name
train_data_path: 'train.xlsx' # Training data path
val_data_path: 'verification.xlsx' # Validation data path

Execution
Run directly:
python BDK.py

Output
Results in lasso_rf_results directory:

lasso_selection_summary.xlsx - Feature selection results

processed_snp_subset.xlsx - Filtered SNP data

rule_extraction_results/extracted_rules.xlsx - Extracted decision rules

Results Interpretation
Rule example:
IF SNP001 is AA AND SNP002 is AB THEN Predict: 1 (Confidence: 92%)

Core rules: Confidence ≥80%

Supplemental rules: Confidence ≥60%

Application
Test new samples for core SNPs (usually 2-4)

Apply extracted rules for selection

Cost: ~$1-2 per sample

Troubleshooting
If "Label column not found" error, check phenotype column name in Excel

If out of memory, reduce rf_n_estimators parameter

Ensure training and validation data from same maternal lineage

Contact
Issues: xuxin@jsscyyjs.cn
