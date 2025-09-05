from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import os
import joblib

# Step 1: Read data
file_path = 'sj.xlsx' 
data = pd.read_excel(file_path, index_col=0)

# Step 2: Data Preprocessing
# Transpose data for consistent feature structure
data_transposed = data.transpose()

# Extract labels and encode
label_col = 'label_column_name'  # Replace with the actual label column name
labels = data_transposed[label_col].values
features = data_transposed.drop(columns=[label_col]) 

# Label encoding for non-numeric columns
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Label encoding for features
label_encoders = {}
for column in features.columns:
    if features[column].dtype == 'object':
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column].astype(str))
        label_encoders[column] = le

features.columns = features.columns.astype(str)

# Step 3: Create KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Results collection
results = {}
best_accuracy = 0
best_alpha = None
best_model = None
best_selector = None

# Regularization parameter for Lasso
lasso_alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]

# Loop through different Lasso alphas
for alpha in lasso_alphas:
    try:
        print(f"\nProcessing λ = {alpha}")

        # Define the pipeline for preprocessing and modeling
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
            ('scaler', StandardScaler()),  # Standardize features
            ('lasso', Lasso(alpha=alpha, max_iter=50000, random_state=42)),  # Lasso feature selection
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest Classifier
        ])

        # Apply cross-validation
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []

        for train_index, test_index in kf.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

            # Fit pipeline and evaluate
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Calculate performance metrics
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            fold_precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            fold_recalls.append(recall_score(y_test, y_pred, average='weighted'))

        # Store the mean results
        mean_accuracy = np.mean(fold_accuracies)
        mean_precision = np.mean(fold_precisions)
        mean_recall = np.mean(fold_recalls)

        # Save results
        results[alpha] = {
            'accuracy': mean_accuracy,
            'precision': mean_precision,
            'recall': mean_recall
        }

        print(f"Average accuracy: {mean_accuracy:.4f} ± {np.std(fold_accuracies):.4f}")

        # Update best model
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_alpha = alpha
            best_model = pipeline

    except Exception as e:
        print(f"Error with λ = {alpha}: {str(e)}")
        continue

# Save the best model to file
if best_model:
    output_dir = 'lasso_results'
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    print(f"Best model saved to {output_dir}/best_model.pkl")
else:
    print("No valid model found.")
