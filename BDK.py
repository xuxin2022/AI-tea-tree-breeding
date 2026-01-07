"""
Democratizing Molecular Breeding in Perennial Crops: A Maternal-Lineage Paradigm for Interpretable, Low-Cost Decision Kits
Author: Xin Xu
Institution: Jiangsu Provincial Tea Research Institute
Nature Plants Submission
"""

# ============================ IMPORT LIBRARIES ============================
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from tqdm import tqdm
import sys

# ============================ GLOBAL CONFIGURATION ============================
CONFIG = {
    # Training parameters
    'rf_cv_folds': 10,
    'rf_cv_repeats': 5,
    'rf_n_estimators': 100,
    'rf_max_depth': 15,
    'rf_min_samples_split': 5,
    'rf_min_samples_leaf': 2,
    'rf_max_features': 'sqrt',
    'rf_random_state': 42,
    'rf_class_weight': 'balanced',
    'rf_max_samples': 0.8,
    
    # Data processing
    'missing_value_markers': ['NA', '缺失', '', 'N/A', 'nan', 'NaN', np.nan, 'None'],
    'missing_class': 'MISSING',
    'label_column': 'label_column_name',  # MODIFY THIS: Specify your label column name
    
    # Lasso feature selection
    'lasso_alphas': [0.00001, 0.00005, 0.0001, 0.001, 0.005, 0.01],
    'lasso_max_iter': 50000,
    'lasso_random_state': 42,
    
    # File paths
    'train_data_path': 'train.xlsx',
    'val_data_path': 'verification.xlsx',
    'output_dir': 'lasso_rf_results',
    'model_prefix': 'rf_model',
    'train_report': 'training_report.txt',
    'val_report': 'validation_report.txt',
    'val_cm_fig': 'confusion_matrix_validation.png',
    'val_prediction_results': 'validation_predictions.xlsx',
    'lasso_selection_summary': 'lasso_selection_summary.xlsx',
    'snp_subset_file': 'processed_snp_subset.xlsx',
    
    # Rule extraction parameters
    'rule_extraction_dir': 'rule_extraction_results',
    'core_rule_min_samples': 3,
    'core_rule_min_confidence': 0.80,
    'core_rule_max_num': 4,
    'supplement_rule_min_samples': 2,
    'supplement_rule_min_confidence': 0.60,
    'supplement_rule_max_num': 6,
    'core_tree_params': {
        'max_depth': 3,
        'min_samples_split': 8,
        'min_samples_leaf': 6,
        'min_impurity_decrease': 0.01,
        'max_features': None,
        'random_state': 42,
        'ccp_alpha': 0.01,
        'splitter': 'best',
        'class_weight': 'balanced',
    },
    'supplement_tree_params': {
        'max_depth': 6,
        'min_samples_split': 6,
        'min_impurity_decrease': 0.005,
        'max_features': None,
        'random_state': 42,
        'ccp_alpha': 0.005,
        'splitter': 'best',
        'class_weight': 'balanced',
    },
    'excluded_features': ['ID', 'id', 'sample_id', 'sample_name'],
    'max_conditions_per_rule': 3,
    'min_rule_quality_confidence': 0.8,
    'min_rule_quality_samples': 5,
    'max_rule_similarity': 0.8,
}

# ============================ VISUAL PROGRESS BARS ============================
class ProgressBar:
    """Custom progress bar for algorithm execution"""
    
    @staticmethod
    def initialize(total_steps, description="Processing"):
        """Initialize progress bar"""
        sys.stdout.write(f"\n{description}\n")
        sys.stdout.write("[" + " " * 50 + "] 0%")
        sys.stdout.flush()
        
    @staticmethod
    def update(step, total_steps, message=""):
        """Update progress bar"""
        percentage = int((step / total_steps) * 100)
        filled_length = int(50 * step // total_steps)
        bar = "█" * filled_length + " " * (50 - filled_length)
        
        sys.stdout.write(f"\r[{bar}] {percentage}% {message}")
        sys.stdout.flush()
        
        if step == total_steps:
            sys.stdout.write("\n✅ Complete!\n")

# ============================ DATA PROCESSING MODULE ============================
class DataProcessor:
    """Handles data loading, preprocessing, and transformation"""
    
    @staticmethod
    def load_and_prepare_data(file_path, label_column):
        """
        Load genetic data from Excel file and prepare for analysis
        
        Parameters:
        -----------
        file_path : str
            Path to Excel file containing genetic data
        label_column : str
            Name of the column containing class labels
            
        Returns:
        --------
        tuple : (features, labels)
        """
        try:
            # Load and transpose data (samples as rows, SNPs as columns)
            data = pd.read_excel(file_path, index_col=0)
            data_transposed = data.transpose()
            
            # Validate label column exists
            if label_column not in data_transposed.columns:
                raise ValueError(f"Label column '{label_column}' not found in data")
            
            # Separate features and labels
            labels = data_transposed[label_column].values
            features = data_transposed.drop(columns=[label_column])
            
            return features, labels
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    @staticmethod
    def handle_categorical_missing_values(df):
        """Replace missing values in categorical features with MISSING class"""
        df_processed = df.copy()
        for col in df_processed.columns:
            df_processed[col] = df_processed[col].replace(CONFIG['missing_value_markers'], CONFIG['missing_class'])
            df_processed[col] = df_processed[col].astype(str)
        return df_processed

# ============================ LASSO FEATURE SELECTION MODULE ============================
class LassoFeatureSelector:
    """
    Implements Lasso-based feature selection with cross-validation
    
    This module selects informative SNP markers while controlling for
    multicollinearity and overfitting through regularization
    """
    
    def __init__(self, config):
        """Initialize selector with configuration parameters"""
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_encoders = {}
        self.train_feature_modes = {}
    
    def _preprocess_features(self, features, fit=True):
        """
        Preprocess categorical SNP features for Lasso analysis
        
        Parameters:
        -----------
        features : DataFrame
            Raw SNP data
        fit : bool
            Whether to fit new encoders or use existing ones
            
        Returns:
        --------
        DataFrame : Encoded features
        """
        # Handle missing values
        features_processed = DataProcessor.handle_categorical_missing_values(features)
        
        # Encode categorical features
        encoded_features = pd.DataFrame()
        
        if fit:
            # Fit new encoders for training data
            for column in tqdm(features_processed.columns, desc="Encoding features", leave=False):
                le = LabelEncoder()
                encoded_features[column] = le.fit_transform(features_processed[column])
                self.feature_encoders[column] = le
                # Calculate mode for each feature
                self.train_feature_modes[column] = encoded_features[column].mode()[0]
        else:
            # Transform validation data using existing encoders
            for column in tqdm(features_processed.columns, desc="Encoding validation features", leave=False):
                if column in self.feature_encoders:
                    le = self.feature_encoders[column]
                    try:
                        # Try to use existing encoder
                        encoded_features[column] = le.transform(features_processed[column])
                    except ValueError:
                        # Handle unknown categories: use most common category
                        most_common = le.classes_[0]
                        transformed_values = []
                        for val in features_processed[column]:
                            if val in le.classes_:
                                transformed_values.append(le.transform([val])[0])
                            else:
                                transformed_values.append(le.transform([most_common])[0])
                        encoded_features[column] = transformed_values
                else:
                    # If feature is new in validation set, fit a new encoder
                    le = LabelEncoder()
                    encoded_features[column] = le.fit_transform(features_processed[column])
                    self.feature_encoders[column] = le
        
        encoded_features.columns = encoded_features.columns.astype(str)
        return encoded_features
    
    def _align_features(self, train_features, val_features, train_feature_names):
        """
        Align features between training and validation sets
        
        Parameters:
        -----------
        train_features : DataFrame
            Training set features
        val_features : DataFrame
            Validation set features
        train_feature_names : list
            Names of training features
            
        Returns:
        --------
        tuple : (aligned_train_features, aligned_val_features)
        """
        train_features_aligned = train_features.copy()
        val_features_aligned = pd.DataFrame()
        
        for col in train_feature_names:
            if col in train_features_aligned.columns:
                # Training feature
                train_feature = train_features_aligned[col]
                
                # Validation feature
                if col in val_features.columns:
                    val_features_aligned[col] = val_features[col]
                else:
                    # If validation set lacks this feature, fill with training mode
                    mode_value = self.train_feature_modes.get(col, train_feature.mode()[0])
                    val_features_aligned[col] = [mode_value] * len(val_features)
        
        return train_features_aligned, val_features_aligned
    
    def select_features(self, train_data_path, val_data_path):
        """
        Perform Lasso feature selection with validation set evaluation
        
        Returns:
        --------
        tuple : (selected_features, results)
        """
        print("\n" + "="*80)
        print("🎯 LASSO FEATURE SELECTION")
        print("="*80)
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize progress
        total_steps = len(self.config['lasso_alphas']) + 6
        current_step = 0
        ProgressBar.initialize(total_steps, "Feature Selection Progress")
        
        # Step 1: Load data
        ProgressBar.update(current_step, total_steps, "Loading training data")
        current_step += 1
        train_features, train_labels = DataProcessor.load_and_prepare_data(
            train_data_path, self.config['label_column']
        )
        
        ProgressBar.update(current_step, total_steps, "Loading validation data")
        current_step += 1
        val_features, val_labels = DataProcessor.load_and_prepare_data(
            val_data_path, self.config['label_column']
        )
        
        original_feature_names = train_features.columns.tolist()
        
        # Step 2: Preprocess training data
        ProgressBar.update(current_step, total_steps, "Preprocessing training data")
        current_step += 1
        train_features_encoded = self._preprocess_features(train_features, fit=True)
        
        # Encode training labels
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)
        
        # Standardize training features
        train_features_scaled = self.scaler.fit_transform(train_features_encoded)
        
        # Step 3: Preprocess validation data
        ProgressBar.update(current_step, total_steps, "Preprocessing validation data")
        current_step += 1
        val_features_encoded = self._preprocess_features(val_features, fit=False)
        
        # Encode validation labels (handle unknown labels)
        try:
            val_labels_encoded = self.label_encoder.transform(val_labels)
        except ValueError as e:
            # Replace unknown labels with most common training label
            most_common_label = self.label_encoder.classes_[
                np.argmax(pd.Series(train_labels_encoded).value_counts())
            ]
            val_labels_clean = []
            for lbl in val_labels:
                if lbl in self.label_encoder.classes_:
                    val_labels_clean.append(lbl)
                else:
                    val_labels_clean.append(most_common_label)
            val_labels_encoded = self.label_encoder.transform(val_labels_clean)
        
        # Step 4: Align features
        ProgressBar.update(current_step, total_steps, "Aligning features")
        current_step += 1
        train_features_aligned, val_features_aligned = self._align_features(
            pd.DataFrame(train_features_scaled, columns=train_features_encoded.columns),
            pd.DataFrame(val_features_encoded, columns=val_features_encoded.columns),
            train_features_encoded.columns.tolist()
        )
        
        # Standardize validation features
        val_features_scaled = self.scaler.transform(val_features_aligned)
        
        # Step 5: Lasso feature selection
        results = {}
        
        for alpha in tqdm(self.config['lasso_alphas'], desc="Lasso parameter evaluation"):
            try:
                # Fit Lasso model
                lasso = Lasso(
                    alpha=alpha,
                    random_state=self.config['lasso_random_state'],
                    max_iter=self.config['lasso_max_iter']
                )
                
                selector = SelectFromModel(lasso)
                train_selected = selector.fit_transform(train_features_aligned, train_labels_encoded)
                
                if hasattr(selector.estimator_, 'coef_'):
                    n_selected = train_selected.shape[1]
                    if n_selected == 0:
                        continue
                    
                    # Train Random Forest on selected features (5-fold CV)
                    model_train = RandomForestClassifier(
                        n_estimators=self.config['rf_n_estimators'],
                        max_depth=self.config['rf_max_depth'],
                        min_samples_split=self.config['rf_min_samples_split'],
                        min_samples_leaf=self.config['rf_min_samples_leaf'],
                        max_features=self.config['rf_max_features'],
                        random_state=self.config['rf_random_state'],
                        class_weight=self.config['rf_class_weight'],
                        bootstrap=True,
                        max_samples=self.config['rf_max_samples'],
                        n_jobs=-1
                    )
                    
                    # 5-fold cross-validation
                    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['rf_random_state'])
                    train_cv_scores = cross_val_score(
                        model_train, train_selected, train_labels_encoded,
                        cv=kf, scoring='accuracy', n_jobs=-1
                    )
                    train_mean_accuracy = np.mean(train_cv_scores)
                    train_std_accuracy = np.std(train_cv_scores)
                    
                    # Evaluate on validation set
                    selected_mask = selector.get_support()
                    selected_feature_indices = [i for i, val in enumerate(selected_mask) if val]
                    
                    val_selected = val_features_scaled[:, selected_feature_indices]
                    
                    model_val = RandomForestClassifier(
                        n_estimators=self.config['rf_n_estimators'],
                        max_depth=self.config['rf_max_depth'],
                        min_samples_split=self.config['rf_min_samples_split'],
                        min_samples_leaf=self.config['rf_min_samples_leaf'],
                        max_features=self.config['rf_max_features'],
                        random_state=self.config['rf_random_state'],
                        class_weight=self.config['rf_class_weight'],
                        bootstrap=True,
                        max_samples=self.config['rf_max_samples'],
                        n_jobs=-1
                    )
                    model_val.fit(train_selected, train_labels_encoded)
                    
                    val_pred = model_val.predict(val_selected)
                    val_accuracy = accuracy_score(val_labels_encoded, val_pred)
                    
                    # Store results
                    results[alpha] = {
                        'n_features': n_selected,
                        'train_accuracy': train_mean_accuracy,
                        'train_std': train_std_accuracy,
                        'val_accuracy': val_accuracy,
                        'selector': selector,
                        'selected_indices': selected_feature_indices,
                        'train_cv_scores': train_cv_scores
                    }
                    
                else:
                    continue
                    
            except Exception as e:
                continue
        
        ProgressBar.update(current_step, total_steps, "Analyzing results")
        current_step += 1
        
        # Step 6: Select best alpha value
        if len(results) == 0:
            selected_feature_names = original_feature_names
            best_selector = None
            best_alpha = None
        else:
            # Create result list for sorting
            result_list = []
            for alpha, result in results.items():
                result_list.append({
                    'alpha': alpha,
                    'val_accuracy': result['val_accuracy'],
                    'train_accuracy': result['train_accuracy'],
                    'n_features': result['n_features'],
                    'result': result
                })
            
            # Sort by priority: 1. validation accuracy 2. training accuracy 3. number of features
            result_list.sort(key=lambda x: (-x['val_accuracy'], -x['train_accuracy'], x['n_features']))
            
            # Select best result
            best_result = result_list[0]
            best_alpha = best_result['alpha']
            best_selector = best_result['result']['selector']
            
            # Get selected features
            selected_mask = best_selector.get_support()
            selected_feature_names = [original_feature_names[i] for i, val in enumerate(selected_mask) if val]
        
        # Step 7: Save Lasso selection results
        if results:
            result_summary = []
            for alpha, result in results.items():
                result_summary.append({
                    'Alpha': alpha,
                    'Selected_Features': result['n_features'],
                    'Train_CV_Accuracy': result['train_accuracy'],
                    'Train_CV_Std': result['train_std'],
                    'Validation_Accuracy': result['val_accuracy']
                })
            
            result_df = pd.DataFrame(result_summary)
            
            # Sort by priority
            result_df = result_df.sort_values(
                ['Validation_Accuracy', 'Train_CV_Accuracy', 'Selected_Features'],
                ascending=[False, False, True]
            )
            
            # Save to Excel
            summary_path = os.path.join(self.config['output_dir'], self.config['lasso_selection_summary'])
            result_df.to_excel(summary_path, index=False)
        
        ProgressBar.update(total_steps, total_steps)
        
        # Save preprocessing objects
        preprocess_objects = {
            'scaler': self.scaler,
            'feature_encoders': self.feature_encoders,
            'label_encoder': self.label_encoder,
            'best_selector': best_selector
        }
        
        for name, obj in preprocess_objects.items():
            if obj is not None:
                joblib.dump(obj, os.path.join(self.config['output_dir'], f"{name}.pkl"))
        
        return selected_feature_names, preprocess_objects, results

# ============================ RULE EXTRACTION MODULE ============================
class RuleExtractor:
    """
    Extracts interpretable classification rules from decision trees
    
    This module identifies key genetic markers and their combinations
    that predict phenotypic traits in plants
    """
    
    def __init__(self, config):
        """Initialize rule extractor with configuration"""
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Configure logging for rule extraction process"""
        log_dir = os.path.join(self.config['output_dir'], self.config['rule_extraction_dir'])
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger('RuleExtractor')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(
            os.path.join(log_dir, f'rule_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        )
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)  # Changed from INFO to WARNING to reduce console output
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    @staticmethod
    def get_original_base_condition(feature_name, threshold, ordinal_encoders, feature_classes, is_right=False):
        """Convert encoded values to original conditions"""
        feature_name = str(feature_name)
        if feature_name not in ordinal_encoders:
            return f"{feature_name}[encoding_error]"
        
        original_bases = feature_classes[feature_name]
        encoded_values = np.arange(len(original_bases))
        
        if not is_right:
            included_encoded = encoded_values <= threshold
        else:
            included_encoded = encoded_values > threshold
        
        if not np.any(included_encoded):
            return f"{feature_name}[no_matching_category]"
        
        included_categories = original_bases[included_encoded]
        display_categories = []
        for cat in included_categories:
            if cat in ['nan', np.nan, 'missing'] or str(cat).upper() == 'NAN':
                display_categories.append('NAN')
            else:
                display_categories.append(str(cat).upper())
        
        if -1 in encoded_values:
            display_categories.append('unknown_category')
        
        return f"{feature_name} is {', '.join(display_categories)}"
    
    @staticmethod
    def simplify_rule_conditions(rule_conditions, config):
        """Simplify rule conditions"""
        if not rule_conditions or rule_conditions.strip() == "":
            return ""
        
        conditions = [cond.strip() for cond in rule_conditions.split(" AND ")]
        unique_conditions = list(dict.fromkeys(conditions))
        
        valid_conditions = []
        for cond in unique_conditions:
            if "[" in cond or "encoding_error" in cond or "no_matching_category" in cond or "unknown_category" in cond or cond.strip() == "":
                continue
            valid_conditions.append(cond)
        
        if len(valid_conditions) > config['max_conditions_per_rule']:
            valid_conditions = valid_conditions[:config['max_conditions_per_rule']]
        
        feature_cond_map = {}
        for cond in valid_conditions:
            if " is " in cond:
                feature, value = cond.split(" is ", 1)
                feature = feature.strip()
                value = value.strip()
                if feature not in feature_cond_map:
                    feature_cond_map[feature] = []
                if value not in feature_cond_map[feature]:
                    feature_cond_map[feature].append(value)
        
        simplified_conditions = []
        for feature, values in feature_cond_map.items():
            if len(values) == 1:
                simplified_conditions.append(f"{feature} is {values[0]}")
            else:
                if len(values) > 3:
                    values = values[:3]
                simplified_conditions.append(f"{feature} is {', '.join(values)}")
        
        return " AND ".join(simplified_conditions)
    
    def extract_from_tree(self, tree_model, feature_names, X_data, y_true, ordinal_encoders, feature_classes, rule_type='core'):
        """Extract rules from decision tree"""
        tree_ = tree_model.tree_
        feature_names = [str(name) for name in feature_names]
        node_sample_indices = {}
        rules = []

        min_samples = self.config[f'{rule_type}_rule_min_samples']
        
        def get_node_samples(node_id, indices):
            node_sample_indices[node_id] = indices.copy()
            if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
                feature_idx = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]
                X_subset = X_data[indices]
                left_mask = X_subset[:, feature_idx] <= threshold
                right_mask = ~left_mask
                get_node_samples(tree_.children_left[node_id], indices[left_mask])
                get_node_samples(tree_.children_right[node_id], indices[right_mask])
        
        get_node_samples(0, np.arange(len(X_data)))

        def extract_node_rules(node_id, current_rule):
            if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
                feature_idx = tree_.feature[node_id]
                if feature_idx >= len(feature_names):
                    return
                feature_name = feature_names[feature_idx]
                threshold = tree_.threshold[node_id]
                
                left_condition = self.get_original_base_condition(
                    feature_name, threshold, ordinal_encoders, feature_classes, is_right=False
                )
                extract_node_rules(tree_.children_left[node_id], current_rule + [left_condition])
                
                right_condition = self.get_original_base_condition(
                    feature_name, threshold, ordinal_encoders, feature_classes, is_right=True
                )
                extract_node_rules(tree_.children_right[node_id], current_rule + [right_condition])
            else:
                # Get original sample indices and count
                node_indices = node_sample_indices.get(node_id, np.array([]))
                n_samples = len(node_indices)
                if n_samples < min_samples:
                    return
                
                # Get true labels and calculate confidence
                if len(node_indices) == 0:
                    return
                node_y_true = y_true[node_indices]
                class_counts = np.bincount(node_y_true)
                predicted_class = np.argmax(class_counts) if len(class_counts) > 0 else 0
                
                # Calculate confidence
                correct_count = class_counts[predicted_class] if predicted_class < len(class_counts) else 0
                confidence = correct_count / n_samples if n_samples > 0 else 0
                confidence = np.clip(confidence, 0, 1)
                
                # Filter low confidence rules
                min_confidence = self.config[f'{rule_type}_rule_min_confidence']
                if confidence < min_confidence:
                    return
                
                raw_conditions = " AND ".join(current_rule)
                simplified_conditions = self.simplify_rule_conditions(raw_conditions, self.config)
                
                if not simplified_conditions:
                    return
                
                rules.append({
                    'raw_conditions': raw_conditions,
                    'simplified_conditions': simplified_conditions,
                    'n_samples': n_samples,
                    'confidence': confidence,
                    'predicted_class': predicted_class,
                    'class_distribution': class_counts,
                    'coverage_percentage': n_samples / len(X_data) * 100
                })
        
        extract_node_rules(0, [])
        return rules
    
    @staticmethod
    def get_rule_coverage_mask(X_raw, rule_conditions):
        """Get mask for samples satisfying rule conditions"""
        mask = np.ones(len(X_raw), dtype=bool)
        if not rule_conditions or rule_conditions.strip() == "" or rule_conditions == "all remaining samples":
            return mask
        
        conditions = rule_conditions.split(" AND ")
        for cond in conditions:
            if " is " not in cond:
                continue
            feature, value = cond.split(" is ", 1)
            feature = feature.strip()
            values = [v.strip() for v in value.split(",")]
            
            if feature not in X_raw.columns:
                mask = np.zeros(len(X_raw), dtype=bool)
                break
            
            feature_values = X_raw[feature].astype(str).str.upper()
            cond_mask = feature_values.isin(values)
            mask = mask & cond_mask
        
        return mask
    
    @staticmethod
    def remove_duplicate_rules(rules):
        """Remove duplicate rules"""
        if not rules:
            return []
        
        unique_rules = {}
        for rule in rules:
            key = rule['simplified_conditions']
            if key not in unique_rules:
                unique_rules[key] = rule
            else:
                existing_rule = unique_rules[key]
                existing_quality = existing_rule['confidence'] * 0.6 + (existing_rule['n_samples'] / 100) * 0.4
                new_quality = rule['confidence'] * 0.6 + (rule['n_samples'] / 100) * 0.4
                if new_quality > existing_quality:
                    unique_rules[key] = rule
        
        return list(unique_rules.values())
    
    @staticmethod
    def filter_low_quality_rules(rules, config, min_confidence=None, min_samples=None):
        """Filter low quality rules"""
        if not rules:
            return []
        
        min_conf = min_confidence or config['min_rule_quality_confidence']
        min_samp = min_samples or config['min_rule_quality_samples']
        
        filtered_rules = []
        for rule in rules:
            if rule['confidence'] < min_conf:
                continue
            if rule['n_samples'] < min_samp:
                continue
            if not rule['simplified_conditions'] or rule['simplified_conditions'].strip() == "":
                continue
            filtered_rules.append(rule)
        
        return filtered_rules
    
    @staticmethod
    def remove_similar_rules(rules, config):
        """Remove highly similar rules"""
        if not rules or len(rules) <= 1:
            return rules
        
        threshold = config['max_rule_similarity']
        rules.sort(key=lambda x: x['confidence'] * x['n_samples'], reverse=True)
        
        unique_rules = []
        for i, rule in enumerate(rules):
            is_similar = False
            for existing_rule in unique_rules:
                similarity = len(set(rule['simplified_conditions'].split(" AND ")) & 
                                 set(existing_rule['simplified_conditions'].split(" AND "))) / \
                            len(set(rule['simplified_conditions'].split(" AND ")) | 
                                set(existing_rule['simplified_conditions'].split(" AND "))) \
                            if (rule['simplified_conditions'] and existing_rule['simplified_conditions']) else 0
                if similarity > threshold:
                    is_similar = True
                    break
            if not is_similar:
                unique_rules.append(rule)
        
        return unique_rules
    
    def train_core_rules(self, X_data, y_true, feature_names, ordinal_encoders, feature_classes):
        """Train and extract core rules"""
        self.logger.info("\n【Extracting Core Rules】")
        
        # Filter features
        target_features = [f for f in feature_names if not any(
            excl in f.lower() for excl in self.config['excluded_features'])]
        
        if not target_features:
            self.logger.error("No available features (ID-type features excluded)")
            return None
        
        feature_indices = [feature_names.index(f) for f in target_features if f in feature_names]
        X_target = X_data[:, feature_indices]
        
        # Train decision tree
        tree_model = DecisionTreeClassifier(**self.config['core_tree_params'])
        tree_model.fit(X_target, y_true)
        
        # Extract rules
        all_rules = self.extract_from_tree(
            tree_model, target_features, X_target, y_true,
            ordinal_encoders, feature_classes, rule_type='core'
        )
        
        # Rule filtering
        all_rules = self.remove_duplicate_rules(all_rules)
        all_rules = self.filter_low_quality_rules(all_rules, self.config)
        all_rules = self.remove_similar_rules(all_rules, self.config)
        
        # Sort and select best rules
        all_rules.sort(key=lambda x: x['confidence'] * x['n_samples'], reverse=True)
        core_rules = all_rules[:self.config['core_rule_max_num']]
        
        self.logger.info(f"Selected core rules: {len(core_rules)}")
        for i, rule in enumerate(core_rules):
            class_label = "Class 1" if rule['predicted_class'] == 1 else "Class 0"
            self.logger.info(f"Rule {i+1} ({class_label}): {rule['simplified_conditions']}")
        
        return core_rules
    
    def train_supplement_rules(self, core_rules, X_raw, X_encoded, y_encoded, 
                               feature_names, ordinal_encoders, feature_classes):
        """Extract supplemental rules"""
        self.logger.info("\n【Extracting Supplemental Rules】")
        
        # Calculate samples not covered by core rules
        core_masks = []
        for rule in core_rules:
            mask = self.get_rule_coverage_mask(X_raw, rule['simplified_conditions'])
            core_masks.append(mask)
        
        total_core_mask = np.any(np.array(core_masks), axis=0) if core_masks else np.zeros(len(X_raw), dtype=bool)
        exception_indices = np.where(~total_core_mask)[0]
        
        if len(exception_indices) == 0:
            self.logger.info("No remaining samples for supplemental rules")
            return []
        
        # Filter features
        target_features = [f for f in feature_names if not any(
            excl in f.lower() for excl in self.config['excluded_features'])]
        feature_indices = [feature_names.index(f) for f in target_features if f in feature_names]
        
        X_exception = X_encoded[exception_indices][:, feature_indices]
        
        if len(exception_indices) < 2:
            self.logger.info("Too few remaining samples, skipping supplemental rules")
            return []
        
        # Train decision tree
        tree_model = DecisionTreeClassifier(**self.config['supplement_tree_params'])
        tree_model.fit(X_exception, y_exception)
        
        # Extract rules
        supplement_rules = self.extract_from_tree(
            tree_model, target_features, X_exception, y_exception,
            ordinal_encoders, feature_classes, rule_type='supplement'
        )
        
        # Only keep class 1 rules
        supplement_rules = [r for r in supplement_rules if r['predicted_class'] == 1]
        if not supplement_rules:
            self.logger.info("No class 1 supplemental rules extracted")
            return []
        
        # Filter with relaxed conditions
        supplement_rules = self.remove_duplicate_rules(supplement_rules)
        supplement_rules = self.filter_low_quality_rules(
            supplement_rules, self.config,
            min_confidence=0.5,
            min_samples=2
        )
        supplement_rules = self.remove_similar_rules(supplement_rules, self.config)
        
        # Sort and select best rules
        supplement_rules.sort(key=lambda x: x['confidence'] * x['n_samples'], reverse=True)
        selected_rules = supplement_rules[:self.config['supplement_rule_max_num']]
        
        self.logger.info(f"Selected supplemental rules: {len(selected_rules)}")
        for i, rule in enumerate(selected_rules):
            class_label = "Class 1" if rule['predicted_class'] == 1 else "Class 0"
            self.logger.info(f"Supplemental rule {i+1} ({class_label}): {rule['simplified_conditions']}")
        
        return selected_rules
    
    def save_snp_subset(self, train_data_path, selected_features):
        """Save SNP subset for rule extraction"""
        print("\n" + "="*80)
        print("💾 Saving SNP Subset for Rule Extraction")
        print("="*80)
        
        # Load original training data
        data = pd.read_excel(train_data_path, index_col=0)
        data_transposed = data.transpose()
        
        # Create filtered dataset
        selected_features_with_label = [self.config['label_column']] + selected_features
        available_features = [f for f in selected_features_with_label if f in data_transposed.columns]
        
        # Create subset
        snp_subset = data_transposed[available_features].copy()
        snp_subset.reset_index(inplace=True)
        
        # Rename index column
        if snp_subset.columns[0] == 'index':
            snp_subset.rename(columns={'index': 'sample_name'}, inplace=True)
        
        # Save to Excel
        subset_path = os.path.join(self.config['output_dir'], self.config['snp_subset_file'])
        snp_subset.to_excel(subset_path, index=False)
        
        return subset_path
    
    def run_extraction(self, snp_subset_path, label_column):
        """Run rule extraction pipeline"""
        print("\n" + "="*80)
        print("🎯 Starting Rule Extraction Pipeline")
        print("="*80)
        
        # Create output directory
        rule_output_dir = os.path.join(self.config['output_dir'], self.config['rule_extraction_dir'])
        os.makedirs(rule_output_dir, exist_ok=True)
        
        # Initialize progress
        total_steps = 7
        current_step = 0
        ProgressBar.initialize(total_steps, "Rule Extraction Progress")
        
        try:
            # Step 1: Load data
            ProgressBar.update(current_step, total_steps, "Loading SNP subset")
            current_step += 1
            train_data = pd.read_excel(snp_subset_path)
            
            if train_data.empty:
                self.logger.error("Input data is empty")
                return False
            
            # Check for sample_name column
            if 'sample_name' in train_data.columns:
                train_data = train_data.drop(columns=['sample_name'])
            
            # Check for ID columns
            id_cols = [col for col in train_data.columns if col != label_column and 
                      any(excl in col.lower() for excl in ['id', 'sample', 'name'])]
            if id_cols:
                train_data = train_data.drop(columns=id_cols)
            
            # Check label column
            if label_column not in train_data.columns:
                self.logger.error(f"Label column {label_column} does not exist")
                return False
            
            # Step 2: Prepare features and labels
            ProgressBar.update(current_step, total_steps, "Preparing features and labels")
            current_step += 1
            X_train_raw = train_data.drop(columns=[label_column])
            y_train_raw = train_data[label_column].values
            
            # Encode labels
            unique_labels = np.unique(y_train_raw)
            if len(unique_labels) > 2:
                y_train_encoded = np.where(y_train_raw == unique_labels[0], 0, 1)
            else:
                label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                y_train_encoded = np.vectorize(label_mapping.get)(y_train_raw)
            
            # Step 3: Preprocess features
            ProgressBar.update(current_step, total_steps, "Preprocessing features")
            current_step += 1
            X_train_raw = X_train_raw.replace(self.config['missing_value_markers'], 'NAN')
            X_train_raw = X_train_raw.apply(lambda col: col.astype(str).str.upper())
            X_train_raw.columns = [str(col) for col in X_train_raw.columns]
            
            # Feature encoding
            ordinal_encoders = {}
            feature_classes = {}
            X_train_encoded_df = X_train_raw.copy()
            
            for col in tqdm(X_train_raw.columns, desc="Encoding features", leave=False):
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_train_encoded_df[col] = oe.fit_transform(X_train_raw[[col]]).flatten()
                ordinal_encoders[col] = oe
                feature_classes[col] = oe.categories_[0]
            
            X_train_encoded = X_train_encoded_df.values
            feature_names = X_train_raw.columns.tolist()
            
            # Step 4: Extract core rules
            ProgressBar.update(current_step, total_steps, "Extracting core rules")
            current_step += 1
            core_rules = self.train_core_rules(
                X_train_encoded, y_train_encoded, feature_names,
                ordinal_encoders, feature_classes
            )
            
            # Step 5: Extract supplemental rules
            ProgressBar.update(current_step, total_steps, "Extracting supplemental rules")
            current_step += 1
            supplement_rules = []
            if core_rules:
                supplement_rules = self.train_supplement_rules(
                    core_rules, X_train_raw, X_train_encoded, y_train_encoded,
                    feature_names, ordinal_encoders, feature_classes
                )
            
            # Step 6: Save results
            ProgressBar.update(current_step, total_steps, "Saving extracted rules")
            current_step += 1
            
            # Combine rules
            all_rules = []
            if core_rules:
                all_rules.extend(core_rules)
            if supplement_rules:
                all_rules.extend(supplement_rules)
            
            if all_rules:
                rules_df = pd.DataFrame([
                    {
                        'Rule Type': 'Core Rule' if i < len(core_rules) else 'Supplement Rule',
                        'Rule ID': f"Rule_{i+1}",
                        'Rule Conditions': rule['simplified_conditions'],
                        'Predicted Class': rule['predicted_class'],
                        'Predicted Class Label': unique_labels[1] if rule['predicted_class'] == 1 else unique_labels[0],
                        'Confidence (Accuracy)': f"{rule['confidence']:.3f}",
                        'Covered Samples': rule['n_samples'],
                        'Coverage (%)': f"{rule['coverage_percentage']:.1f}%",
                        'Class Distribution': str(rule['class_distribution'])
                    }
                    for i, rule in enumerate(all_rules)
                ])
                
                rules_path = os.path.join(rule_output_dir, "extracted_rules.xlsx")
                rules_df.to_excel(rules_path, index=False)
                
                # Save parameters
                params_df = pd.DataFrame([
                    {'Parameter': 'Core tree max depth', 'Value': self.config['core_tree_params']['max_depth']},
                    {'Parameter': 'Core tree min_samples_leaf', 'Value': self.config['core_tree_params']['min_samples_leaf']},
                    {'Parameter': 'Core rule min samples', 'Value': self.config['core_rule_min_samples']},
                    {'Parameter': 'Core rule min confidence', 'Value': self.config['core_rule_min_confidence']},
                    {'Parameter': 'Supplement rule min samples', 'Value': self.config['supplement_rule_min_samples']},
                    {'Parameter': 'Supplement rule min confidence', 'Value': self.config['supplement_rule_min_confidence']},
                ])
                params_path = os.path.join(rule_output_dir, "extraction_params.xlsx")
                params_df.to_excel(params_path, index=False)
            
            ProgressBar.update(total_steps, total_steps)
            
            return True if all_rules else False
            
        except Exception as e:
            self.logger.error(f"Rule extraction failed: {e}")
            return False

# ============================ MAIN EXECUTION PIPELINE ============================
def main():
    """
    Main execution pipeline for the complete analysis workflow
    
    This pipeline integrates:
    1. Lasso-based feature selection
    2. Random Forest classification
    3. Interpretable rule extraction
    """
    print("\n" + "="*80)
    print("🌱 NATURE PLANTS - Breeding Decision Kit")
    print("="*80)
    
    try:
        # Step 1: Initialize feature selector
        selector = LassoFeatureSelector(CONFIG)
        
        # Step 2: Perform feature selection
        selected_features, lasso_preprocess, lasso_results = selector.select_features(
            CONFIG['train_data_path'],
            CONFIG['val_data_path']
        )
        
        # Step 3: Save SNP subset and run rule extraction
        extractor = RuleExtractor(CONFIG)
        
        # Save SNP subset
        snp_subset_path = extractor.save_snp_subset(
            CONFIG['train_data_path'], selected_features
        )
        
        # Run rule extraction
        rule_extraction_success = extractor.run_extraction(
            snp_subset_path, CONFIG['label_column']
        )
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 Complete Workflow Finished!")
        print("="*80)
        print(f"📁 All results saved in: {CONFIG['output_dir']}")
        print(f"📊 Lasso selected features: {len(selected_features)}")
        print(f"📋 Rule extraction results: {'Success' if rule_extraction_success else 'Partial'}")
        print(f"\n📈 Main output files:")
        print(f"   - Lasso selection results: {os.path.join(CONFIG['output_dir'], CONFIG['lasso_selection_summary'])}")
        print(f"   - SNP feature subset: {snp_subset_path}")
        print(f"   - Extracted rules: {os.path.join(CONFIG['output_dir'], CONFIG['rule_extraction_dir'], 'extracted_rules.xlsx')}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================ EXECUTION GUARD ============================
if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Configure matplotlib for publication-quality figures
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.autolayout': True,
        'axes.unicode_minus': False
    })
    
    # Check for openpyxl
    try:
        import openpyxl
    except ImportError:
        print("📦 Installing openpyxl...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    
    # Execute main pipeline
    main()
