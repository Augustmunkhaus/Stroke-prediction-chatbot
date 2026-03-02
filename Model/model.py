import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import randint, uniform
# Removed SMOTE imports - using class weights instead
import json
import os
import warnings

warnings.filterwarnings('ignore')


class StrokePredictionModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}

    def load_and_explore_data(self):
        """Load the CSV and perform initial data exploration"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)

        print(f"Dataset shape: {self.df.shape}")
        print("\nDataset info:")
        print(self.df.info())
        print("\nFirst few rows:")
        print(self.df.head())
        print("\nTarget variable distribution:")
        print(self.df['stroke'].value_counts())
        print(f"\nStroke rate: {self.df['stroke'].mean():.3f}")

    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n=== DATA CLEANING ===")

        # Check for missing values
        print("Missing values before cleaning:")
        print(self.df.isnull().sum())

        # Remove rows with missing gender (if any)
        if self.df['gender'].isnull().sum() > 0:
            self.df = self.df.dropna(subset=['gender'])

        # Handle 'Other' in gender - remove these rows as they're very few
        if 'Other' in self.df['gender'].values:
            print(f"Removing {len(self.df[self.df['gender'] == 'Other'])} rows with 'Other' gender")
            self.df = self.df[self.df['gender'] != 'Other']

        # Handle BMI missing values - impute with median
        if self.df['bmi'].isnull().sum() > 0:
            bmi_imputer = SimpleImputer(strategy='median')
            self.df['bmi'] = bmi_imputer.fit_transform(self.df[['bmi']]).ravel()
            print(f"Imputed {self.df['bmi'].isnull().sum()} missing BMI values with median")

        # Handle smoking_status 'Unknown' - treat as separate category for now
        print(f"Smoking status distribution:\n{self.df['smoking_status'].value_counts()}")

        # Remove duplicate rows
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows")

        # Handle outliers in numerical columns
        numerical_cols = ['age', 'avg_glucose_level', 'bmi']
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_before = len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)])
            # Cap outliers instead of removing them
            self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
            print(f"Capped {outliers_before} outliers in {col}")

        print(f"\nFinal dataset shape: {self.df.shape}")
        print("Missing values after cleaning:")
        print(self.df.isnull().sum())

    def encode_features(self):
        """Encode categorical variables"""
        print("\n=== FEATURE ENCODING ===")

        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"Encoded {col}: {list(le.classes_)}")

    def prepare_features(self):
        """Prepare features for training - NO SAMPLING, just moderate class weights"""
        print("\n=== FEATURE PREPARATION (MODERATE CLASS WEIGHTS) ===")

        # Drop id column if exists
        if 'id' in self.df.columns:
            self.df = self.df.drop('id', axis=1)

        # Separate features and target
        X = self.df.drop('stroke', axis=1)
        y = self.df['stroke']

        print(f"Features: {list(X.columns)}")
        print(f"Feature matrix shape: {X.shape}")

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Training target distribution:\n{self.y_train.value_counts()}")
        print(f"Training stroke rate: {self.y_train.mean():.3f}")

        # Calculate moderate class weights (less extreme than 'balanced')
        stroke_rate = self.y_train.mean()
        no_stroke_rate = 1 - stroke_rate

        # Use square root of the imbalance ratio for more moderate weighting
        moderate_ratio = (no_stroke_rate / stroke_rate) ** 0.5  # Square root dampening

        self.class_weight_dict = {
            0: 1.0,  # No stroke gets weight 1
            1: moderate_ratio  # Stroke gets moderate weight (not extreme)
        }

        print(f"\nCalculated MODERATE class weights:")
        print(f"Class 0 (No stroke): {self.class_weight_dict[0]:.3f}")
        print(f"Class 1 (Stroke): {self.class_weight_dict[1]:.3f}")
        print(f"Stroke class is weighted {self.class_weight_dict[1]:.1f}x higher (vs 19.5x before)")
        print(f"This should improve recall while maintaining reasonable precision")

        # NO SMOTE - keeping original data distribution
        print("\n📊 Using ORIGINAL data with MODERATE CLASS WEIGHTS")
        print("This balances precision and recall better than extreme weighting")

        # Scale the features
        self.scaler.fit(self.X_train)
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\nTest set shape: {self.X_test.shape}")
        print(f"Test target distribution:\n{self.y_test.value_counts()}")

    def load_best_params(self):
        """Load best parameters from JSON file"""
        self.best_params_file = 'best_stroke_model_params.json'
        if os.path.exists(self.best_params_file):
            with open(self.best_params_file, 'r') as f:
                self.best_params = json.load(f)
            print(f"Loaded best parameters from {self.best_params_file}")
            return True
        return False

    def train_models_with_best_params(self):
        """Train models using previously found best parameters and moderate class weights"""
        print("\n=== TRAINING WITH BEST PARAMETERS + MODERATE CLASS WEIGHTS ===")

        if not self.load_best_params():
            print("No saved parameters found. Using simple training with moderate weights.")
            return self.train_models_simple()

        # --- Random Forest with best params + moderate class weights ---
        print("Training Random Forest with best parameters + moderate class weights...")
        rf_params = self.best_params.get('Random Forest', {
            'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5,
            'min_samples_leaf': 2
        })
        # Use moderate class weights instead of 'balanced'
        rf_params['class_weight'] = self.class_weight_dict

        rf = RandomForestClassifier(random_state=42, **rf_params)
        rf.fit(self.X_train, self.y_train)  # Use original training data
        self.models['Random Forest'] = rf
        print(f"RF params used: {rf_params}")

        # --- Logistic Regression with best params + moderate class weights ---
        print("\nTraining Logistic Regression with best parameters + moderate class weights...")
        lr_params = self.best_params.get('Logistic Regression', {
            'C': 1, 'penalty': 'l2', 'solver': 'liblinear'
        })
        # Use moderate class weights
        lr_params['class_weight'] = self.class_weight_dict

        lr = LogisticRegression(random_state=42, max_iter=2000, **lr_params)
        lr.fit(self.X_train_scaled, self.y_train)  # Use original scaled training data
        self.models['Logistic Regression'] = lr
        print(f"LR params used: {lr_params}")

        # --- Neural Network with best params (moderate sample weights) ---
        print("\nTraining Neural Network with best parameters...")
        print("Note: MLPClassifier doesn't support sample_weight, using moderate data balancing")
        nn_params = self.best_params.get('Neural Network', {
            'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'early_stopping': True,
            'learning_rate_init': 0.001, 'alpha': 0.01, 'solver': 'adam'
        })

        # For Neural Network, use moderate duplication (not extreme like before)
        X_train_balanced, y_train_balanced = self.balance_data_for_nn_moderate()

        nn = MLPClassifier(random_state=42, **nn_params)
        nn.fit(X_train_balanced, y_train_balanced)  # Use moderately balanced data
        self.models['Neural Network'] = nn
        print(f"NN params used: {nn_params}")

    def balance_data_for_nn_moderate(self):
        """Moderately balance data for Neural Network (not extreme duplication)"""
        print("Moderately balancing data for Neural Network...")

        # Separate majority and minority classes
        majority_mask = self.y_train == 0
        minority_mask = self.y_train == 1

        X_majority = self.X_train_scaled[majority_mask]
        y_majority = self.y_train[majority_mask]
        X_minority = self.X_train_scaled[minority_mask]
        y_minority = self.y_train[minority_mask]

        # Use moderate duplication (not full balancing)
        majority_count = len(y_majority)
        minority_count = len(y_minority)

        # Target ratio: make minority class about 1/3 of majority (instead of 1/1)
        target_minority_count = majority_count // 3
        duplication_factor = target_minority_count // minority_count

        print(f"Majority class: {majority_count}, Minority class: {minority_count}")
        print(f"Target minority count: {target_minority_count}")
        print(f"Moderate duplication factor: {duplication_factor}x (vs ~20x before)")

        # Duplicate minority class moderately
        X_minority_duplicated = np.tile(X_minority, (duplication_factor, 1))
        y_minority_duplicated = np.tile(y_minority, duplication_factor)

        # Combine balanced data
        X_balanced = np.vstack([X_majority, X_minority_duplicated])
        y_balanced = np.hstack([y_majority, y_minority_duplicated])

        print(f"Moderately balanced dataset size: {len(y_balanced)} (was {len(self.y_train)})")
        print(f"Moderate class distribution: {np.bincount(y_balanced)}")

        return X_balanced, y_balanced

    def train_models_simple(self):
        """Simple training with moderate class weights"""
        print("\n=== SIMPLE MODEL TRAINING WITH MODERATE CLASS WEIGHTS ===")

        # Random Forest
        print("Training Random Forest with moderate class weights...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight=self.class_weight_dict  # Use moderate weights instead of 'balanced'
        )
        rf.fit(self.X_train, self.y_train)  # Original training data
        self.models['Random Forest'] = rf

        # Logistic Regression
        print("Training Logistic Regression with moderate class weights...")
        lr = LogisticRegression(
            random_state=42,
            class_weight=self.class_weight_dict,  # Use moderate weights
            max_iter=1000
        )
        lr.fit(self.X_train_scaled, self.y_train)  # Original scaled training data
        self.models['Logistic Regression'] = lr

        # Neural Network with moderate balancing
        print("Training Neural Network with moderate data balancing...")
        print("Note: Using moderate duplication (not extreme) for better balance")

        # Create moderately balanced dataset for neural network
        X_train_balanced, y_train_balanced = self.balance_data_for_nn_moderate()

        nn = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=0.001,
            alpha=0.01,
            solver='adam'
        )
        nn.fit(X_train_balanced, y_train_balanced)  # Use moderately balanced data
        self.models['Neural Network'] = nn

    def evaluate_models(self):
        """Evaluate all trained models with focus on stroke prediction capability"""
        print("\n=== MODEL EVALUATION ===")

        for name, model in self.models.items():
            print(f"\n--- {name} (Moderate Class Weights) ---")

            # Use scaled data for models that need it
            if name in ['Logistic Regression', 'Neural Network']:
                X_test_eval = self.X_test_scaled
                X_train_eval = self.X_train_scaled
            else:
                X_test_eval = self.X_test
                X_train_eval = self.X_train

            # Predictions
            y_pred = model.predict(X_test_eval)
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]

            # Metrics
            train_score = model.score(X_train_eval, self.y_train)
            test_score = model.score(X_test_eval, self.y_test)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            print(f"Training Accuracy: {train_score:.4f}")
            print(f"Test Accuracy: {test_score:.4f}")
            print(f"ROC AUC Score: {roc_auc:.4f}")

            # Cross-validation with accuracy
            cv_scores = cross_val_score(model, X_train_eval, self.y_train, cv=5, scoring='accuracy')
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Classification report
            from sklearn.metrics import precision_recall_fscore_support
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))

            # Confusion Matrix and stroke-specific metrics
            print("\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            print(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
            print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")

            # Calculate stroke-specific metrics
            true_positives = cm[1, 1]
            false_negatives = cm[1, 0]
            false_positives = cm[0, 1]

            # Stroke recall (sensitivity) - most important for stroke prediction
            stroke_recall = true_positives / (true_positives + false_negatives) if (
                                                                                               true_positives + false_negatives) > 0 else 0
            # Stroke precision
            stroke_precision = true_positives / (true_positives + false_positives) if (
                                                                                                  true_positives + false_positives) > 0 else 0
            # F1 score for stroke class
            stroke_f1 = 2 * (stroke_precision * stroke_recall) / (stroke_precision + stroke_recall) if (
                                                                                                                   stroke_precision + stroke_recall) > 0 else 0

            print(f"\n🎯 STROKE PREDICTION METRICS:")
            print(
                f"Stroke Recall (Sensitivity): {stroke_recall:.4f} - {true_positives}/{true_positives + false_negatives} strokes caught")
            print(
                f"Stroke Precision: {stroke_precision:.4f} - {true_positives}/{true_positives + false_positives} stroke predictions correct")
            print(f"Stroke F1-Score: {stroke_f1:.4f}")

            # Store results with stroke-specific metrics
            self.results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'accuracy': test_score,
                'roc_auc': roc_auc,
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'stroke_recall': stroke_recall,  # Key metric for stroke prediction
                'stroke_precision': stroke_precision,
                'stroke_f1': stroke_f1,
                'strokes_caught': true_positives,
                'strokes_missed': false_negatives,
                'total_strokes': true_positives + false_negatives,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

    def plot_results(self):
        """Plot model comparison and results with best model selected by stroke recall"""
        print("\n=== PLOTTING RESULTS ===")

        # Get best model first (based on stroke recall)
        best_model_name = max(self.results.keys(),
                              key=lambda x: (self.results[x]['stroke_recall'], self.results[x]['stroke_f1']))

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Model comparison - Stroke Recall (most important metric)
        models = list(self.results.keys())
        stroke_recalls = [self.results[model]['stroke_recall'] for model in models]

        # Color the best model differently
        colors = ['gold' if model == best_model_name else 'lightblue' for model in models]

        bars = axes[0, 0].bar(models, stroke_recalls, color=colors)
        axes[0, 0].set_title('Model Comparison - Stroke Recall (Sensitivity)')
        axes[0, 0].set_ylabel('Stroke Recall')
        axes[0, 0].set_ylim(0, max(stroke_recalls) * 1.2)

        # Add value labels and highlight best model
        for i, (model, recall) in enumerate(zip(models, stroke_recalls)):
            label = f'{recall:.3f}'
            if model == best_model_name:
                label += ' ⭐ BEST'
            axes[0, 0].text(i, recall + 0.01, label, ha='center',
                            fontweight='bold' if model == best_model_name else 'normal')

        # ROC Curves
        for name, model in self.models.items():
            y_pred_proba = self.results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)

            # Highlight best model curve
            linewidth = 3 if name == best_model_name else 1
            linestyle = '-' if name == best_model_name else '--'

            axes[0, 1].plot(fpr, tpr,
                            label=f'{name} (AUC = {self.results[name]["roc_auc"]:.3f})' + (
                                ' ⭐' if name == best_model_name else ''),
                            linewidth=linewidth, linestyle=linestyle)

        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves (⭐ = Best for Stroke Prediction)')
        axes[0, 1].legend()

        # Feature importance for the BEST model (if it supports feature importance)
        if hasattr(self.models[best_model_name], 'feature_importances_'):
            # Best model has feature importance (Random Forest)
            model = self.models[best_model_name]
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)

            axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 0].set_title(f'Feature Importance - {best_model_name} ⭐')
            axes[1, 0].set_xlabel('Importance')

        elif hasattr(self.models[best_model_name], 'coef_'):
            # Best model has coefficients (Logistic Regression)
            model = self.models[best_model_name]
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': abs(model.coef_[0])  # Use absolute values of coefficients
            }).sort_values('importance', ascending=True)

            axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 0].set_title(f'Feature Importance (|Coefficients|) - {best_model_name} ⭐')
            axes[1, 0].set_xlabel('|Coefficient|')

        else:
            # Best model doesn't support feature importance (Neural Network)
            axes[1, 0].text(0.5, 0.5, f'{best_model_name}\ndoes not support\nfeature importance',
                            ha='center', va='center', transform=axes[1, 0].transAxes,
                            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 0].set_title(f'Feature Importance - {best_model_name} ⭐')

        # Confusion Matrix for BEST model
        best_y_pred = self.results[best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, best_y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name} ⭐\n(Best for Stroke Prediction)')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')

        # Add stroke metrics to confusion matrix plot
        stroke_recall = self.results[best_model_name]['stroke_recall']
        stroke_precision = self.results[best_model_name]['stroke_precision']
        strokes_caught = self.results[best_model_name]['strokes_caught']
        total_strokes = self.results[best_model_name]['total_strokes']

        axes[1, 1].text(0.02, 0.98,
                        f'Stroke Recall: {stroke_recall:.3f}\nCaught: {strokes_caught}/{total_strokes} strokes\nPrecision: {stroke_precision:.3f}',
                        transform=axes[1, 1].transAxes, va='top', ha='left',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

        plt.tight_layout()
        plt.show()

        print(f"\n📊 All plots now show results for {best_model_name} - the best model for stroke prediction!")
        print(f"🎯 This model catches {strokes_caught}/{total_strokes} strokes ({stroke_recall * 100:.1f}% recall)")

    def get_best_model(self):
        """Return the best performing model based on stroke prediction capability"""

        print(f"\n=== MODEL COMPARISON FOR STROKE PREDICTION ===")
        print(f"{'Model':<20} {'Accuracy':<10} {'Stroke Recall':<15} {'Strokes Caught':<15} {'Stroke F1':<12}")
        print("-" * 75)

        for name in self.results.keys():
            result = self.results[name]
            print(f"{name:<20} {result['accuracy']:<10.3f} {result['stroke_recall']:<15.3f} "
                  f"{result['strokes_caught']}/{result['total_strokes']:<13} {result['stroke_f1']:<12.3f}")

        # Select best model based on stroke recall (ability to catch strokes)
        # If there's a tie in stroke recall, use stroke F1 score as tiebreaker
        best_model_name = max(self.results.keys(),
                              key=lambda x: (self.results[x]['stroke_recall'], self.results[x]['stroke_f1']))

        result = self.results[best_model_name]
        print(f"\n=== BEST MODEL FOR STROKE PREDICTION: {best_model_name} ===")
        print(f"✅ Stroke Recall (Sensitivity): {result['stroke_recall']:.4f}")
        print(
            f"   → Catches {result['strokes_caught']} out of {result['total_strokes']} strokes ({result['stroke_recall'] * 100:.1f}%)")
        print(f"✅ Stroke Precision: {result['stroke_precision']:.4f}")
        print(f"✅ Stroke F1-Score: {result['stroke_f1']:.4f}")
        print(f"✅ Overall Accuracy: {result['accuracy']:.4f}")
        print(f"✅ ROC AUC Score: {result['roc_auc']:.4f}")
        print(f"\n🏥 This model is best at catching strokes (minimizing missed diagnoses)")

        return best_model_name, self.models[best_model_name]

    def save_model_info(self):
        """Save model information for later use"""
        best_model_name, best_model = self.get_best_model()

        # Convert class_weight_dict keys to regular integers for JSON compatibility
        class_weights_json = {int(k): float(v) for k, v in self.class_weight_dict.items()}

        # Save feature names and encoders info
        model_info = {
            'feature_names': list(self.X_train.columns),
            'label_encoders': {k: list(v.classes_) for k, v in self.label_encoders.items()},
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'best_model': best_model_name,
            'sampling_method': 'Moderate_Class_Weights',  # Updated to reflect moderate approach
            'class_weights': class_weights_json,  # Use JSON-compatible version
            'selection_criteria': 'stroke_recall'  # Added to show selection was based on stroke prediction
        }

        print(f"\nModel info for chatbot implementation:")
        print(f"Features: {model_info['feature_names']}")
        print(f"Label encoders: {model_info['label_encoders']}")
        print(f"Best model: {model_info['best_model']} (selected for best stroke recall)")
        print(f"Balancing method: {model_info['sampling_method']}")
        print(f"Class weights: {model_info['class_weights']}")

        return model_info

# Usage
def main(mode='simple'):
    """
    mode options:
    - 'use_best': Use saved best parameters (fast, daily use)
    - 'simple': Use default parameters (fastest, baseline)
    """
    # Initialize the model trainer
    stroke_model = StrokePredictionModel(
        'C:/Users/August/PycharmProjects/stroke_prediction/data/healthcare-dataset-stroke-data.csv')

    # Execute the data preparation pipeline
    stroke_model.load_and_explore_data()
    stroke_model.clean_data()
    stroke_model.encode_features()
    stroke_model.prepare_features()

    # Choose training method based on mode
    if mode == 'use_best':
        print("⚡ FAST MODE (using saved best parameters with Moderate Class Weights)")
        stroke_model.train_models_with_best_params()
    else:  # simple
        print("🏃 SIMPLE MODE (using default parameters with Moderate Class Weights)")
        stroke_model.train_models_simple()

    stroke_model.evaluate_models()
    stroke_model.plot_results()

    # Get the best model and save info for chatbot
    model_info = stroke_model.save_model_info()

    # Save the trained model and preprocessing components
    best_model_name, best_model = stroke_model.get_best_model()

    # Save the best model
    import joblib
    joblib.dump(best_model, 'best_stroke_model_moderate_weights.pkl')
    print(f"✅ Saved best model ({best_model_name}) to 'best_stroke_model_moderate_weights.pkl'")

    # Save model info for chatbot
    with open('model_info_moderate_weights.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("✅ Saved model info to 'model_info_moderate_weights.json'")

    # Save the scaler and label encoders separately
    joblib.dump(stroke_model.scaler, 'scaler_moderate_weights.pkl')
    joblib.dump(stroke_model.label_encoders, 'label_encoders_moderate_weights.pkl')
    print("✅ Saved preprocessing components")

    return stroke_model, model_info


if __name__ == "__main__":
    # Run with moderate class weights
    stroke_model, model_info = main(mode='simple')

    print("\n" + "=" * 50)
    print("SUMMARY: Moderate Class Weights Approach")
    print("=" * 50)
    print("✅ No synthetic data generation")
    print("✅ Preserves natural feature relationships")
    print("✅ Uses moderate class weighting (not extreme)")
    print("✅ Should improve recall while maintaining precision")
    print("✅ Better balance between catching strokes and avoiding false alarms")
    print("=" * 50)