"""
Customer Churn Prediction Model
This script builds a predictive model for customer churn based on behavior and demographic data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_PATH = r'c:\Users\IversonScarlettTAC\Downloads\Predictive Analysis\Predictive Analytics for Customer Churn Dataset'
OUTPUT_PATH = r'c:\Users\IversonScarlettTAC\Downloads\Predictive Analysis\Predictive_Analysis_Portfolio\Customer_Churn_Prediction'

def load_and_explore_data():
    """Load and explore the customer churn dataset"""
    print("Loading customer churn dataset...")
    
    # Load the data
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    
    # Load data descriptions if available
    try:
        desc_df = pd.read_csv(os.path.join(DATA_PATH, 'data_descriptions.csv'))
        print("\nData descriptions:")
        print(desc_df)
    except:
        print("Data descriptions file not found or could not be loaded.")
    
    # Display basic info
    print(f"\nTraining dataset shape: {train_df.shape}")
    print(f"Testing dataset shape: {test_df.shape}")
    
    print("\nFirst few rows of training data:")
    print(train_df.head())
    
    # Check for missing values
    print("\nMissing values in training data:")
    print(train_df.isnull().sum())
    
    # Data types
    print("\nData types in training data:")
    print(train_df.dtypes)
    
    # Basic statistics
    print("\nBasic statistics for training data:")
    print(train_df.describe())
    
    # Identify the target column (churn indicator)
    target_cols = [col for col in train_df.columns if 'churn' in col.lower() or 'target' in col.lower()]
    if target_cols:
        target_col = target_cols[0]
    else:
        # Try to identify a binary column that might be the target
        binary_cols = []
        for col in train_df.columns:
            if train_df[col].nunique() == 2:
                binary_cols.append(col)
        
        if binary_cols:
            target_col = binary_cols[-1]  # Use the last binary column as a fallback
            print(f"\nAssuming {target_col} as the target variable")
        else:
            print("Could not identify a suitable target column")
            return None
    
    # Class distribution
    print(f"\nClass distribution for {target_col}:")
    class_dist = train_df[target_col].value_counts(normalize=True) * 100
    print(class_dist)
    
    # Visualize class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=train_df)
    plt.title(f'Distribution of {target_col}')
    plt.savefig(os.path.join(OUTPUT_PATH, 'class_distribution.png'))
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = train_df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'correlation_heatmap.png'))
    
    return train_df, test_df, target_col

def preprocess_data(train_df, test_df, target_col):
    """Preprocess the customer churn dataset for modeling"""
    print("\nPreprocessing data...")
    
    # Make copies to avoid modifying the original dataframes
    train_data = train_df.copy()
    test_data = test_df.copy()
    
    # Check if target column exists in test data
    test_has_target = target_col in test_data.columns
    
    # Handle missing values
    for col in train_data.columns:
        if train_data[col].dtype == 'object':
            train_data[col].fillna('Unknown', inplace=True)
            if col in test_data.columns:
                test_data[col].fillna('Unknown', inplace=True)
        else:
            train_data[col].fillna(train_data[col].median(), inplace=True)
            if col in test_data.columns:
                test_data[col].fillna(train_data[col].median(), inplace=True)
    
    # Identify categorical and numerical columns
    categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from features
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # Define preprocessing for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Prepare features and target for training data
    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    
    # Prepare test data
    if test_has_target:
        X_test = test_data.drop(target_col, axis=1)
        y_test = test_data[target_col]
    else:
        # If test data doesn't have target, split training data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols

def build_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols):
    """Build and evaluate multiple classification models"""
    print("\nBuilding and evaluating models...")
    
    # Define models to try
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create a pipeline with preprocessing and the model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Store results
        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    # Find the best model based on ROC-AUC (good for imbalanced datasets)
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    best_metrics = {
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc']
    }
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best model metrics - Accuracy: {best_metrics['accuracy']:.4f}, Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1: {best_metrics['f1']:.4f}, ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    # Confusion Matrix for the best model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'))
    
    # ROC Curve for all models
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, 'roc_curve.png'))
    
    # Precision-Recall Curve for the best model
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, results[best_model_name]['y_prob'])
    plt.plot(recall, precision, label=f"{best_model_name}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, 'precision_recall_curve.png'))
    
    return best_model, best_model_name, best_metrics, results

def feature_importance(best_model, best_model_name, numerical_cols, categorical_cols):
    """Extract and visualize feature importance"""
    print("\nAnalyzing feature importance...")
    
    # Extract the model from the pipeline
    model = best_model.named_steps['model']
    
    # Check if the model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature names from preprocessor
        preprocessor = best_model.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_features)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Limit to top 20 features if there are many
        if len(feature_names) > 20:
            indices = np.argsort(importances)[-20:]
            plt.figure(figsize=(10, 12))
            plt.title(f'Top 20 Feature Importances - {best_model_name}')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        else:
            indices = np.argsort(importances)
            plt.figure(figsize=(10, 12))
            plt.title(f'Feature Importances - {best_model_name}')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'feature_importance.png'))
        
        # Return top features and their importance
        top_features = [(feature_names[i], importances[i]) for i in indices[::-1]]
        return top_features
    elif hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        # Get feature names from preprocessor
        preprocessor = best_model.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_features)
        
        # Get feature importances (absolute values of coefficients)
        importances = np.abs(model.coef_[0])
        
        # Limit to top 20 features if there are many
        if len(feature_names) > 20:
            indices = np.argsort(importances)[-20:]
            plt.figure(figsize=(10, 12))
            plt.title(f'Top 20 Feature Importances - {best_model_name}')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        else:
            indices = np.argsort(importances)
            plt.figure(figsize=(10, 12))
            plt.title(f'Feature Importances - {best_model_name}')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'feature_importance.png'))
        
        # Return top features and their importance
        top_features = [(feature_names[i], importances[i]) for i in indices[::-1]]
        return top_features
    else:
        print(f"Feature importance not available for {best_model_name}")
        return None

def save_model_and_results(best_model, best_model_name, best_metrics, top_features):
    """Save the model, metrics, and analysis results"""
    print("\nSaving model and results...")
    
    # Save the model
    joblib.dump(best_model, os.path.join(OUTPUT_PATH, 'customer_churn_model.pkl'))
    
    # Save metrics
    with open(os.path.join(OUTPUT_PATH, 'model_metrics.txt'), 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {best_metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {best_metrics['roc_auc']:.4f}\n\n")
        
        if top_features:
            f.write("Top Features:\n")
            for feature, importance in top_features[:10]:  # Save top 10 features
                f.write(f"{feature}: {importance:.4f}\n")
    
    # Create a README for this project
    with open(os.path.join(OUTPUT_PATH, 'README.md'), 'w') as f:
        f.write(f"# Customer Churn Prediction Model\n\n")
        f.write("## Overview\n")
        f.write("This project builds a machine learning model to predict customer churn based on behavior and demographic data.\n\n")
        f.write("## Dataset\n")
        f.write("The dataset contains customer information, including various behavioral and demographic features that might influence churn.\n\n")
        f.write("## Model Performance\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"- Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"- Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"- Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {best_metrics['f1']:.4f}\n")
        f.write(f"- ROC-AUC: {best_metrics['roc_auc']:.4f}\n\n")
        f.write("## Key Insights\n")
        if top_features:
            f.write("Top factors influencing customer churn:\n")
            for feature, importance in top_features[:5]:  # List top 5 features
                f.write(f"- {feature}: {importance:.4f}\n")
        f.write("\n## Files\n")
        f.write("- `customer_churn_prediction.py`: Main script for data analysis and model building\n")
        f.write("- `customer_churn_model.pkl`: Saved model file\n")
        f.write("- `model_metrics.txt`: Detailed model performance metrics\n")
        f.write("- `class_distribution.png`: Distribution of churn cases\n")
        f.write("- `correlation_heatmap.png`: Correlation between features\n")
        f.write("- `confusion_matrix.png`: Confusion matrix for the best model\n")
        f.write("- `roc_curve.png`: ROC curve for all models\n")
        f.write("- `precision_recall_curve.png`: Precision-Recall curve for the best model\n")
        f.write("- `feature_importance.png`: Visualization of feature importance\n")

def main():
    """Main function to run the customer churn prediction analysis"""
    print("Starting Customer Churn Prediction Analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load and explore data
    train_df, test_df, target_col = load_and_explore_data()
    if train_df is None:
        print("Error loading data. Exiting.")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols = preprocess_data(train_df, test_df, target_col)
    
    # Build and evaluate models
    best_model, best_model_name, best_metrics, results = build_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols
    )
    
    # Analyze feature importance
    top_features = feature_importance(best_model, best_model_name, numerical_cols, categorical_cols)
    
    # Save model and results
    save_model_and_results(best_model, best_model_name, best_metrics, top_features)
    
    print("\nCustomer Churn Prediction Analysis completed successfully!")

if __name__ == "__main__":
    main()
