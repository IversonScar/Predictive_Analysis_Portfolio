"""
Diabetes Prediction Model
This script builds a predictive model for diabetes risk based on health metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
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
DATA_PATH = r'c:\Users\IversonScarlettTAC\Downloads\Predictive Analysis\diabetes_dataset.csv'
OUTPUT_PATH = r'c:\Users\IversonScarlettTAC\Downloads\Predictive Analysis\Predictive_Analysis_Portfolio\Diabetes_Prediction'

def load_and_explore_data():
    """Load and explore the diabetes dataset"""
    print("Loading diabetes dataset...")
    
    # Load the data
    diabetes_df = pd.read_csv(os.path.join(DATA_PATH, 'diabetes_dataset.csv'))
    
    # Display basic info
    print(f"Dataset shape: {diabetes_df.shape}")
    print("\nFirst few rows:")
    print(diabetes_df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(diabetes_df.isnull().sum())
    
    # Data types
    print("\nData types:")
    print(diabetes_df.dtypes)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(diabetes_df.describe())
    
    # Check class distribution
    target_col = None
    for col in diabetes_df.columns:
        if col.lower() in ['diabetes', 'diabetic', 'outcome', 'target']:
            target_col = col
            break
    
    if target_col is None:
        # Try to identify a binary column that might be the target
        binary_cols = []
        for col in diabetes_df.columns:
            if diabetes_df[col].nunique() == 2:
                binary_cols.append(col)
        
        if binary_cols:
            target_col = binary_cols[0]  # Use the first binary column as a fallback
            print(f"\nAssuming {target_col} as the target variable")
        else:
            print("Could not identify a suitable target column")
            return None
    
    # Class distribution
    print(f"\nClass distribution for {target_col}:")
    class_dist = diabetes_df[target_col].value_counts(normalize=True) * 100
    print(class_dist)
    
    # Visualize class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=diabetes_df)
    plt.title(f'Distribution of {target_col}')
    plt.savefig(os.path.join(OUTPUT_PATH, 'class_distribution.png'))
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = diabetes_df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'correlation_heatmap.png'))
    
    return diabetes_df, target_col

def preprocess_data(df, target_col):
    """Preprocess the diabetes dataset for modeling"""
    print("\nPreprocessing data...")
    
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna('Unknown', inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Identify numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from features
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # Prepare features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, numerical_cols

def build_and_evaluate_models(X_train, X_test, y_train, y_test, numerical_cols):
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
            ('scaler', StandardScaler()),
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
    
    # Find the best model based on F1 score (balance between precision and recall)
    best_model_name = max(results, key=lambda x: results[x]['f1'])
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
    
    return best_model, best_model_name, best_metrics, results

def feature_importance(best_model, best_model_name, numerical_cols):
    """Extract and visualize feature importance"""
    print("\nAnalyzing feature importance...")
    
    # Extract the model from the pipeline
    model = best_model.named_steps['model']
    
    # Check if the model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importances - {best_model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [numerical_cols[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'feature_importance.png'))
        
        # Return top features and their importance
        top_features = [(numerical_cols[i], importances[i]) for i in indices]
        return top_features
    elif hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        importances = np.abs(model.coef_[0])
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importances - {best_model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [numerical_cols[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'feature_importance.png'))
        
        # Return top features and their importance
        top_features = [(numerical_cols[i], importances[i]) for i in indices]
        return top_features
    else:
        print(f"Feature importance not available for {best_model_name}")
        return None

def save_model_and_results(best_model, best_model_name, best_metrics, top_features):
    """Save the model, metrics, and analysis results"""
    print("\nSaving model and results...")
    
    # Save the model
    joblib.dump(best_model, os.path.join(OUTPUT_PATH, 'diabetes_prediction_model.pkl'))
    
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
        f.write(f"# Diabetes Prediction Model\n\n")
        f.write("## Overview\n")
        f.write("This project builds a machine learning model to predict diabetes risk based on various health metrics.\n\n")
        f.write("## Dataset\n")
        f.write("The dataset contains health metrics and diabetes status information for individuals.\n\n")
        f.write("## Model Performance\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"- Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"- Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"- Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {best_metrics['f1']:.4f}\n")
        f.write(f"- ROC-AUC: {best_metrics['roc_auc']:.4f}\n\n")
        f.write("## Key Insights\n")
        if top_features:
            f.write("Top factors influencing diabetes risk:\n")
            for feature, importance in top_features[:5]:  # List top 5 features
                f.write(f"- {feature}: {importance:.4f}\n")
        f.write("\n## Files\n")
        f.write("- `diabetes_prediction.py`: Main script for data analysis and model building\n")
        f.write("- `diabetes_prediction_model.pkl`: Saved model file\n")
        f.write("- `model_metrics.txt`: Detailed model performance metrics\n")
        f.write("- `class_distribution.png`: Distribution of diabetes cases\n")
        f.write("- `correlation_heatmap.png`: Correlation between features\n")
        f.write("- `confusion_matrix.png`: Confusion matrix for the best model\n")
        f.write("- `roc_curve.png`: ROC curve for all models\n")
        f.write("- `feature_importance.png`: Visualization of feature importance\n")

def main():
    """Main function to run the diabetes prediction analysis"""
    print("Starting Diabetes Prediction Analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load and explore data
    df, target_col = load_and_explore_data()
    if df is None:
        print("Error loading data. Exiting.")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, numerical_cols = preprocess_data(df, target_col)
    
    # Build and evaluate models
    best_model, best_model_name, best_metrics, results = build_and_evaluate_models(
        X_train, X_test, y_train, y_test, numerical_cols
    )
    
    # Analyze feature importance
    top_features = feature_importance(best_model, best_model_name, numerical_cols)
    
    # Save model and results
    save_model_and_results(best_model, best_model_name, best_metrics, top_features)
    
    print("\nDiabetes Prediction Analysis completed successfully!")

if __name__ == "__main__":
    main()
