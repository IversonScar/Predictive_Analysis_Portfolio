"""
Coffee Rating Prediction Model
This script builds a predictive model for coffee ratings based on various attributes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_PATH = r'c:\Users\IversonScarlettTAC\Downloads\Predictive Analysis\Coffee_Data_CoffeeReview'
OUTPUT_PATH = r'c:\Users\IversonScarlettTAC\Downloads\Predictive Analysis\Predictive_Analysis_Portfolio\Coffee_Rating_Prediction'

def load_and_explore_data():
    """Load and explore the coffee dataset"""
    print("Loading coffee dataset...")
    
    # Load the data
    coffee_df = pd.read_csv(os.path.join(DATA_PATH, 'coffee_clean.csv'))
    
    # Display basic info
    print(f"Dataset shape: {coffee_df.shape}")
    print("\nFirst few rows:")
    print(coffee_df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(coffee_df.isnull().sum())
    
    # Data types
    print("\nData types:")
    print(coffee_df.dtypes)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(coffee_df.describe())
    
    # If 'rating' column exists, show its distribution
    if 'rating' in coffee_df.columns:
        target_col = 'rating'
    elif 'total_cup_points' in coffee_df.columns:
        target_col = 'total_cup_points'
    else:
        # Try to identify a numerical column that might be the rating
        num_cols = coffee_df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            target_col = num_cols[0]  # Use the first numerical column as a fallback
            print(f"\nAssuming {target_col} as the target variable")
        else:
            print("Could not identify a suitable target column")
            return None
    
    plt.figure(figsize=(10, 6))
    sns.histplot(coffee_df[target_col], kde=True)
    plt.title(f'Distribution of {target_col}')
    plt.savefig(os.path.join(OUTPUT_PATH, 'rating_distribution.png'))
    
    return coffee_df, target_col

def preprocess_data(df, target_col):
    """Preprocess the coffee dataset for modeling"""
    print("\nPreprocessing data...")
    
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna('Unknown', inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
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
    
    # Prepare features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols

def build_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols):
    """Build and evaluate multiple regression models"""
    print("\nBuilding and evaluating models...")
    
    # Define models to try
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42)
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
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Store results
        results[name] = {
            'model': pipeline,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    # Find the best model based on RMSE
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    best_model = results[best_model_name]['model']
    best_metrics = {
        'rmse': results[best_model_name]['rmse'],
        'mae': results[best_model_name]['mae'],
        'r2': results[best_model_name]['r2']
    }
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best model metrics - RMSE: {best_metrics['rmse']:.4f}, MAE: {best_metrics['mae']:.4f}, R²: {best_metrics['r2']:.4f}")
    
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
    else:
        print(f"Feature importance not available for {best_model_name}")
        return None

def save_model_and_results(best_model, best_model_name, best_metrics, top_features):
    """Save the model, metrics, and analysis results"""
    print("\nSaving model and results...")
    
    # Save the model
    joblib.dump(best_model, os.path.join(OUTPUT_PATH, 'coffee_rating_model.pkl'))
    
    # Save metrics
    with open(os.path.join(OUTPUT_PATH, 'model_metrics.txt'), 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"RMSE: {best_metrics['rmse']:.4f}\n")
        f.write(f"MAE: {best_metrics['mae']:.4f}\n")
        f.write(f"R²: {best_metrics['r2']:.4f}\n\n")
        
        if top_features:
            f.write("Top Features:\n")
            for feature, importance in top_features[:10]:  # Save top 10 features
                f.write(f"{feature}: {importance:.4f}\n")
    
    # Create a README for this project
    with open(os.path.join(OUTPUT_PATH, 'README.md'), 'w') as f:
        f.write(f"# Coffee Rating Prediction Model\n\n")
        f.write("## Overview\n")
        f.write("This project builds a machine learning model to predict coffee ratings based on various attributes such as origin, processing method, and flavor profiles.\n\n")
        f.write("## Dataset\n")
        f.write("The dataset contains information about different coffee varieties, including their characteristics and ratings.\n\n")
        f.write("## Model Performance\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"- RMSE: {best_metrics['rmse']:.4f}\n")
        f.write(f"- MAE: {best_metrics['mae']:.4f}\n")
        f.write(f"- R²: {best_metrics['r2']:.4f}\n\n")
        f.write("## Key Insights\n")
        if top_features:
            f.write("Top factors influencing coffee ratings:\n")
            for feature, importance in top_features[:5]:  # List top 5 features
                f.write(f"- {feature}: {importance:.4f}\n")
        f.write("\n## Files\n")
        f.write("- `coffee_rating_prediction.py`: Main script for data analysis and model building\n")
        f.write("- `coffee_rating_model.pkl`: Saved model file\n")
        f.write("- `model_metrics.txt`: Detailed model performance metrics\n")
        f.write("- `rating_distribution.png`: Distribution of coffee ratings\n")
        f.write("- `feature_importance.png`: Visualization of feature importance\n")

def main():
    """Main function to run the coffee rating prediction analysis"""
    print("Starting Coffee Rating Prediction Analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load and explore data
    df, target_col = load_and_explore_data()
    if df is None:
        print("Error loading data. Exiting.")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols = preprocess_data(df, target_col)
    
    # Build and evaluate models
    best_model, best_model_name, best_metrics, results = build_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols
    )
    
    # Analyze feature importance
    top_features = feature_importance(best_model, best_model_name, numerical_cols, categorical_cols)
    
    # Save model and results
    save_model_and_results(best_model, best_model_name, best_metrics, top_features)
    
    print("\nCoffee Rating Prediction Analysis completed successfully!")

if __name__ == "__main__":
    main()
