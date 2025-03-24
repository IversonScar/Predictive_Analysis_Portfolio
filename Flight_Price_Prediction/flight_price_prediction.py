"""
Flight Price Prediction Model
This script builds a predictive model for flight prices based on various factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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
DATA_PATH = r'c:\Users\IversonScarlettTAC\Downloads\Predictive Analysis\Flight Price Prediction'
OUTPUT_PATH = r'c:\Users\IversonScarlettTAC\Downloads\Predictive Analysis\Predictive_Analysis_Portfolio\Flight_Price_Prediction'

def load_and_explore_data():
    """Load and explore the flight price dataset"""
    print("Loading flight price dataset...")
    
    # Load the data
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'Data_Train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'Test_set.csv'))
    
    # Combine for exploration (we'll separate them again for modeling)
    if 'Price' in test_df.columns:
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = train_df.copy()
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Identify the price column
    price_cols = [col for col in df.columns if col.lower() in ['price', 'fare', 'cost', 'ticket_price']]
    if price_cols:
        target_col = price_cols[0]
    else:
        # Try to identify a numerical column that might be the price
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            target_col = num_cols[-1]  # Use the last numerical column as a fallback
            print(f"\nAssuming {target_col} as the target variable")
        else:
            print("Could not identify a suitable target column")
            return None
    
    # Visualize price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col].dropna(), kde=True)
    plt.title(f'Distribution of {target_col}')
    plt.savefig(os.path.join(OUTPUT_PATH, 'price_distribution.png'))
    
    # Check for outliers in price
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[target_col].dropna())
    plt.title(f'Boxplot of {target_col}')
    plt.savefig(os.path.join(OUTPUT_PATH, 'price_boxplot.png'))
    
    # Explore categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Plot count of flights by airline
    if 'Airline' in categorical_cols:
        plt.figure(figsize=(12, 6))
        sns.countplot(y='Airline', data=df)
        plt.title('Count of Flights by Airline')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'airline_counts.png'))
    
    # Plot average price by airline
    if 'Airline' in categorical_cols:
        plt.figure(figsize=(12, 6))
        airline_price = df.groupby('Airline')[target_col].mean().sort_values(ascending=False)
        sns.barplot(x=airline_price.values, y=airline_price.index)
        plt.title('Average Price by Airline')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'airline_avg_price.png'))
    
    return df, target_col, train_df, test_df

def preprocess_data(df, target_col, train_df, test_df):
    """Preprocess the flight price dataset for modeling"""
    print("\nPreprocessing data...")
    
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Handle date and time features
    if 'Date_of_Journey' in data.columns:
        data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], format='%d/%m/%Y')
        data['Journey_Day'] = data['Date_of_Journey'].dt.day
        data['Journey_Month'] = data['Date_of_Journey'].dt.month
        data['Journey_Year'] = data['Date_of_Journey'].dt.year
        data.drop('Date_of_Journey', axis=1, inplace=True)
    
    if 'Dep_Time' in data.columns:
        data['Dep_Time'] = pd.to_datetime(data['Dep_Time'])
        data['Dep_Hour'] = data['Dep_Time'].dt.hour
        data['Dep_Minute'] = data['Dep_Time'].dt.minute
        data.drop('Dep_Time', axis=1, inplace=True)
    
    if 'Arrival_Time' in data.columns:
        # Handle arrival time format which might include next day indicator
        data['Arrival_Time'] = data['Arrival_Time'].str.replace(' +1', '')
        data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'])
        data['Arrival_Hour'] = data['Arrival_Time'].dt.hour
        data['Arrival_Minute'] = data['Arrival_Time'].dt.minute
        data.drop('Arrival_Time', axis=1, inplace=True)
    
    # Handle duration
    if 'Duration' in data.columns:
        # Extract hours and minutes from duration
        data['Duration_Hours'] = data['Duration'].str.extract('(\\d+)h').fillna(0).astype(int)
        data['Duration_Minutes'] = data['Duration'].str.extract('(\\d+)m').fillna(0).astype(int)
        data['Total_Duration_Minutes'] = data['Duration_Hours'] * 60 + data['Duration_Minutes']
        data.drop('Duration', axis=1, inplace=True)
    
    # Handle stops
    if 'Total_Stops' in data.columns:
        data['Total_Stops'] = data['Total_Stops'].str.replace('non-stop', '0 stop')
        data['Total_Stops'] = data['Total_Stops'].str.extract('(\\d+)').fillna(0).astype(int)
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna('Unknown', inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Handle outliers in the target variable using capping
    if target_col in data.columns:
        Q1 = data[target_col].quantile(0.25)
        Q3 = data[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[target_col] = np.where(data[target_col] > upper_bound, upper_bound, data[target_col])
        data[target_col] = np.where(data[target_col] < lower_bound, lower_bound, data[target_col])
    
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
    
    # Prepare features and target using the training data
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    
    # Apply the same preprocessing to test data if it has the target column
    if target_col in test_df.columns:
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]
    else:
        # If test data doesn't have target, use a portion of train data for testing
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
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
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, results[best_model_name]['model'].predict(X_test), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Actual vs Predicted Prices - {best_model_name}')
    plt.savefig(os.path.join(OUTPUT_PATH, 'actual_vs_predicted.png'))
    
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
    joblib.dump(best_model, os.path.join(OUTPUT_PATH, 'flight_price_model.pkl'))
    
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
        f.write(f"# Flight Price Prediction Model\n\n")
        f.write("## Overview\n")
        f.write("This project builds a machine learning model to predict flight prices based on various factors such as airline, departure time, arrival time, duration, and number of stops.\n\n")
        f.write("## Dataset\n")
        f.write("The dataset contains information about flights, including their prices and various attributes that might influence pricing.\n\n")
        f.write("## Model Performance\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"- RMSE: {best_metrics['rmse']:.4f}\n")
        f.write(f"- MAE: {best_metrics['mae']:.4f}\n")
        f.write(f"- R²: {best_metrics['r2']:.4f}\n\n")
        f.write("## Key Insights\n")
        if top_features:
            f.write("Top factors influencing flight prices:\n")
            for feature, importance in top_features[:5]:  # List top 5 features
                f.write(f"- {feature}: {importance:.4f}\n")
        f.write("\n## Files\n")
        f.write("- `flight_price_prediction.py`: Main script for data analysis and model building\n")
        f.write("- `flight_price_model.pkl`: Saved model file\n")
        f.write("- `model_metrics.txt`: Detailed model performance metrics\n")
        f.write("- `price_distribution.png`: Distribution of flight prices\n")
        f.write("- `price_boxplot.png`: Boxplot showing price distribution and outliers\n")
        f.write("- `airline_counts.png`: Count of flights by airline\n")
        f.write("- `airline_avg_price.png`: Average price by airline\n")
        f.write("- `actual_vs_predicted.png`: Comparison of actual vs predicted prices\n")
        f.write("- `feature_importance.png`: Visualization of feature importance\n")

def main():
    """Main function to run the flight price prediction analysis"""
    print("Starting Flight Price Prediction Analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load and explore data
    df, target_col, train_df, test_df = load_and_explore_data()
    if df is None:
        print("Error loading data. Exiting.")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols = preprocess_data(df, target_col, train_df, test_df)
    
    # Build and evaluate models
    best_model, best_model_name, best_metrics, results = build_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols
    )
    
    # Analyze feature importance
    top_features = feature_importance(best_model, best_model_name, numerical_cols, categorical_cols)
    
    # Save model and results
    save_model_and_results(best_model, best_model_name, best_metrics, top_features)
    
    print("\nFlight Price Prediction Analysis completed successfully!")

if __name__ == "__main__":
    main()
