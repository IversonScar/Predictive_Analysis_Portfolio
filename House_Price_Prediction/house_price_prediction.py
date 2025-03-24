import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import os

# Set the random seed for reproducibility
np.random.seed(42)

# Load the dataset
data_path = os.path.join('..', 'House Pricing Dataset', 'house_prices.csv')
df = pd.read_csv(data_path)

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Data Preprocessing
print("\nChecking for missing values:")
print(df.isnull().sum())

# Convert categorical variables
df['waterfront'] = df['waterfront'].map({'Y': 1, 'N': 0})
df['condition'] = df['condition'].map({'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Very Good': 5})

# Feature Engineering
df['age'] = 2024 - df['yr_built']
df['renovated'] = (df['yr_renovated'] > 0).astype(int)
df['total_sqft'] = df['sqft_living'] + df['sqft_lot']
df['price_per_sqft'] = df['price'] / df['sqft_living']

# Exploratory Data Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('House Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig('price_distribution.png')
plt.close()

# Price vs Living Area
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sqft_living', y='price', data=df)
plt.title('Price vs Living Area')
plt.xlabel('Square Feet Living')
plt.ylabel('Price')
plt.savefig('price_vs_living_area.png')
plt.close()

# Feature Selection
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
            'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
            'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 
            'lat', 'long', 'age', 'renovated', 'total_sqft']

X = df[features]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training and Evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.4f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"R2 Score: {results[best_model_name]['R2']:.4f}")
print(f"RMSE: {results[best_model_name]['RMSE']:.2f}")

# Save the best model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, f'models/{best_model_name.replace(" ", "_").lower()}_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Feature Importance for tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = features
    
    if best_model_name == 'XGBoost':
        feature_importance['importance'] = best_model.feature_importances_
    else:
        feature_importance['importance'] = best_model.feature_importances_
    
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

print("\nHouse Price Prediction Model completed successfully!")
