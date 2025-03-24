import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import os

# Set the random seed for reproducibility
np.random.seed(42)

# Load the dataset
data_path = os.path.join('..', 'Energy consumption prediction', 'Energy_consumption_dataset.csv')
df = pd.read_csv(data_path)

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Data Preprocessing
print("\nChecking for missing values:")
print(df.isnull().sum())

# Convert categorical variables
categorical_cols = ['DayOfWeek', 'Holiday', 'HVACUsage', 'LightingUsage']
numerical_cols = ['Month', 'Hour', 'Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy']

# Exploratory Data Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Energy Consumption Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['EnergyConsumption'], kde=True)
plt.title('Energy Consumption Distribution')
plt.xlabel('Energy Consumption')
plt.ylabel('Frequency')
plt.savefig('energy_consumption_distribution.png')
plt.close()

# Energy Consumption by Month
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='EnergyConsumption', data=df)
plt.title('Energy Consumption by Month')
plt.xlabel('Month')
plt.ylabel('Energy Consumption')
plt.savefig('energy_by_month.png')
plt.close()

# Energy Consumption by Hour
plt.figure(figsize=(14, 6))
sns.lineplot(x='Hour', y='EnergyConsumption', data=df, ci=None)
plt.title('Average Energy Consumption by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Energy Consumption')
plt.savefig('energy_by_hour.png')
plt.close()

# Energy Consumption vs Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temperature', y='EnergyConsumption', data=df, alpha=0.5)
plt.title('Energy Consumption vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Energy Consumption')
plt.savefig('energy_vs_temperature.png')
plt.close()

# Feature Engineering
df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
df['IsBusinessHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17) & (~df['DayOfWeek'].isin(['Saturday', 'Sunday']))).astype(int)
df['Season'] = pd.cut(df['Month'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])

# Update categorical columns
categorical_cols = ['DayOfWeek', 'Holiday', 'HVACUsage', 'LightingUsage', 'Season']
numerical_cols = ['Month', 'Hour', 'Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 
                  'RenewableEnergy', 'IsWeekend', 'IsBusinessHour']

# Prepare the data for modeling
X = df[categorical_cols + numerical_cols]
y = df['EnergyConsumption']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Model Training and Evaluation
models = {
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
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

# Hyperparameter tuning for the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7]
    }
else:  # Linear Regression
    param_grid = {}

if param_grid:
    print("\nPerforming hyperparameter tuning for the best model...")
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Evaluate the tuned model
    y_pred = best_model.predict(X_test)
    tuned_r2 = r2_score(y_test, y_pred)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Tuned model - R2: {tuned_r2:.4f}, RMSE: {tuned_rmse:.2f}")

# Save the best model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, f'models/{best_model_name.replace(" ", "_").lower()}_model.pkl')

# Feature Importance Analysis (if applicable)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    # Get feature names after one-hot encoding
    preprocessor_fitted = best_model.named_steps['preprocessor']
    feature_names = (
        numerical_cols +
        list(preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    )
    
    # Get feature importances
    feature_importances = best_model.named_steps['regressor'].feature_importances_
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(15))
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

print("\nEnergy Consumption Prediction Model completed successfully!")
