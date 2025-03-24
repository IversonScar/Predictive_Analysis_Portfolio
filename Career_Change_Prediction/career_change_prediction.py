import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import joblib
import os

# Set the random seed for reproducibility
np.random.seed(42)

# Load the dataset
data_path = os.path.join('..', 'career vs occupation data', 'career_change_prediction_dataset.csv')
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

# Check class distribution
print("\nClass Distribution:")
print(df['Likely to Change Occupation'].value_counts())
print(df['Likely to Change Occupation'].value_counts(normalize=True) * 100)

# Identify categorical and numerical columns
categorical_cols = ['Field of Study', 'Current Occupation', 'Gender', 'Education Level', 
                    'Family Influence', 'Mentorship Available', 'Certifications', 
                    'Freelancing Experience', 'Geographic Mobility', 'Career Change Events']
numerical_cols = ['Age', 'Years of Experience', 'Industry Growth Rate', 'Job Satisfaction', 
                  'Work-Life Balance', 'Job Opportunities', 'Salary', 'Job Security', 
                  'Career Change Interest', 'Skills Gap', 'Professional Networks', 
                  'Technology Adoption']

# Exploratory Data Analysis
plt.figure(figsize=(12, 8))
correlation = df[numerical_cols + ['Likely to Change Occupation']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Job Satisfaction vs Career Change
plt.figure(figsize=(10, 6))
sns.boxplot(x='Likely to Change Occupation', y='Job Satisfaction', data=df)
plt.title('Job Satisfaction vs Career Change')
plt.savefig('job_satisfaction_vs_career_change.png')
plt.close()

# Work-Life Balance vs Career Change
plt.figure(figsize=(10, 6))
sns.boxplot(x='Likely to Change Occupation', y='Work-Life Balance', data=df)
plt.title('Work-Life Balance vs Career Change')
plt.savefig('work_life_balance_vs_career_change.png')
plt.close()

# Salary vs Career Change
plt.figure(figsize=(10, 6))
sns.boxplot(x='Likely to Change Occupation', y='Salary', data=df)
plt.title('Salary vs Career Change')
plt.savefig('salary_vs_career_change.png')
plt.close()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Likely to Change Occupation', multiple='stack', bins=20)
plt.title('Age Distribution by Career Change Likelihood')
plt.savefig('age_distribution.png')
plt.close()

# Feature Engineering
# Create age groups
df['Age Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 70], labels=['20-30', '30-40', '40-50', '50-60', '60-70'])

# Create experience levels
df['Experience Level'] = pd.cut(df['Years of Experience'], bins=[0, 5, 10, 15, 20, 50], 
                               labels=['Entry', 'Junior', 'Mid', 'Senior', 'Expert'])

# Create salary bands
df['Salary Band'] = pd.qcut(df['Salary'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Update categorical columns
categorical_cols = categorical_cols + ['Age Group', 'Experience Level', 'Salary Band']

# Prepare the data for modeling
X = df[numerical_cols + categorical_cols]
y = df['Likely to Change Occupation']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Model Training and Evaluation
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"Cross-validation F1 Scores: {cv_scores}")
    print(f"Mean CV F1 Score: {cv_scores.mean():.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # ROC Curve (if the model supports predict_proba)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{name.replace(" ", "_").lower()}.png')
        plt.close()

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['Accuracy']:.4f}")
print(f"Precision: {results[best_model_name]['Precision']:.4f}")
print(f"Recall: {results[best_model_name]['Recall']:.4f}")
print(f"F1 Score: {results[best_model_name]['F1 Score']:.4f}")

# Detailed Classification Report for the best model
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

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
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title(f'Top 20 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

print("\nCareer Change Prediction Model completed successfully!")
