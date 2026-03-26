import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logging.info('%s - MAE: %.4f, RMSE: %.4f, R2: %.4f', name, mae, rmse, r2)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def plot_visualizations(df, y, evaluation):
    logging.info('Vẽ biểu đồ: histogram, boxplot, correlation, comparison')

    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Salary in USD')
    plt.xlabel('Salary in USD')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=y, color='skyblue')
    plt.title('Boxplot of Salary in USD')
    plt.ylabel('Salary in USD')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['experience_level'], y=df['salary_in_usd'], palette='Set3')
    plt.title('Experience Level vs Salary')
    plt.xlabel('Experience Level')
    plt.ylabel('Salary in USD')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 8))
    top_country = df['company_location'].value_counts().head(10).index
    sns.boxplot(data=df[df['company_location'].isin(top_country)], x='company_location', y='salary_in_usd', palette='Set3')
    plt.xticks(rotation=45)
    plt.title('Salary by Company Location (Top 10 Countries)')
    plt.xlabel('Company Location')
    plt.ylabel('Salary in USD')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 8))
    corr = df[['work_year', 'salary_in_usd']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    plt.figure(figsize=(8, 6))
    r2_values = [evaluation[model]['R2'] for model in evaluation.keys()]
    plt.bar(evaluation.keys(), r2_values, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('R2 Score Comparison')
    plt.ylabel('R2 Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    plt.show()


if __name__ == '__main__':
    logging.info('Bắt đầu phân tích salary_in_usd')

    df = pd.read_csv('data/raw/data_science_salaries.csv')

    logging.info('=== Data Exploration ===')
    logging.info('Shape: %s', df.shape)
    logging.info('\nHead (first 5 rows):\n%s', df.head())
    logging.info('\nDescribe:\n%s', df.describe())

    X = df.drop(['salary_in_usd', 'salary'], axis=1, errors='ignore')
    y = df['salary_in_usd']

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    logging.info('Features used for training: %s', X.columns.tolist())


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
        'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))]),
        'XGBoost': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, verbosity=0))])
    }

    results = {}
    evaluation = {}

    for name, model in models.items():
        logging.info('Huấn luyện mô hình: %s', name)
        model.fit(X_train, y_train)
        results[name] = model
        evaluation[name] = evaluate_model(name, model, X_test, y_test)

    comparison_df = pd.DataFrame(evaluation).T
    logging.info('=== Model Comparison Table ===\n%s', comparison_df.round(3))
    best_model_name = comparison_df['R2'].idxmax()
    best_score = comparison_df['R2'].max()
    logging.info('Best Model: %s (R2=%.4f)', best_model_name, best_score)

    plot_visualizations(df, y, evaluation)

    rf_model = results['Random Forest']
    feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out()
    importances = rf_model.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    top_10_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    logging.info('=== Feature Importance (Random Forest) ===')
    logging.info('\n%s', top_10_features.round(4))

    plt.figure(figsize=(10, 8))
    plt.barh(top_10_features['Feature'], top_10_features['Importance'], color='skyblue', alpha=0.8)
    plt.title('Top 10 Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.show()

    best_pipeline = models[best_model_name]
    joblib.dump(best_pipeline, 'models/best_salary_model.joblib')
    logging.info('Best model đã được lưu vào best_salary_model.joblib')
