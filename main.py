# =========================================================
# IT Salary Analysis Project
# Mục tiêu:
# - Phân tích dữ liệu lương IT (EDA)
# - Xây dựng mô hình dự đoán lương
# - So sánh hiệu suất các mô hình
# =========================================================

# ===== IMPORT THƯ VIỆN =====
import logging              # Ghi log thay cho print
import joblib               # Lưu / load mô hình
import pandas as pd         # Xử lý dữ liệu dạng bảng
import numpy as np          # Tính toán số học

# Thư viện ML
from sklearn.model_selection import train_test_split   # Chia train/test
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Chuẩn hóa & encode
from sklearn.compose import ColumnTransformer          # Áp dụng xử lý theo cột
from sklearn.pipeline import Pipeline                  # Pipeline xử lý + model

# Các mô hình hồi quy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mô hình nâng cao
import xgboost as xgb

# Thư viện vẽ biểu đồ
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# =========================================================
# HÀM ĐÁNH GIÁ MÔ HÌNH
# =========================================================
def evaluate_model(name, model, X_test, y_test):
    """
    Đánh giá model bằng MAE, RMSE, R2
    """
    # Dự đoán
    y_pred = model.predict(X_test)

    # Tính metric
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # In log
    logging.info('%s - MAE: %.4f, RMSE: %.4f, R2: %.4f',
                 name, mae, rmse, r2)

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# =========================================================
# HÀM VẼ BIỂU ĐỒ (EDA + MODEL COMPARISON)
# =========================================================
def plot_visualizations(df, y, evaluation):
    logging.info('Vẽ biểu đồ trực quan hóa')

    # ===== Histogram (phân bố lương) =====
    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Salary in USD')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
    plt.show()

    # ===== Boxplot (phát hiện outlier) =====
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=y, color='skyblue')
    plt.title('Boxplot of Salary')
    plt.show()

    # ===== So sánh theo kinh nghiệm =====
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['experience_level'], y=df['salary_in_usd'])
    plt.title('Experience vs Salary')
    plt.show()

    # ===== So sánh theo quốc gia =====
    plt.figure(figsize=(12, 8))
    top_country = df['company_location'].value_counts().head(10).index
    sns.boxplot(data=df[df['company_location'].isin(top_country)],
                x='company_location', y='salary_in_usd')
    plt.xticks(rotation=45)
    plt.title('Salary by Country')
    plt.show()

    # ===== Correlation =====
    plt.figure(figsize=(10, 8))
    corr = df[['work_year', 'salary_in_usd']].corr()
    sns.heatmap(corr, annot=True)
    plt.title('Correlation Matrix')
    plt.show()

    # ===== So sánh R2 giữa model =====
    plt.figure(figsize=(8, 6))
    r2_values = [evaluation[m]['R2'] for m in evaluation]
    plt.bar(evaluation.keys(), r2_values)

    # Hiển thị số trên cột
    for i, v in enumerate(r2_values):
        plt.text(i, v, f'{v:.3f}', ha='center')

    plt.title('Model Comparison (R2)')
    plt.show()


# =========================================================
# CHƯƠNG TRÌNH CHÍNH
# =========================================================
if __name__ == '__main__':
    logging.info('Bắt đầu chương trình')

    # ===== Đọc dữ liệu =====
    df = pd.read_csv('data/raw/data_science_salaries.csv')

    # ===== Khám phá dữ liệu =====
    logging.info('Shape: %s', df.shape)
    logging.info('Head:\n%s', df.head())
    logging.info('Describe:\n%s', df.describe())

    # ===== Tách input/output =====
    X = df.drop(['salary_in_usd', 'salary'], axis=1, errors='ignore')
    y = df['salary_in_usd']

    # ===== Phân loại feature =====
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # ===== Tiền xử lý =====
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # ===== Chia dữ liệu =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # ===== Khởi tạo model =====
    models = {
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),

        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100))
        ]),

        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1))
        ])
    }

    results = {}
    evaluation = {}

    # ===== Train + Evaluate =====
    for name, model in models.items():
        logging.info('Training: %s', name)

        model.fit(X_train, y_train)

        results[name] = model
        evaluation[name] = evaluate_model(name, model, X_test, y_test)

    # ===== So sánh =====
    comparison_df = pd.DataFrame(evaluation).T
    logging.info('\n%s', comparison_df)

    # ===== Chọn model tốt nhất =====
    best_model_name = comparison_df['R2'].idxmax()
    logging.info('Best model: %s', best_model_name)

    # ===== Vẽ biểu đồ =====
    plot_visualizations(df, y, evaluation)

    # ===== Feature Importance (Random Forest) =====
    rf_model = results['Random Forest']

    feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out()
    importances = rf_model.named_steps['regressor'].feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    logging.info('\nTop Features:\n%s', importance_df)

    # ===== Vẽ importance =====
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title('Feature Importance')
    plt.show()

    # ===== Lưu model =====
    best_pipeline = models[best_model_name]
    joblib.dump(best_pipeline, 'models/best_salary_model.joblib')

    logging.info('Đã lưu model')