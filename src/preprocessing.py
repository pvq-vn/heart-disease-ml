import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline():
    """
    Tạo một scikit-learn Pipeline để tiền xử lý dữ liệu.
    
    Phân loại đặc trưng (features):
    - Numerical: age, trestbps, chol, thalach, oldpeak
    - Categorical/Discrete: sex, cp, fbs, restecg, exang, slope, ca, thal
    """
    
    # 1. Numerical Pipeline
    # Điền khuyết bằng trung vị (median) và chuẩn hóa bằng StandardScaler
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Categorical Pipeline
    # Điền khuyết bằng giá trị xuất hiện nhiều nhất (mode) và One-Hot Encoding
    # Sử dụng handle_unknown='ignore' để tránh lỗi khi tập test xuất hiện category mới
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 3. Kết hợp bằng ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])
    
    return preprocessor
