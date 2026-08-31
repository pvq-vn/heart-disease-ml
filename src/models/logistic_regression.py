from sklearn.linear_model import LogisticRegression

def get_logistic_regression():
    """
    Khởi tạo mô hình Logistic Regression.
    Các siêu tham số (hyperparameters) như C, penalty, solver sẽ được 
    tuning qua GridSearchCV ở script run_experiments.py.
    """
    return LogisticRegression(max_iter=1000, random_state=42)
