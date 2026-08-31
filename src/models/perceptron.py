from sklearn.linear_model import Perceptron

def get_perceptron():
    """
    Khởi tạo mô hình Perceptron.
    Các siêu tham số như alpha, penalty, max_iter sẽ được
    tuning qua GridSearchCV.
    """
    return Perceptron(max_iter=1000, random_state=42)
