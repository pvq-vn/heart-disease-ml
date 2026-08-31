from sklearn.neighbors import KNeighborsClassifier

def get_knn_classifier():
    """
    Khởi tạo mô hình K-Nearest Neighbors Classifier.
    Các siêu tham số như n_neighbors, weights (uniform/distance) sẽ được
    tuning qua GridSearchCV.
    """
    return KNeighborsClassifier()
