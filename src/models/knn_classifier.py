"""
K-Nearest Neighbors (KNN) Classifier implemented from scratch using NumPy.
Supports uniform and distance weighting. No scikit-learn dependencies.
"""

import numpy as np


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier for binary classification.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction:
        - 'uniform': All points in each neighborhood are weighted equally.
        - 'distance': Weight points by the inverse of their distance.
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        """
        Store training feature matrix and target labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : KNNClassifier
        """
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64).ravel()
        return self

    def _compute_pairwise_distances(self, X):
        """
        Compute Euclidean distances between test samples X and training samples X_train.
        Uses expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x . y
        """
        X_arr = np.asarray(X, dtype=np.float64)
        x_norm = np.sum(X_arr ** 2, axis=1, keepdims=True)
        y_norm = np.sum(self.X_train_ ** 2, axis=1, keepdims=True).T
        cross = np.dot(X_arr, self.X_train_.T)
        dist_sq = np.maximum(x_norm + y_norm - 2.0 * cross, 0.0)
        return np.sqrt(dist_sq)

    def predict_proba(self, X):
        """
        Return probability estimates for the test data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        probabilities : ndarray of shape (n_queries, 2)
            Column 0: P(y=0), Column 1: P(y=1).
        """
        distances = self._compute_pairwise_distances(X)
        n_queries = distances.shape[0]
        k = min(self.n_neighbors, self.X_train_.shape[0])

        # Find top-k nearest neighbors
        # argpartition finds smallest k elements (not sorted)
        k_indices = np.argpartition(distances, k - 1, axis=1)[:, :k]

        # Gather distances and neighbor labels
        rows = np.arange(n_queries)[:, None]
        sorted_in_k = np.argsort(distances[rows, k_indices], axis=1)
        nearest_indices = k_indices[rows, sorted_in_k]
        nearest_distances = distances[rows, nearest_indices]
        nearest_labels = self.y_train_[nearest_indices]

        proba = np.zeros((n_queries, 2), dtype=np.float64)
        eps = 1e-8

        for i in range(n_queries):
            labels_i = nearest_labels[i]
            if self.weights == "distance":
                dists_i = nearest_distances[i]
                weights_i = 1.0 / (dists_i + eps)
                sum_weights = np.sum(weights_i)
                w_pos = np.sum(weights_i[labels_i == 1])
                p1 = w_pos / sum_weights if sum_weights > 0 else 0.5
            else:  # 'uniform'
                p1 = np.mean(labels_i == 1)

            proba[i, 1] = p1
            proba[i, 0] = 1.0 - p1

        return proba

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
        """
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, 1, 0)

    def decision_function(self, X):
        """
        Return decision score: P(class=1) - P(class=0).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        scores : ndarray of shape (n_queries,)
        """
        proba = self.predict_proba(X)
        return proba[:, 1] - proba[:, 0]


def get_knn_classifier(n_neighbors=5, weights="uniform"):
    """
    Factory function returning a KNNClassifier instance.
    """
    return KNNClassifier(n_neighbors=n_neighbors, weights=weights)
