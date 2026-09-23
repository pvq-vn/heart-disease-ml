import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
        self.n_features_in_ = None
        self.n_samples_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2: raise ValueError()

        n_samples, n_features = X.shape
        if n_samples < 1: raise ValueError()

        self.n_samples_ = n_samples
        self.n_features_in_ = n_features

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance_matrix = (X_centered.T @ X_centered) / n_samples
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        eigenvalues = np.maximum(eigenvalues, 0.0)
        total_variance = np.sum(eigenvalues)

        if total_variance > 0.0: explained_variance_ratio = (eigenvalues / total_variance)
        else: explained_variance_ratio = np.zeros_like(eigenvalues)

        max_components = min(n_samples, n_features)

        if self.n_components is None: k = max_components
        elif isinstance(self.n_components, (int, np.integer)):
            k = int(self.n_components)
            if k <= 0: raise ValueError()
            k = min(k, max_components)
        elif isinstance(self.n_components, (float, np.floating)) and 0.0 < self.n_components <= 1.0:
            cumulative_variance = np.cumsum(explained_variance_ratio)
            k = np.searchsorted(cumulative_variance, self.n_components) + 1
            k = min(k, max_components)
        else: raise ValueError()

        self.n_components_ = k
        self.components_ = eigenvectors[:, :k].T
        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = explained_variance_ratio[:k]
        
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.components_ is None: raise ValueError()
        if X.ndim != 2: raise ValueError()
        if X.shape[1] != self.n_features_in_: raise ValueError()

        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.components_ is None: raise ValueError()
        if X.ndim != 2: raise ValueError()
        if X.shape[1] != self.n_components_: raise ValueError()

        return X @ self.components_ + self.mean_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)