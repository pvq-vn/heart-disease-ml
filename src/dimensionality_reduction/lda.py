import numpy as np

class LDA:
    def __init__(self, n_components=None, reg_param=1e-4):
        self.n_components = n_components
        self.reg_param = float(reg_param)

        self.scalings_ = None
        self.means_ = {}
        self.overall_mean_ = None
        self.classes_ = None
        self.n_components_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        if X.ndim != 2: raise ValueError()
        if X.shape[0] != y.shape[0]: raise ValueError()
        if self.reg_param <= 0: raise ValueError()

        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2: raise ValueError()

        max_components = min(n_features, n_classes - 1)

        if self.n_components is None: k = max_components
        else:
            k = int(self.n_components)
            if k <= 0: raise ValueError()
            k = min(k, max_components)

        self.n_components_ = k
        self.n_features_in_ = n_features

        self.overall_mean_ = np.mean(X, axis=0)
        self.means_ = {}

        S_W = np.zeros((n_features, n_features), dtype=np.float64)
        S_B = np.zeros((n_features, n_features), dtype=np.float64)

        for c in self.classes_:
            X_c = X[y == c]
            n_c = X_c.shape[0]

            mean_c = np.mean(X_c, axis=0)
            self.means_[c] = mean_c

            X_c_centered = X_c - mean_c
            S_W += X_c_centered.T @ X_c_centered

            mean_diff = (mean_c - self.overall_mean_).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        S_W_reg = S_W + self.reg_param * np.eye(n_features)

        matrix = np.linalg.solve(S_W_reg, S_B)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]
        self.scalings_ = eigenvectors[:, :k]

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.scalings_ is None: raise RuntimeError()
        if X.ndim != 2: raise ValueError()
        if X.shape[1] != self.n_features_in_: raise ValueError()

        return X @ self.scalings_

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)