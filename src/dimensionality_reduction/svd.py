import numpy as np

class TruncatedSVD:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U_ = None
        self.S_ = None
        self.Vt_ = None
        self.n_components_ = None
        self.n_features_in_ = None
        self.n_samples_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2: raise ValueError()
        n_samples, n_features = X.shape

        self.n_samples_ = n_samples
        self.n_features_in_ = n_features

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        rank = len(S)

        if self.n_components is None: k = rank
        else:
            k = int(self.n_components)
            if k <= 0: raise ValueError()
            k = min(k, rank)

        self.n_components_ = k

        self.U_ = U[:, :k]
        self.S_ = S[:k]
        self.Vt_ = Vt[:k]

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.Vt_ is None: raise ValueError()
        if X.ndim != 2: raise ValueError()
        if X.shape[1] != self.n_features_in_: raise ValueError()

        return X @ self.Vt_.T

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.Vt_ is None: raise ValueError()
        if X.ndim != 2: raise ValueError()
        if X.shape[1] != self.n_components_: raise ValueError()

        return X @ self.Vt_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)