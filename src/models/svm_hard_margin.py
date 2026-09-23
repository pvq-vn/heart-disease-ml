import numpy as np

class HardMarginSVM:
    def __init__(self, epochs=10000, tol=1e-6):
        self.epochs = int(epochs)
        self.tol = float(tol)

        self.w_ = None
        self.b_ = None
        self.lambda_ = None
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        self.is_separable_ = None
        self.n_violations_ = 0
        self.loss_history_ = []

    @staticmethod
    def _project_lambda(lam, y):
        lo, hi = -1.0, 1.0

        def constraint_value(t): return np.dot(y, np.maximum(0.0, lam - t * y))

        while constraint_value(lo) < 0.0: lo *= 2.0
        while constraint_value(hi) > 0.0: hi *= 2.0

        for _ in range(100):
            mid = 0.5 * (lo + hi)

            if constraint_value(mid) > 0.0: lo = mid
            else: hi = mid

        t = 0.5 * (lo + hi)

        return np.maximum(0.0, lam - t * y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2: raise ValueError()
        if X.shape[0] != y.shape[0]: raise ValueError()
        if not np.all(np.isin(y, [-1.0, 1.0])): raise ValueError()
        if self.epochs <= 0: raise ValueError()
        if self.tol <= 0: raise ValueError()

        n_samples, n_features = X.shape

        if np.unique(y).size != 2: raise ValueError()

        V = y[:, np.newaxis] * X
        Q = V @ V.T

        eigenvalues = np.linalg.eigvalsh(Q)
        L = np.max(eigenvalues)

        if L <= 0.0: raise ValueError()

        step = 1.0 / L
        lam = np.zeros(n_samples, dtype=np.float64)
        self.loss_history_ = []

        for _ in range(self.epochs):
            gradient = np.ones(n_samples) - Q @ lam
            new_lam = self._project_lambda(lam + step * gradient, y)
            change = np.linalg.norm(new_lam - lam)

            lam = new_lam
            w = (lam * y) @ X
            primal_objective = 0.5 * np.dot(w, w)

            self.loss_history_.append(float(primal_objective))
            if change <= self.tol: break

        self.lambda_ = lam
        support_mask = lam > 1e-8

        self.support_vector_indices_ = np.where(support_mask)[0]
        self.support_vectors_ = X[self.support_vector_indices_]

        if self.support_vector_indices_.size == 0:
            self.w_ = (lam * y) @ X
            self.b_ = 0.0
            self.is_separable_ = False
            self.n_violations_ = n_samples
            return self

        self.w_ = (lam * y) @ X
        support_indices = self.support_vector_indices_

        self.b_ = np.mean(y[support_indices] - X[support_indices] @ self.w_)
        margins = y * (X @ self.w_ + self.b_)

        violation_mask = margins < 1.0 - self.tol
        self.n_violations_ = int(np.sum(violation_mask))
        self.is_separable_ = (self.n_violations_ == 0)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.w_ is None or self.b_ is None: raise RuntimeError()

        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)