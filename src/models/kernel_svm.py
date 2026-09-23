import numpy as np

class KernelSVM:
    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma=1.0,
        degree=2.0,
        coef0=1.0,
        learning_rate=0.001,
        epochs=10000,
        tol=1e-6
    ):
        self.kernel = kernel
        self.C = float(C)
        self.gamma = float(gamma)
        self.degree = float(degree)
        self.coef0 = float(coef0)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.tol = float(tol)

        self.alphas_ = None
        self.b_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.support_vector_alphas_ = None
        self.support_vector_indices_ = None
        self.margin_vector_indices_ = None
        self.loss_history_ = []

    def _kernel_function(self, X1, X2):
        if self.kernel == "linear": return X1 @ X2.T
        if self.kernel == "poly": return (self.coef0 + self.gamma * (X1 @ X2.T)) ** self.degree
        if self.kernel == "sigmoid": return np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)
        if self.kernel == "rbf":
            X1_norm = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_norm = np.sum(X2 ** 2, axis=1, keepdims=True).T

            squared_distance = (X1_norm + X2_norm - 2.0 * (X1 @ X2.T))
            squared_distance = np.maximum(squared_distance, 0.0)

            return np.exp(-self.gamma * squared_distance)

        raise ValueError()

    @staticmethod
    def _project_alphas(alpha, y, C):
        lower, upper = -1.0, 1.0

        def constraint_value(t): return np.dot(y, np.clip(alpha - t * y, 0.0, C))

        while constraint_value(lower) < 0.0: lower *= 2.0
        while constraint_value(upper) > 0.0: upper *= 2.0

        for _ in range(100):
            middle = 0.5 * (lower + upper)

            if constraint_value(middle) > 0.0: lower = middle
            else: upper = middle

        t = 0.5 * (lower + upper)

        return np.clip(alpha - t * y, 0.0, C)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2: raise ValueError()
        if X.shape[0] != y.shape[0]: raise ValueError()
        if not np.all(np.isin(y, [-1.0, 1.0])): raise ValueError()
        if np.unique(y).size != 2: raise ValueError()
        if self.C <= 0: raise ValueError()
        if self.gamma <= 0: raise ValueError()
        if self.degree <= 0: raise ValueError()
        if self.learning_rate <= 0: raise ValueError()
        if self.epochs <= 0: raise ValueError()

        self.X_train_ = X
        self.y_train_ = y

        K = self._kernel_function(X, X)
        Q = y[:, np.newaxis] * y[np.newaxis, :] * K

        eigenvalues = np.linalg.eigvalsh(Q)
        L = np.max(eigenvalues)
        if L <= 0: raise ValueError()

        step = min(self.learning_rate, 1.0 / L)
        alpha = np.zeros(X.shape[0], dtype=np.float64)
        self.loss_history_ = []

        for _ in range(self.epochs):
            gradient = np.ones(X.shape[0]) - Q @ alpha
            new_alpha = self._project_alphas(alpha + step * gradient, y, self.C)
            change = np.linalg.norm(new_alpha - alpha)

            alpha = new_alpha
            dual_objective = np.sum(alpha) - 0.5 * alpha @ Q @ alpha
            self.loss_history_.append(float(dual_objective))

            if change <= self.tol: break

        self.alphas_ = alpha
        support_mask = alpha > self.tol

        self.support_vector_indices_ = np.where(support_mask)[0]
        self.support_vectors_ = X[self.support_vector_indices_]
        self.support_vector_labels_ = y[self.support_vector_indices_]
        self.support_vector_alphas_ = alpha[self.support_vector_indices_]

        margin_mask = (alpha > self.tol) & (alpha < self.C - self.tol)
        self.margin_vector_indices_ = np.where(margin_mask)[0]

        if self.margin_vector_indices_.size == 0: raise RuntimeError()

        M = self.margin_vector_indices_
        K_margin = self._kernel_function(X[M], X[self.support_vector_indices_])

        scores = K_margin @ (self.support_vector_alphas_ * self.support_vector_labels_)
        self.b_ = np.mean(y[M] - scores)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.alphas_ is None: raise RuntimeError()

        K = self._kernel_function(X, self.support_vectors_)

        return K @ (self.support_vector_alphas_ * self.support_vector_labels_) + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)