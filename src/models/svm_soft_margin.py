import numpy as np

class SoftMarginSVM:
    def __init__(self, C=1.0, learning_rate=0.05, epochs=10000, random_state=42):
        self.C = float(C)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.random_state = random_state

        self.w_ = None
        self.b_ = None
        self.loss_history_ = []

    def _loss(self, X, y):
        w = self.w_
        b = self.b_

        if w is None or b is None: raise RuntimeError()

        z = X @ w + b
        yz = y * z
        hinge_loss = np.maximum(0.0, 1.0 - yz)
        lam = 1.0 / self.C

        return (np.sum(hinge_loss) + 0.5 * lam * np.dot(w, w)) / X.shape[0]

    def _gradient(self, X, y):
        w = self.w_
        b = self.b_

        if w is None or b is None: raise RuntimeError()

        z = X @ w + b
        yz = y * z
        active = yz <= 1.0
        lam = 1.0 / self.C

        y_active = y[active]
        X_active = X[active]

        grad_w = (-np.sum(y_active[:, np.newaxis] * X_active, axis=0) + lam * w) / X.shape[0]
        grad_b = -np.sum(y_active) / X.shape[0]

        return grad_w, grad_b

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2: raise ValueError()
        if X.shape[0] != y.shape[0]: raise ValueError()
        if not np.all(np.isin(y, [-1.0, 1.0])): raise ValueError()
        if self.C <= 0: raise ValueError()
        if self.learning_rate <= 0: raise ValueError()
        if self.epochs <= 0: raise ValueError()

        rng = np.random.default_rng(self.random_state)

        self.w_ = 0.1 * rng.normal(size=X.shape[1])
        self.b_ = 0.1 * rng.normal()
        self.loss_history_ = []

        for _ in range(self.epochs):
            grad_w, grad_b = self._gradient(X, y)

            self.w_ -= self.learning_rate * grad_w
            self.b_ -= self.learning_rate * grad_b

            self.loss_history_.append(float(self._loss(X, y)))

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.w_ is None: raise RuntimeError()

        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)