import numpy as np

class MulticlassSVM:
    def __init__(self, reg=0.1, learning_rate=1e-3, batch_size=1000, epochs=50, random_state=42):
        self.reg = float(reg)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.random_state = random_state

        self.W_ = None
        self.loss_history_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self.n_classes_ = None

    def _augment(self, X): return np.column_stack((np.ones(X.shape[0]), X))

    def _loss_gradient(self, W, X, y):
        N = X.shape[0]
        scores = X @ W

        correct_class_scores = scores[np.arange(N), y].reshape(N, 1)
        margins = np.maximum(0.0, 1.0 - correct_class_scores + scores)
        margins[np.arange(N), y] = 0.0

        loss = np.sum(margins) / N
        loss += 0.5 * self.reg * np.sum(W * W)

        mask = (margins > 0.0).astype(np.float64)
        mask[np.arange(N), y] = -np.sum(mask, axis=1)

        dW = X.T @ mask / N
        dW += self.reg * W

        return loss, dW

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()

        if X.ndim != 2: raise ValueError()
        if X.shape[0] != y.shape[0]: raise ValueError()
        if self.reg < 0: raise ValueError()
        if self.learning_rate <= 0: raise ValueError()
        if self.batch_size <= 0: raise ValueError()
        if self.epochs <= 0: raise ValueError()

        self.classes_ = np.unique(y)

        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            raise ValueError()

        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)

        X_augmented = self._augment(X)
        rng = np.random.default_rng(self.random_state)
        self.W_ = 1e-5 * rng.normal(size=(X_augmented.shape[1], self.n_classes_))

        self.loss_history_ = []
        N = X_augmented.shape[0]

        for _ in range(self.epochs):
            indices = rng.permutation(N)

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_indices = indices[start:end]

                X_batch = X_augmented[batch_indices]
                y_batch = y[batch_indices]

                loss, dW = self._loss_gradient(self.W_, X_batch, y_batch)

                self.W_ -= self.learning_rate * dW
                self.loss_history_.append(float(loss))

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.W_ is None: raise RuntimeError()
        if X.ndim != 2: raise ValueError()
        if X.shape[1] != self.n_features_in_: raise ValueError()

        X_augmented = self._augment(X)

        return X_augmented @ self.W_

    def predict(self, X):
        scores = self.decision_function(X)

        return np.argmax(scores, axis=1)