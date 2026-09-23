import numpy as np

class MLPClassifier:
    def __init__(
        self,
        hidden_layers=(32, 16),
        learning_rate=0.001,
        epochs=300,
        batch_size=32,
        random_state=42
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = random_state

        self.weights_ = []
        self.biases_ = []
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        z = np.asarray(z, dtype=np.float64)
        pos = z >= 0
        neg = ~pos

        result = np.empty_like(z)
        result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

        exp_z = np.exp(z[neg])
        result[neg] = exp_z / (1.0 + exp_z)

        return result

    @staticmethod
    def _relu(z): return np.maximum(0.0, z)

    @staticmethod
    def _relu_deriv(z): return (z > 0.0).astype(np.float64)

    @staticmethod
    def _sigmoid_deriv(a): return a * (1.0 - a)

    def _init_parameters(self, n_features, rng):
        layer_dims = [n_features] + list(self.hidden_layers) + [1]

        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i + 1]

            W = rng.normal(loc=0.0, scale=0.01, size=(n_in, n_out))
            b = np.zeros((1, n_out), dtype=np.float64)

            self.weights_.append(W)
            self.biases_.append(b)

    def _forward(self, X):
        activations = [X]
        zs = []

        a = X

        for l in range(len(self.weights_) - 1):
            W = self.weights_[l]
            b = self.biases_[l]

            z = a @ W + b
            a = self._relu(z)

            zs.append(z)
            activations.append(a)

        W = self.weights_[-1]
        b = self.biases_[-1]

        z = a @ W + b
        a = self._sigmoid(z)

        zs.append(z)
        activations.append(a)

        return activations, zs

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        if X.ndim != 2: raise ValueError()
        if y.shape[0] != X.shape[0]: raise ValueError()

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self._init_parameters(n_features, rng)
        self.loss_history_ = []

        batch_size = min(self.batch_size, n_samples)
        num_layers = len(self.weights_)

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                m = X_batch.shape[0]
                activations, zs = self._forward(X_batch)
                y_pred = activations[-1]

                loss = np.mean((y_batch - y_pred) ** 2)

                epoch_loss += loss
                num_batches += 1

                dW = [None] * num_layers
                db = [None] * num_layers

                da = 2.0 * (y_pred - y_batch) / m
                dz = da * self._sigmoid_deriv(y_pred)

                dW[-1] = activations[-2].T @ dz
                db[-1] = np.sum(dz, axis=0, keepdims=True)

                da = dz @ self.weights_[-1].T

                for l in reversed(range(num_layers - 1)):
                    dz = da * self._relu_deriv(zs[l])

                    dW[l] = activations[l].T @ dz
                    db[l] = np.sum(dz, axis=0, keepdims=True)

                    if l > 0: da = dz @ self.weights_[l].T

                for l in range(num_layers):
                    self.weights_[l] -= self.learning_rate * dW[l]
                    self.biases_[l] -= self.learning_rate * db[l]

            self.loss_history_.append(epoch_loss / num_batches)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)

        _, zs = self._forward(X)

        return zs[-1].ravel()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)

        activations, _ = self._forward(X)

        p1 = activations[-1].ravel()
        p0 = 1.0 - p1

        return np.column_stack((p0, p1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)