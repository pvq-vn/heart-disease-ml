import numpy as np

class MLPClassifier:
    def __init__(
        self,
        hidden_layers=(32, 16),
        learning_rate=0.001,
        epochs=300,
        batch_size=32,
        weight_decay=0.0,
        dropout_rate=0.0,
        random_state=42
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.dropout_rate = float(dropout_rate)
        self.random_state = random_state

        if self.weight_decay < 0.0: raise ValueError()
        if not 0.0 <= self.dropout_rate < 1.0: raise ValueError()

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

    def _forward(self, X, training=False, rng=None):
        activations = [X]
        zs = []
        dropout_masks = []

        a = X

        for l in range(len(self.weights_) - 1):
            W = self.weights_[l]
            b = self.biases_[l]

            z = a @ W + b
            a = self._relu(z)

            if training and self.dropout_rate > 0.0:
                if rng is None: raise ValueError()
                mask = (rng.random(a.shape) >= self.dropout_rate).astype(np.float64)
                a = a * mask / (1.0 - self.dropout_rate)
            else:
                mask = None

            zs.append(z)
            activations.append(a)
            dropout_masks.append(mask)

        W = self.weights_[-1]
        b = self.biases_[-1]

        z = a @ W + b
        a = self._sigmoid(z)

        zs.append(z)
        activations.append(a)

        return activations, zs, dropout_masks

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

                activations, zs, dropout_masks = self._forward(X_batch, training=True, rng=rng)
                y_pred = activations[-1]

                data_loss = np.mean((y_batch - y_pred) ** 2)
                regularization_loss = (self.weight_decay * sum(np.sum(W ** 2) for W in self.weights_))

                loss = data_loss + regularization_loss

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
                    if dropout_masks[l] is not None:
                        da = da * dropout_masks[l] / (1.0 - self.dropout_rate)

                    dz = da * self._relu_deriv(zs[l])

                    dW[l] = activations[l].T @ dz
                    db[l] = np.sum(dz, axis=0, keepdims=True)

                    if l > 0: da = dz @ self.weights_[l].T

                for l in range(num_layers):
                    dW[l] += 2.0 * self.weight_decay * self.weights_[l]

                for l in range(num_layers):
                    self.weights_[l] -= (self.learning_rate * dW[l])
                    self.biases_[l] -= (self.learning_rate * db[l])

            self.loss_history_.append(epoch_loss / num_batches)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        _, zs, _ = self._forward(X, training=False)

        return zs[-1].ravel()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        activations, _, _ = self._forward(X, training=False)

        p1 = activations[-1].ravel()
        p0 = 1.0 - p1

        return np.column_stack((p0, p1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)