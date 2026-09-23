"""
Perceptron linear classification algorithm implemented from scratch using NumPy.
Includes L1 and L2 regularization options. No scikit-learn dependencies.
"""

import numpy as np


class PerceptronClassifier:
    """
    Perceptron linear classifier for binary classification.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size (eta) for parameter updates.
    epochs : int, default=1000
        Maximum number of passes over the training dataset.
    penalty : {None, 'l1', 'l2'}, default=None
        Regularization term to use.
    alpha : float, default=0.0001
        Regularization strength.
    random_state : int or None, default=42
        Seed for shuffle and parameter initialization.
    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000,
        penalty=None,
        alpha=0.0001,
        random_state=42
    ):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.penalty = penalty
        self.alpha = float(alpha)
        self.random_state = random_state

        self.w_ = None
        self.b_ = 0.0
        self.errors_ = []

    def fit(self, X, y):
        """
        Fit Perceptron on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) with labels in {0, 1}

        Returns
        -------
        self : PerceptronClassifier
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape

        rng = np.random.default_rng(self.random_state)
        # Initialize weights with small normal values or zeros
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = 0.0
        self.errors_ = []

        indices = np.arange(n_samples)

        for epoch in range(self.epochs):
            rng.shuffle(indices)
            num_errors = 0

            for idx in indices:
                xi = X_arr[idx]
                yi = y_arr[idx]

                linear_output = np.dot(xi, self.w_) + self.b_
                y_pred = 1.0 if linear_output >= 0.0 else 0.0

                update = self.learning_rate * (yi - y_pred)
                if update != 0.0:
                    num_errors += 1
                    self.w_ += update * xi
                    self.b_ += update

                # Apply weight regularization penalty (not to bias)
                if self.penalty == "l2":
                    self.w_ -= self.learning_rate * self.alpha * self.w_
                elif self.penalty == "l1":
                    self.w_ -= self.learning_rate * self.alpha * np.sign(self.w_)

            self.errors_.append(num_errors)
            if num_errors == 0:
                # Early stop if perfectly separated
                break

        return self

    def decision_function(self, X):
        """
        Evaluate net input z = Xw + b.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.w_) + self.b_

    def predict(self, X):
        """
        Predict binary labels {0, 1}.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        return np.where(self.decision_function(X) >= 0.0, 1, 0)


def get_perceptron(alpha=0.0001, penalty=None, learning_rate=0.01, epochs=1000):
    """
    Factory function returning a PerceptronClassifier instance.
    """
    return PerceptronClassifier(
        alpha=alpha,
        penalty=penalty,
        learning_rate=learning_rate,
        epochs=epochs,
        random_state=42
    )
