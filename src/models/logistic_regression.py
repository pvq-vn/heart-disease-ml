"""
Logistic Regression classifier implemented from scratch using NumPy.
Includes numerically stable sigmoid, binary cross-entropy loss, and L2 regularization.
No scikit-learn dependencies.
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression binary classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    epochs : int, default=2000
        Maximum number of training iterations.
    l2 : float, default=0.0
        L2 regularization parameter (lambda). If C is provided, l2 can be set as 1/C.
    fit_intercept : bool, default=True
        Whether to fit a bias/intercept term.
    random_state : int or None, default=42
        Seed for reproducibility.
    C : float or None, default=None
        Inverse of regularization strength. If provided, overrides l2 with 1.0 / C.
    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=2000,
        l2=0.0,
        fit_intercept=True,
        random_state=42,
        C=None
    ):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.C = C

        if C is not None and C > 0:
            self.l2 = float(1.0 / C)
        else:
            self.l2 = float(l2)

        self.w_ = None
        self.b_ = 0.0
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        """
        Numerically stable sigmoid function:
        For z >= 0: 1 / (1 + exp(-z))
        For z < 0:  exp(z) / (1 + exp(z))
        """
        z = np.asarray(z, dtype=np.float64)
        pos_mask = z >= 0
        neg_mask = ~pos_mask

        res = np.empty_like(z)
        res[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
        exp_neg = np.exp(z[neg_mask])
        res[neg_mask] = exp_neg / (1.0 + exp_neg)
        return res

    def fit(self, X, y):
        """
        Fit Logistic Regression model using batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) with labels in {0, 1}

        Returns
        -------
        self : LogisticRegression
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape

        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = 0.0
        self.loss_history_ = []

        eps = 1e-15

        for epoch in range(self.epochs):
            # Forward pass: z = Xw + b
            z = np.dot(X_arr, self.w_) + (self.b_ if self.fit_intercept else 0.0)
            p = self._sigmoid(z)

            # Binary cross entropy loss with epsilon stability
            p_clipped = np.clip(p, eps, 1.0 - eps)
            bce_loss = -np.mean(y_arr * np.log(p_clipped) + (1.0 - y_arr) * np.log(1.0 - p_clipped))
            reg_loss = 0.5 * self.l2 * np.sum(self.w_ ** 2)
            total_loss = bce_loss + reg_loss
            self.loss_history_.append(float(total_loss))

            # Gradients
            # dw = (1/n) * X^T (p - y) + lambda * w
            error = p - y_arr
            dw = (np.dot(X_arr.T, error) / n_samples) + (self.l2 * self.w_)

            # Gradient update
            self.w_ -= self.learning_rate * dw

            if self.fit_intercept:
                db = np.mean(error)
                self.b_ -= self.learning_rate * db

        return self

    def decision_function(self, X):
        """
        Compute decision values (logits) z = Xw + b.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.w_) + (self.b_ if self.fit_intercept else 0.0)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0: P(y=0), Column 1: P(y=1).
        """
        z = self.decision_function(X)
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

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
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, 1, 0)


def get_logistic_regression(learning_rate=0.01, epochs=2000, C=1.0, fit_intercept=True):
    """
    Factory function returning a LogisticRegression instance.
    """
    return LogisticRegression(
        learning_rate=learning_rate,
        epochs=epochs,
        C=C,
        fit_intercept=fit_intercept,
        random_state=42
    )
