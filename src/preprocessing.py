"""
Data preprocessing pipeline for the Heart Disease dataset implemented from scratch.
Eliminates scikit-learn Pipeline, ColumnTransformer, SimpleImputer,
StandardScaler, and OneHotEncoder.
Prevents data leakage by learning all statistics exclusively from training data.
"""

import numpy as np
import pandas as pd


class HeartDiseasePreprocessor:
    """
    Preprocessor for the Heart Disease Cleveland dataset.

    Features:
    - Numerical: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
      Imputation: Median
      Scaling: Standardization (mean=0, std=1)
    - Categorical: ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
      Imputation: Mode (most frequent)
      Encoding: One-hot encoding (unknown test categories map to all-zero vectors)
    """

    def __init__(self):
        self.num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

        # Learned statistics
        self.num_medians_ = {}
        self.num_means_ = {}
        self.num_stds_ = {}
        self.cat_modes_ = {}
        self.cat_categories_ = {}
        self.feature_names_ = []
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Learn imputation statistics, means, standard deviations, and
        categorical levels from training data.

        Parameters
        ----------
        X : pandas.DataFrame or dict-like
            Training feature dataset.
        y : None
            Ignored, present for API consistency.

        Returns
        -------
        self : HeartDiseasePreprocessor
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Fit numerical features
        self.num_medians_ = {}
        self.num_means_ = {}
        self.num_stds_ = {}
        for col in self.num_features:
            series = pd.to_numeric(X[col], errors="coerce")
            med = float(series.median())
            self.num_medians_[col] = med
            filled = series.fillna(med).to_numpy(dtype=np.float64)
            mean = float(np.mean(filled))
            std = float(np.std(filled, ddof=0))
            if std < 1e-8:
                std = 1.0
            self.num_means_[col] = mean
            self.num_stds_[col] = std

        # 2. Fit categorical features
        self.cat_modes_ = {}
        self.cat_categories_ = {}
        feature_names = list(self.num_features)

        for col in self.cat_features:
            series = X[col].dropna()
            if len(series) > 0:
                mode_val = series.mode().iloc[0]
            else:
                mode_val = 0
            self.cat_modes_[col] = mode_val

            # Unique deterministic sorted categories
            unique_cats = sorted(series.unique())
            self.cat_categories_[col] = unique_cats

            for cat in unique_cats:
                feature_names.append(f"{col}_{cat}")

        self.feature_names_ = feature_names
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform features using statistics learned during fit().

        Parameters
        ----------
        X : pandas.DataFrame or dict-like
            Feature dataset to transform.

        Returns
        -------
        X_out : numpy.ndarray of shape (n_samples, n_features), dtype float64
        """
        if not self.is_fitted_:
            raise RuntimeError("HeartDiseasePreprocessor is not fitted yet. Call fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_samples = len(X)
        transformed_parts = []

        # 1. Transform numerical features
        num_arr = np.zeros((n_samples, len(self.num_features)), dtype=np.float64)
        for i, col in enumerate(self.num_features):
            series = pd.to_numeric(X[col], errors="coerce").fillna(self.num_medians_[col])
            vals = series.to_numpy(dtype=np.float64)
            num_arr[:, i] = (vals - self.num_means_[col]) / self.num_stds_[col]
        transformed_parts.append(num_arr)

        # 2. Transform categorical features (one-hot encoding)
        for col in self.cat_features:
            series = X[col].fillna(self.cat_modes_[col])
            cats = self.cat_categories_[col]
            cat_arr = np.zeros((n_samples, len(cats)), dtype=np.float64)
            for j, cat in enumerate(cats):
                cat_arr[:, j] = (series == cat).to_numpy(dtype=np.float64)
            transformed_parts.append(cat_arr)

        return np.hstack(transformed_parts).astype(np.float64)

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pandas.DataFrame
        y : None

        Returns
        -------
        X_out : numpy.ndarray of dtype float64
        """
        return self.fit(X, y).transform(X)


def get_preprocessing_pipeline():
    """
    Factory function returning an instance of HeartDiseasePreprocessor
    for backward compatibility with existing scripts.
    """
    return HeartDiseasePreprocessor()
