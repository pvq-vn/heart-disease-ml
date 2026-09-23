"""
Data utility functions for Heart Disease ML project.
Self-implemented data splitting without scikit-learn.
"""

import numpy as np
import pandas as pd


def train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=True
):
    """
    Split arrays or DataFrames into random train and test subsets.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target vector.
    test_size : float or int, default=0.2
        Proportion of dataset to include in the test split (if float)
        or absolute number of test samples (if int).
    random_state : int or None, default=42
        Controls the shuffling applied to the data before splitting.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    stratify : bool, default=True
        If True, data is split in a stratified fashion using the class labels of y.

    Returns
    -------
    X_train, X_test, y_train, y_test : same types as input
        Train-test split of inputs.
    """
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(
            f"X and y must have the same length. Got len(X)={n_samples}, len(y)={len(y)}"
        )

    rng = np.random.default_rng(random_state)

    # Convert y to 1D numpy array for indexing/stratification
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_arr = np.asarray(y).ravel()
    else:
        y_arr = np.asarray(y).ravel()

    # Determine stratification target
    do_stratify = False
    stratify_labels = None
    if isinstance(stratify, bool):
        if stratify:
            do_stratify = True
            stratify_labels = y_arr
    elif stratify is not None:
        do_stratify = True
        stratify_labels = np.asarray(stratify).ravel()

    if do_stratify:
        unique_classes, counts = np.unique(stratify_labels, return_counts=True)
        train_indices_list = []
        test_indices_list = []

        for cls in unique_classes:
            cls_indices = np.where(stratify_labels == cls)[0]
            n_cls = len(cls_indices)

            if shuffle:
                cls_indices = cls_indices.copy()
                rng.shuffle(cls_indices)

            if isinstance(test_size, float):
                n_cls_test = int(round(n_cls * test_size))
            else:
                # Approximate proportion based on class frequency
                n_cls_test = int(round(test_size * (n_cls / n_samples)))

            # Guarantee at least 1 test sample if test_size > 0 and n_cls > 1
            if test_size > 0 and n_cls > 1 and n_cls_test == 0:
                n_cls_test = 1
            elif n_cls_test >= n_cls:
                n_cls_test = n_cls - 1

            test_indices_list.append(cls_indices[:n_cls_test])
            train_indices_list.append(cls_indices[n_cls_test:])

        train_indices = np.concatenate(train_indices_list)
        test_indices = np.concatenate(test_indices_list)

        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)
    else:
        indices = np.arange(n_samples)
        if shuffle:
            rng.shuffle(indices)

        if isinstance(test_size, float):
            n_test = int(round(n_samples * test_size))
        else:
            n_test = int(test_size)

        n_train = n_samples - n_test
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    # Index X
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices].copy().reset_index(drop=True)
        X_test = X.iloc[test_indices].copy().reset_index(drop=True)
    elif isinstance(X, pd.Series):
        X_train = X.iloc[train_indices].copy().reset_index(drop=True)
        X_test = X.iloc[test_indices].copy().reset_index(drop=True)
    else:
        X_arr = np.asarray(X)
        X_train = X_arr[train_indices].copy()
        X_test = X_arr[test_indices].copy()

    # Index y
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_train = y.iloc[train_indices].copy().reset_index(drop=True)
        y_test = y.iloc[test_indices].copy().reset_index(drop=True)
    else:
        y_arr_full = np.asarray(y)
        y_train = y_arr_full[train_indices].copy()
        y_test = y_arr_full[test_indices].copy()

    return X_train, X_test, y_train, y_test
