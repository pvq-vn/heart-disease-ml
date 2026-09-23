"""
Evaluation metrics for classification tasks implemented from scratch using NumPy.
No scikit-learn dependencies.
"""

import numpy as np


def _to_numpy_1d(arr):
    """Ensure array is a 1D numpy array."""
    if hasattr(arr, "to_numpy"):
        arr = arr.to_numpy()
    return np.asarray(arr).ravel()


def accuracy_score(y_true, y_pred):
    """
    Accuracy classification score.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))


def precision_score(y_true, y_pred):
    """
    Precision score for binary classification (positive class = 1).

    Precision = TP / (TP + FP)
    Returns 0.0 if denominator is 0.
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_score(y_true, y_pred):
    """
    Recall score for binary classification (positive class = 1).

    Recall = TP / (TP + FN)
    Returns 0.0 if denominator is 0.
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    tp = np.sum((yt == 1) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true, y_pred):
    """
    F1 score for binary classification.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Returns 0.0 if denominator is 0.
    """
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    denom = prec + rec
    return float(2.0 * prec * rec / denom) if denom > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    For binary classification with labels (0, 1), returns:
        [[TN, FP],
         [FN, TP]]
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def roc_auc_score(y_true, y_score):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    from prediction scores using the trapezoidal rule.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels in {0, 1}.
    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class or non-thresholded decision values.

    Returns
    -------
    auc : float
        Area Under the ROC Curve.
    """
    yt = _to_numpy_1d(y_true)
    ys = _to_numpy_1d(y_score)

    pos_count = np.sum(yt == 1)
    neg_count = np.sum(yt == 0)

    if pos_count == 0 or neg_count == 0:
        return 0.5

    # Sort descending by score
    desc_order = np.argsort(-ys)
    yt_sorted = yt[desc_order]
    ys_sorted = ys[desc_order]

    # Find unique score thresholds to handle duplicate scores correctly
    distinct_mask = np.empty(len(ys_sorted), dtype=bool)
    distinct_mask[:-1] = ys_sorted[:-1] != ys_sorted[1:]
    distinct_mask[-1] = True

    cum_tp = np.cumsum(yt_sorted == 1)[distinct_mask]
    cum_fp = np.cumsum(yt_sorted == 0)[distinct_mask]

    tpr = np.concatenate(([0.0], cum_tp / pos_count))
    fpr = np.concatenate(([0.0], cum_fp / neg_count))

    # Calculate AUC using trapezoidal rule
    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(tpr, fpr)
    else:
        auc = np.trapz(tpr, fpr)

    return float(auc)


def calculate_metrics(y_true, y_pred, y_score=None):
    """
    Calculate classification metrics dictionary.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_score : array-like or None, default=None
        Predicted probabilities or decision function values for ROC-AUC.

    Returns
    -------
    dict
        Dictionary containing Accuracy, Precision, Recall, F1_Score, and ROC_AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": "N/A"
    }

    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
            metrics["ROC_AUC"] = auc
        except Exception:
            metrics["ROC_AUC"] = "N/A"

    return metrics
