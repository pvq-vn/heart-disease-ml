from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def calculate_metrics(y_true, y_pred, y_score=None):
    """
    Tính toán các metrics cho bài toán classification.
    
    Parameters:
    - y_true: Nhãn thực tế (Ground truth)
    - y_pred: Nhãn dự đoán (Predicted labels)
    - y_score: Xác suất dự đoán hoặc giá trị decision function (dùng cho ROC-AUC).
               Mặc định là None.
               
    Returns:
    - dictionary chứa các giá trị metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'ROC_AUC': 'N/A'
    }
    
    if y_score is not None:
        try:
            # y_score có thể là array 1D của xác suất class 1
            auc = roc_auc_score(y_true, y_score)
            metrics['ROC_AUC'] = auc
        except Exception:
            pass
            
    return metrics
