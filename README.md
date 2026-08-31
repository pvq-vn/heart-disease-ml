# Heart Disease Classification

## Overview
Dự án **heart-disease-ml** là dự án thứ hai trong chuỗi học tập Machine Learning, nối tiếp sau dự án `california-housing-ml`. Trong khi dự án trước tập trung vào các thuật toán Regression (dự báo giá trị liên tục), dự án này đi sâu vào bài toán **Classification** (phân loại nhị phân), ứng dụng để chẩn đoán bệnh tim dựa trên các chỉ số y tế.

*Progression: Regression (california-housing-ml) → Classification (heart-disease-ml)*

## Learning Objectives
1. Nắm vững bản chất và cách cài đặt các mô hình Classification trong Scikit-learn.
2. Hiểu rõ về Data Leakage và tầm quan trọng của `Pipeline`.
3. Biết cách đánh giá Classification qua các độ đo (Accuracy, Precision, Recall, F1, ROC-AUC).
4. Áp dụng PCA (Principal Component Analysis) để giảm chiều dữ liệu.

## Dataset
Bộ dữ liệu **Heart Cleveland** gồm 297 bệnh nhân và 14 thuộc tính:
- Các biến phân loại (Categorical): `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`
- Các biến liên tục (Numerical): `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- Biến mục tiêu (Target): `condition` (0: Không mắc bệnh, 1: Mắc bệnh)

## Project Structure
```
heart-disease-ml/
├── data/                       # Chứa dataset gốc
├── notebooks/                  # 7 Notebooks học thuật (EDA, Model tuning, PCA...)
├── src/                        # Chứa source code preprocessing, metrics, và models
├── experiments/                # Nơi chứa kết quả evaluation và plots
├── report/                     # Báo cáo tổng quan dự án
├── run_experiments.py          # Script chính để chạy toàn bộ grid search
├── requirements.txt            # Package dependencies
└── README.md
```

## Algorithms
Các thuật toán được học và ứng dụng trong dự án (sử dụng 100% `scikit-learn`):
- Perceptron
- Logistic Regression
- KNN Classification
- PCA (kết hợp với Logistic Regression & KNN)

## Preprocessing
Tiền xử lý dữ liệu được chia làm 2 nhánh để tránh rò rỉ dữ liệu (Data Leakage):
- **Numerical:** Imputation (Median) + `StandardScaler`.
- **Categorical:** Imputation (Mode) + `OneHotEncoder`.

## Hyperparameter Tuning
Sử dụng `GridSearchCV` với 5-fold Cross-Validation trên tập **Training Data**:
- `Logistic Regression`: Tuning `C` và `solver`.
- `KNN`: Tuning `n_neighbors (K)` và `weights`.
- `Perceptron`: Tuning `alpha` và `penalty`.

## Evaluation Metrics
Sử dụng hàm tự viết `calculate_metrics()` tại `src/metrics.py` trả về:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Results
Kết quả thực nghiệm trên tập Test (Hold-out):
- **Logistic Regression** (với và không với PCA) đạt kết quả tốt nhất: Accuracy 90%, F1 0.88, ROC-AUC 0.96.
- K-Nearest Neighbors theo sau với Accuracy khoảng 86.67%.
- Chi tiết xin đọc tại `report/report.md` và `experiments/results.csv`.

## How to Run

1. Tạo virtual environment và cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy Grid Search và sinh biểu đồ:
   ```bash
   python run_experiments.py
   ```
3. Đọc Notebook: Mở thư mục `notebooks/` và chạy từng bài học (01 đến 07).

## Key Learnings
- **Logistic Regression** là một lựa chọn cực kỳ mạnh mẽ cho Classification nhờ cơ chế tối ưu xác suất bằng Sigmoid.
- **Scaling** có tác động khổng lồ tới hiệu suất của thuật toán KNN.
- **PCA** giúp tiết kiệm chi phí tính toán mà gần như không suy giảm (thậm chí giảm overfitting) cho model.

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- The Heart Disease Data Set from UCI Machine Learning Repository (Cleveland database).
