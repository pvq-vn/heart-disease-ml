# Heart Disease Classification from Scratch

## Overview
Dự án **heart-disease-ml** là dự án nghiên cứu và học tập Machine Learning chuyên sâu, kế thừa và mở rộng từ nền tảng của `california-housing-ml`. Trong khi dự án trước tập trung vào hồi quy giá trị liên tục, dự án này đi sâu vào bài toán **Classification** (phân loại nhị phân và mở rộng đa lớp) ứng dụng trong y tế để chẩn đoán nguy cơ bệnh tim dựa trên các chỉ số sinh hóa và lâm sàng.

> **Core Philosophy:**  
> Core machine-learning algorithms are implemented from scratch using Python and NumPy. Scikit-learn is not used for model training, dimensionality reduction, metrics, or model selection.

---

## Learning Objectives & Architectural Principles
1. **From-Scratch Implementation:** Toàn bộ thuật toán lõi (Perceptron, Logistic Regression, KNN, MLP, SVD, PCA, LDA, SVMs) đều được xây dựng trực tiếp từ các biểu thức toán học thuần túy trên nền tảng vector hóa bằng NumPy.
2. **Ngăn chặn triệt để rò rỉ dữ liệu (Data Leakage Prevention):** `HeartDiseasePreprocessor` học các tham số thống kê (median, mean, std, category modes) độc quyền từ tập Training và áp dụng sang tập Test một cách nghiêm ngặt.
3. **Độ đo đánh giá tự cài đặt:** Toàn bộ độ đo phân loại (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC qua tích phân hình thang) được tính toán không dùng thư viện ngoài.
4. **Mở rộng thuật toán nâng cao:** Tích hợp Neural Networks (Chapter 16), Giảm chiều dữ liệu có giám sát và không giám sát (Chapters 20–22), Support Vector Machines từ lề cứng, lề mềm đến phi tuyến Kernel và đa lớp One-vs-Rest (Chapters 26–29).

---

## Dataset & Target Specification
Bộ dữ liệu **Heart Cleveland** gồm 297 bệnh nhân và 14 thuộc tính:
- **Biến liên tục (Numerical):** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`.
- **Biến phân loại/rời rạc (Categorical):** `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`.
- **Nhiệm vụ chính (Binary task):**
  - Nhãn `condition = 0` (Bình thường / Không mắc bệnh) và `condition = 1` (Mắc bệnh tim).
- **Nhiệm vụ mở rộng (Multi-class task):**
  - *Ghi chú quan trọng:* File dữ liệu hiện tại trong repository (`data/heart_cleveland_upload.csv`) đã gộp nhãn thành nhị phân (0 vs 1) và không còn bảo lưu nhãn độ nặng 0–4 (severity). Theo đúng nguyên tắc khoa học, chúng tôi không tạo nhãn giả lập trên dữ liệu lâm sàng mà kiểm chứng kiến trúc `OneVsRestSVM` và `OneVsOneSVM` trên bài toán đa lớp tổng quát.

---

## Implemented Algorithms in `src/`

### 1. Classification
- **Perceptron (`PerceptronClassifier`):** Cập nhật trọng số theo lỗi phân tách $(y - \\hat{y})$, hỗ trợ điều chuẩn L1 và L2.
- **Logistic Regression (`LogisticRegression`):** Hàm Sigmoid ổn định số học, tối ưu hàm mất mát Binary Cross-Entropy với điều chuẩn L2 bằng Gradient Descent.
- **K-Nearest Neighbors (`KNNClassifier`):** Tính toán khoảng cách Euclidean ma trận hóa, hỗ trợ biểu quyết đồng đều (`uniform`) và nghịch đảo khoảng cách (`distance`).

### 2. Neural Network (Chapter 16)
- **Multi-Layer Perceptron (`MLPClassifier`):**
  - Khởi tạo He Normal cho các tầng ẩn và Xavier cho tầng ra.
  - Hàm kích hoạt ReLU (ẩn) và Sigmoid (đầu ra).
  - Tích hợp **Inverted Dropout** tự động và **L2 Weight Decay**.
  - Lan truyền ngược (Backpropagation) chuẩn xác theo quy tắc chuỗi đạo hàm với Mini-batch SGD.

### 3. Dimensionality Reduction (Chapters 20–22)
- **Principal Component Analysis (`PCA`):** Giảm chiều không giám sát qua ma trận hiệp phương sai và phân rã trị riêng (`eigh`).
- **Truncated SVD (`TruncatedSVD`):** Phân rã giá trị suy biến trực tiếp trên ma trận dữ liệu đã trừ trung bình.
- **Linear Discriminant Analysis (`LDA`):** Giảm chiều có giám sát tối đa hóa tỷ số tán xạ giữa các lớp ($S_B$) so với nội bộ lớp ($S_W$).

### 4. Support Vector Machines (Chapters 26–29)
- **Hard-Margin SVM (`HardMarginSVM`):** Tối ưu bài toán nguyên thủy $\\min \\frac{1}{2}\\|w\\|^2$, kiểm tra và phát hiện tính phi phân tách tuyến tính.
- **Soft-Margin SVM (`SoftMarginSVM`):** Tối ưu Hinge Loss với biến bù trượt và hệ số phạt $C$.
- **Kernel SVM (`KernelSVM`):** Tối ưu bài toán đối ngẫu qua giải thuật SMO rút gọn, hỗ trợ hàm nhân **Linear, Polynomial, RBF, Sigmoid**.
- **Multi-class SVM (`OneVsRestSVM`, `OneVsOneSVM`):** Phân rã bài toán đa lớp thành các bài toán nhị phân.

---

## Project Structure
```text
heart-disease-ml/
│
├── data/
│   └── heart_cleveland_upload.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                     # Khám phá phân tích dữ liệu (EDA)
│   ├── 02_preprocessing.ipynb           # Tiền xử lý từ đầu & chống Data Leakage
│   ├── 03_perceptron.ipynb              # Perceptron classifier
│   ├── 04_logistic_regression.ipynb     # Logistic Regression
│   ├── 05_knn_classification.ipynb      # KNN classification
│   ├── 06_pca.ipynb                     # PCA dimensionality reduction
│   ├── 07_model_comparison.ipynb        # Tổng hợp và so sánh mô hình
│   ├── 08_mlp_classification.ipynb      # Chapter 16: Mạng nơ-ron MLP
│   ├── 09_svd_classification.ipynb      # Chapter 20-21: Truncated SVD
│   ├── 10_lda.ipynb                     # Chapter 22: Phân tích biệt số LDA
│   ├── 11_svm_hard_margin.ipynb         # Chapter 26: Hard-margin SVM
│   ├── 12_svm_soft_margin.ipynb         # Chapter 27: Soft-margin SVM
│   ├── 13_kernel_svm.ipynb              # Chapter 28: Kernel SVM (Linear, Poly, RBF, Sigmoid)
│   └── 14_multiclass_svm.ipynb          # Chapter 29: Multi-class SVM (OvR / OvO)
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                    # Stratified train_test_split from scratch
│   ├── metrics.py                       # Accuracy, Precision, Recall, F1, ROC-AUC
│   ├── preprocessing.py                 # HeartDiseasePreprocessor
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── perceptron.py
│   │   ├── logistic_regression.py
│   │   ├── knn_classifier.py
│   │   ├── neural_network.py
│   │   ├── svm_hard_margin.py
│   │   ├── svm_soft_margin.py
│   │   ├── kernel_svm.py
│   │   └── multiclass_svm.py
│   │
│   └── dimensionality_reduction/
│       ├── __init__.py
│       ├── pca.py
│       ├── svd.py
│       └── lda.py
│
├── experiments/
│   ├── results.csv                      # Kết quả benchmark toàn bộ mô hình
│   └── figures/                         # Biểu đồ phân tích và so sánh
│
├── report/
│   └── report.md                        # Báo cáo kỹ thuật chi tiết
│
├── run_experiments.py                   # Script chạy toàn bộ thực nghiệm
├── requirements.txt                     # Dependencies (NumPy, Pandas, Matplotlib, Seaborn)
└── README.md
```

---

## Installation & How to Run

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy toàn bộ thực nghiệm và sinh biểu đồ:**
   ```bash
   python run_experiments.py
   ```
   Kết quả kiểm định sẽ được lưu tại `experiments/results.csv` và các đồ thị trực quan hóa tại `experiments/figures/`.

3. **Khởi chạy kiểm thử tự động (Unit Tests):**
   ```bash
   python -m unittest tests/test_algorithms.py
   ```

4. **Nghiên cứu tương tác trên Notebook:**
   Khởi động Jupyter Notebook và mở các bài học từ `01` đến `14` trong thư mục `notebooks/`.

---

## References
- The Heart Disease Data Set from UCI Machine Learning Repository (Cleveland database).
- Christopher Bishop, *Pattern Recognition and Machine Learning*, Springer.
- Jerome Friedman, Trevor Hastie, Robert Tibshirani, *The Elements of Statistical Learning*, Springer.
