# Heart Disease Classification

## 1. Giới thiệu
Dự án **heart-disease-ml** là dự án thứ hai trong chuỗi học tập Machine Learning, nối tiếp sau dự án `california-housing-ml`. Trong khi dự án trước tập trung vào các thuật toán Regression (dự đoán giá trị liên tục), dự án này đi sâu vào bài toán **Classification** (phân loại), ứng dụng để chẩn đoán bệnh tim dựa trên các chỉ số y tế.

## 2. Bài toán
Đây là bài toán **Binary Classification** (phân loại nhị phân).
Mục tiêu là dự đoán biến `condition` (0: Không bị bệnh tim, 1: Bị bệnh tim) từ các thông tin thu thập được của bệnh nhân.

## 3. Dataset
Bộ dữ liệu **Heart Cleveland** gồm 297 mẫu và 14 cột:
- 13 đặc trưng (features) đầu vào.
- 1 biến mục tiêu (target) `condition`.
- Các biến số học (Numerical): `age`, `trestbps`, `chol`, `thalach`, `oldpeak`.
- Các biến phân loại/rời rạc (Categorical/Discrete): `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`.

Không có dữ liệu bị thiếu (missing values) được ghi nhận trong schema gốc. 

## 4. Exploratory Data Analysis
- Nhãn mục tiêu `condition` khá cân bằng (không bị mất cân bằng dữ liệu nghiêm trọng).
- Một số đặc trưng có sự tương quan tuyến tính (correlation) mức trung bình với `condition`, ví dụ: `thalach` (nhịp tim tối đa), `oldpeak`.

## 5. Data Preprocessing
- Pipeline xử lý phân chia rạch ròi 2 loại đặc trưng:
  - **Numerical**: Áp dụng `SimpleImputer(median)` và `StandardScaler` (rất quan trọng cho KNN và Perceptron để chuẩn hóa khoảng cách và tốc độ hội tụ).
  - **Categorical**: Áp dụng `SimpleImputer(most_frequent)` và `OneHotEncoder` để đưa về dạng biểu diễn vector nhị phân.
- Pipeline giúp ngăn chặn **data leakage** khi fit chỉ trên tập Train và transform cho cả Train/Test.

## 6. Perceptron
- Là một bộ phân loại tuyến tính đơn giản nhất, cập nhật trọng số liên tục dựa trên lỗi dự đoán.
- Tuy nhiên, do bản chất chỉ tìm kiếm siêu phẳng tuyến tính, mô hình này thường cho kết quả kém ổn định nhất (Accuracy: 73.3%, Precision: 64.3%).

## 7. Logistic Regression
- Dùng hàm sigmoid để ước lượng xác suất một mẫu thuộc về lớp 1.
- Tối ưu `C` (sức mạnh của regularization) thông qua GridSearchCV giúp mô hình đạt kết quả rất tốt.
- Mô hình này đạt hiệu suất ổn định và cao nhất trên tập Test (Accuracy: 90.0%, F1: 0.88).

## 8. KNN Classification
- K-Nearest Neighbors dựa trên việc biểu quyết từ `K` hàng xóm gần nhất.
- Tại `K=11`, với `weights='uniform'`, mô hình đạt độ chính xác khá tốt (Accuracy: 86.67%). Tuy nhiên, nó bị đánh bại bởi Logistic Regression trong kịch bản này.
- Bị ảnh hưởng rất nhiều nếu các biến không được chuẩn hóa (đó là lý do `StandardScaler` đóng vai trò tối quan trọng).

## 9. PCA
- Principal Component Analysis giúp **giảm chiều dữ liệu** (dimensionality reduction).
- Qua phân tích, chỉ cần khoảng một nửa số components ban đầu đã có thể giữ lại được **95% phương sai (variance)** của dữ liệu.

## 10. PCA + Classification
- **PCA + Logistic Regression**: Đạt Accuracy tương đương Logistic Regression gốc (90.0%) nhưng với số chiều ít hơn, giúp mô hình đào tạo nhanh hơn và kháng nhiễu tốt.
- **PCA + KNN Classification**: Đạt 86.67%, cũng tương đương KNN gốc (giữ nguyên phong độ).

## 11. Experimental Setup
- Chia Train/Test với tỷ lệ 80/20, sử dụng `stratify=y` để giữ nguyên tỷ lệ nhãn.
- GridSearchCV với 5-fold Cross-Validation trên tập Train.
- Test set chỉ được dùng ở bước đánh giá cuối cùng.

## 12. Evaluation Metrics
- **Accuracy**: Độ chính xác tổng thể.
- **Precision**: Độ chính xác của các ca được dự đoán là bệnh.
- **Recall**: Độ bao phủ - khả năng không bỏ sót người bị bệnh.
- **F1-Score**: Trung bình điều hòa của Precision và Recall.
- **ROC-AUC**: Khả năng phân tách giữa hai lớp dựa trên xác suất (Probability) hoặc Decision Function.

## 13. Results
Kết quả thu được trên tập Test:

| Model | CV_Accuracy | Test_Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 83.1% | **90.0%** | 1.00 | 0.786 | 0.880 | **0.962** |
| **PCA + Logistic Reg** | 82.3% | **90.0%** | 1.00 | 0.786 | 0.880 | 0.961 |
| **KNN Classifier** | 81.9% | 86.7% | 1.00 | 0.714 | 0.833 | 0.948 |
| **PCA + KNN Classifier**| 81.9% | 86.7% | 1.00 | 0.714 | 0.833 | 0.943 |
| **Perceptron** | 79.8% | 73.3% | 0.643| 0.964 | 0.771 | 0.925 |

## 14. Discussion
- **Model nào tốt nhất?** Logistic Regression (có PCA hoặc không) là tốt nhất với Accuracy 90% và ROC-AUC > 0.96.
- **Perceptron vs Logistic Regression?** Perceptron có Recall cực cao (0.964) nhưng Precision rất thấp (0.643), chứng tỏ nó dự đoán "có bệnh" quá nhiều (False Positive cao). Logistic Regression cân bằng rất tốt.
- **PCA có giúp không?** Có, nó giữ nguyên mức độ chính xác của các mô hình trong khi giảm đáng kể lượng tính toán, đây là một điểm sáng giá khi scale lên các tập dữ liệu lớn.

## 15. Conclusion
Bài toán Classification yêu cầu cách xử lý Metric và mô hình rất khác với Regression. Logistic Regression chứng minh là baseline model mạnh mẽ và hiệu quả nhất cho các bài toán y tế (Medical Diagnosis) nơi dữ liệu tuyến tính chiếm ưu thế và xác suất rủi ro được quan tâm. PCA là một công cụ mạnh để trích xuất những cụm features quan trọng nhất.
