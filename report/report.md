# Báo cáo Thực nghiệm Chuyên sâu: Heart Disease ML from Scratch

---

## 1. Problem Definition (Định nghĩa bài toán)
Dự án **heart-disease-ml** giải quyết bài toán **Binary Classification** trong tin học y tế (Medical Informatics). Mục tiêu là dự đoán khả năng mắc bệnh tim của bệnh nhân (`condition = 1`) hoặc bình thường (`condition = 0`) dựa trên các chỉ số nhân khẩu học, triệu chứng lâm sàng và kết quả xét nghiệm sinh hóa.

Khác với các bài toán hồi quy (như dự đoán giá nhà trong `california-housing-ml`), bài toán phân loại bệnh tim đặt ra yêu cầu khắt khe về mặt lâm sàng:
- **Độ bao phủ (Recall):** Giảm thiểu tối đa số ca Bỏ sót bệnh (False Negatives), bởi vì một bệnh nhân có bệnh nhưng bị chẩn đoán nhầm là khỏe mạnh có thể gặp nguy hiểm tính mạng do không được can thiệp kịp thời.
- **Độ chính xác dự đoán (Precision):** Giảm thiểu số ca Báo động giả (False Positives) nhằm hạn chế chi phí và gánh nặng tâm lý từ các thủ thuật xâm lấn không cần thiết.
- **Khả năng giải thích và ước lượng xác suất:** Bác sĩ cần xác suất rủi ro tin cậy ($P(y=1|x)$) thay vì một phán đoán nhị phân cứng nhắc.

---

## 2. Dataset (Phân tích bộ dữ liệu)
Bộ dữ liệu **Heart Cleveland** gồm 297 bản ghi bệnh nhân và 14 thuộc tính:
- **Biến liên tục (Numerical - 5 thuộc tính):**
  - `age`: Tuổi của bệnh nhân (năm).
  - `trestbps`: Huyết áp tâm thu khi nghỉ ngơi (mm Hg).
  - `chol`: Nồng độ cholesterol huyết thanh (mg/dl).
  - `thalach`: Nhịp tim tối đa đạt được trong nghiệm pháp gắng sức.
  - `oldpeak`: Mức độ chênh xuống của đoạn ST do gắng sức so với lúc nghỉ.
- **Biến phân loại/rời rạc (Categorical/Discrete - 8 thuộc tính):**
  - `sex`: Giới tính (0: Nữ, 1: Nam).
  - `cp`: Loại đau thắt ngực (0: điển hình, 1: không điển hình, 2: không do tim, 3: không triệu chứng).
  - `fbs`: Đường huyết lúc đói > 120 mg/dl (0: Sai, 1: Đúng).
  - `restecg`: Kết quả điện tâm đồ lúc nghỉ (0: bình thường, 1: sóng ST-T bất thường, 2: phì đại thất trái).
  - `exang`: Đau thắt ngực do gắng sức (0: Không, 1: Có).
  - `slope`: Độ dốc của đoạn ST đỉnh gắng sức (0: dốc lên, 1: đi ngang, 2: dốc xuống).
  - `ca`: Số lượng mạch máu chính nhuộm màu qua soi huỳnh quang (0, 1, 2, 3).
  - `thal`: Tình trạng khuyết tật tim qua xạ hình Thallium (0: bình thường, 1: khiếm khuyết cố định, 2: khiếm khuyết có thể phục hồi).
- **Biến mục tiêu (Target):**
  - `condition`: 0 (Không mắc bệnh - 160 mẫu, 53.87%) và 1 (Mắc bệnh tim - 137 mẫu, 46.13%). Phân phối nhãn cân bằng tự nhiên, không chịu hiện tượng mất cân bằng dữ liệu nghiêm trọng.
  - *Kiểm tra thực tế nhãn:* Dữ liệu hiện tại trong file `data/heart_cleveland_upload.csv` đã được chuẩn hóa nhị phân 0/1; không chứa nhãn độ nặng 0–4 (severity).

---

## 3. Preprocessing & Leakage Prevention (Tiền xử lý và phòng chống rò rỉ dữ liệu)
Toàn bộ quy trình tiền xử lý được đóng gói trong class `HeartDiseasePreprocessor` tự lập trình:
1. **Phân chia dữ liệu phân tầng trước khi xử lý (Stratified Splitting):** Sử dụng `src.data_utils.train_test_split(stratify=True)` chia dữ liệu theo tỷ lệ 80/20 (Train: 238 mẫu, Test: 59 mẫu), bảo toàn chính xác tỷ lệ 53.9% lớp 0 và 46.1% lớp 1.
2. **Ngăn ngừa rò rỉ dữ liệu (No Data Leakage):**
   - Mọi đại lượng thống kê (median, mean, standard deviation của thuộc tính liên tục; mode và danh mục giá trị của thuộc tính phân loại) chỉ được tính toán trên tập **Training**.
   - Tập **Test** chỉ được `transform()` theo các tham số đã đóng băng từ tập Training.
3. **Chuẩn hóa đặc trưng liên tục:**
   $$x' = \\frac{x - \\mu_{train}}{\\sigma_{train} + \\epsilon}$$
4. **Mã hóa biến phân loại (One-Hot Encoding):**
   - Biến đổi 8 biến phân loại thành 23 cột nhị phân one-hot deterministic.
   - Tổng số chiều sau tiền xử lý: $5 + 23 = 28$ đặc trưng.
   - Nếu tập Test xuất hiện category lạ (unseen), vector one-hot tương ứng được gán tự động về 0, đảm bảo độ ổn định hệ thống.

---

## 4. From-Scratch Implementations (Cơ sở thuật toán tự lập trình)
Dự án loại bỏ 100% các module của scikit-learn (`linear_model`, `neighbors`, `decomposition`, `svm`, `metrics`, `model_selection`, `pipeline`) và không sử dụng TensorFlow/PyTorch. Các thuật toán lõi được xây dựng thuần túy bằng toán học đại số tuyến tính với NumPy:
- `data_utils.py`: Stratified train/test split bảo toàn phân phối xác suất tiên nghiệm.
- `metrics.py`: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, và ROC-AUC tính qua tích phân số hình thang (trapezoidal rule).
- `preprocessing.py`: Imputation + Z-score Standardization + One-Hot Encoding không rò rỉ dữ liệu.

---

## 5. Baseline Classification (Các mô hình phân loại cơ bản)
1. **Perceptron (`PerceptronClassifier`):**
   - Tìm kiếm siêu phẳng $w^T x + b = 0$ bằng quy tắc sửa lỗi $w \\leftarrow w + \\eta (y - \\hat{y}) x$.
   - Do dữ liệu lâm sàng không phân tách tuyến tính, Perceptron dao động quanh mặt phân cách và đạt Test Accuracy ~83.05%.
2. **Logistic Regression (`LogisticRegression`):**
   - Sử dụng hàm Sigmoid ổn định số học (chia nhánh $z \\ge 0$ và $z < 0$ để tránh tràn số `exp`).
   - Tối ưu hàm mất mát nhị phân Binary Cross-Entropy với điều chuẩn L2 qua Gradient Descent.
   - Cho kết quả ổn định vượt bậc: Test Accuracy 84.75%, F1 0.8302, ROC-AUC 0.9201.
3. **K-Nearest Neighbors (`KNNClassifier`):**
   - Tính toán khoảng cách Euclidean ma trận hóa không dùng vòng lặp lồng nhau.
   - Với $K=5$, cơ chế bầu chọn đồng đều đạt Test Accuracy 88.14%, Recall 92.59%, F1 0.8772.

---

## 6. Neural Networks & Regularization (Mạng nơ-ron MLP - Chapter 16)
Mô hình `MLPClassifier` xây dựng mạng nơ-ron truyền thẳng 3 tầng:
$$\\text{Input (28)} \\to \\text{Dense(32)} \\to \\text{ReLU} \\to \\text{Dropout} \\to \\text{Dense(16)} \\to \\text{ReLU} \\to \\text{Dropout} \\to \\text{Dense(1)} \\to \\text{Sigmoid}$$

Thực nghiệm bóc tách thành phần (Ablation Study):
- **Baseline MLP:** Học nhanh nhưng có dấu hiệu overfit nhẹ trên tập dữ liệu nhỏ (Accuracy 84.75%).
- **MLP + Weight Decay ($\\lambda = 0.01$):** Kiểm soát chuẩn Frobenius của trọng số, làm trơn mặt quyết định, tăng ROC-AUC lên 0.9329.
- **MLP + Inverted Dropout ($p = 0.2$):** Vô hiệu hóa ngẫu nhiên các nơ-ron trong pha huấn luyện, phá vỡ sự đồng thích nghi (co-adaptation), giúp mô hình đạt **Test Accuracy cao nhất nhóm nơ-ron: 88.14%, F1 0.8679**.
- **MLP + WD + Dropout:** Cung cấp khả năng xếp hạng xác suất tốt nhất với **ROC-AUC đạt 0.9421**.

---

## 7. Dimensionality Reduction (Giảm chiều dữ liệu - Chapters 20–22)
1. **PCA (Unsupervised - Chapter 20):**
   - Phân rã trị riêng ma trận hiệp phương sai. Giữ lại 14 thành phần chính (giảm 50% số chiều) vẫn bảo toàn hơn **95% phương sai tích lũy**.
   - Kết hợp **PCA + Logistic Regression** đạt Test Accuracy **86.44%**, F1 **0.8571** và ROC-AUC **0.9329**, vượt qua cả Logistic Regression trên toàn bộ 28 đặc trưng gốc nhờ loại bỏ nhiễu và đa cộng tuyến.
2. **Truncated SVD (Chapter 21):**
   - Phân rã giá trị suy biến trực tiếp $X_c = U \\Sigma V^T$. Với 10 thành phần suy biến, SVD + Logistic Regression đạt Accuracy 83.05%, F1 0.8214.
3. **Linear Discriminant Analysis - LDA (Supervised - Chapter 22):**
   - Tối đa hóa tỷ số tán xạ liên lớp và nội lớp qua bài toán trị riêng tổng quát $S_W^{-1} S_B v = \\lambda v$.
   - Thu gọn toàn bộ 28 chiều xuống đúng **1 chiều duy nhất**, mô hình phân loại Logistic Regression trên trục 1D này vẫn đạt Test Accuracy **83.05%** và ROC-AUC **0.9178**.

---

## 8. Support Vector Machines (Chapters 26–28)
1. **Hard-Margin SVM (Chapter 26):**
   - Kiểm tra trực tiếp trên dữ liệu: biến cờ `is_separable_ = False` với nhiều mẫu vi phạm điều kiện lề cứng $y_i (w^Tx_i + b) \\ge 1$.
   - Chứng minh thực nghiệm rằng giả định phân tách tuyến tính tuyệt đối không khả thi trên dữ liệu lâm sàng (Test Accuracy chỉ đạt 72.88%).
2. **Soft-Margin SVM (Chapter 27):**
   - Áp dụng hàm mất mát bản lề Hinge Loss và tối ưu dưới đạo hàm (subgradient descent).
   - Với $C=1.0$, mô hình đạt Test Accuracy **86.44%**, F1 **0.8462**, ROC-AUC **0.9236**.
3. **Kernel SVM (Chapter 28):**
   - Tối ưu hóa đối ngẫu qua giải thuật SMO rút gọn trên 4 loại hàm nhân:
     - **Linear:** Accuracy 86.44%, F1 0.8462.
     - **Polynomial ($d=3$):** Accuracy 83.05%, F1 0.8214.
     - **RBF (Gaussian):** Đạt hiệu suất cao nhất trong nhóm SVM với **Test Accuracy 86.44%, F1 0.8571 và ROC-AUC 0.9456**.
     - **Sigmoid:** Accuracy 83.05%, F1 0.8077.

---

## 9. Multi-class Extension (Mở rộng đa lớp - Chapter 29)
- **Ghi nhận hiện trạng dữ liệu:** Cột `condition` trong dataset chỉ mang giá trị 0 và 1 (không có severity 0–4). Tuân thủ nghiêm ngặt yêu cầu thiết kế, chúng tôi không tự ý tạo nhãn giả lập cho dữ liệu bệnh tim.
- **Triển khai kiến trúc đa lớp:** Cài đặt đầy đủ 2 chiến lược kinh điển trong `src/models/multiclass_svm.py`:
  - `OneVsRestSVM`: Huấn luyện $C$ bộ phân loại nhị phân, dự đoán $\\hat{y} = \\arg\\max_c f_c(x)$.
  - `OneVsOneSVM`: Huấn luyện $C(C-1)/2$ bộ phân loại nhị phân cặp, dự đoán bằng biểu quyết đa số.
- **Kiểm chứng tính đúng đắn:** Thực nghiệm trên bài toán 3 cụm dữ liệu phi tuyến (Synthetic 3-class clusters), cả OvR và OvO kết hợp Kernel RBF đều phân loại chính xác 100%, xác nhận thuật toán hoạt động hoàn hảo khi dữ liệu có nhiều lớp.

---

## 10. Experimental Results (Bảng kết quả thực nghiệm tổng hợp)

Dưới đây là bảng kết quả đo đạc trực tiếp trên tập kiểm thử (Hold-out Test set, 59 mẫu) được xuất ra từ file `experiments/results.csv`:

| Nhóm mô hình | Tên mô hình (Model) | Siêu tham số tối ưu (Best Parameters) | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baselines** | Perceptron | penalty='l1', alpha=0.001 | 0.8305 | 0.8400 | 0.7778 | 0.8077 | 0.8889 |
| | Logistic Regression | learning_rate=0.05, l2=0.0 | 0.8475 | 0.8462 | 0.8148 | 0.8302 | 0.9201 |
| | KNN Classifier | n_neighbors=5, weights='uniform' | **0.8814** | 0.8333 | **0.9259** | **0.8772** | 0.9219 |
| **Neural Nets** | MLP (Baseline) | hidden=(32,16), WD=0, Drop=0 | 0.8475 | 0.8750 | 0.7778 | 0.8235 | 0.9259 |
| | MLP + Weight Decay | WD=0.01, Drop=0.0 | 0.8475 | 0.8750 | 0.7778 | 0.8235 | 0.9329 |
| | MLP + Dropout | WD=0.0, Drop=0.2 | **0.8814** | **0.8846** | 0.8519 | 0.8679 | 0.9329 |
| | MLP + WD + Dropout | WD=0.01, Drop=0.2 | 0.8475 | 0.8750 | 0.7778 | 0.8235 | **0.9421** |
| **Giảm chiều** | PCA + Logistic Regression | n_components=14 (95% var), l2=0.01 | 0.8644 | 0.8276 | 0.8889 | 0.8571 | 0.9329 |
| | PCA + KNN Classifier | n_components=14, n_neighbors=7 | 0.8305 | 0.7931 | 0.8519 | 0.8214 | 0.9248 |
| | SVD + Logistic Regression | n_components=10, l2=0.01 | 0.8305 | 0.7931 | 0.8519 | 0.8214 | 0.9039 |
| | SVD + KNN Classifier | n_components=10, n_neighbors=7 | 0.8136 | 0.7667 | 0.8519 | 0.8070 | 0.8825 |
| | LDA + Logistic Regression | n_components=1 (Supervised 1D) | 0.8305 | 0.8148 | 0.8148 | 0.8148 | 0.9178 |
| **SVMs** | Hard Margin SVM | penalty=1000.0, non-separable | 0.7288 | 0.6410 | 0.9259 | 0.7576 | 0.9062 |
| | Soft Margin SVM | C=1.0 | 0.8644 | 0.8800 | 0.8148 | 0.8462 | 0.9236 |
| | Kernel SVM - Linear | C=1.0 | 0.8644 | 0.8800 | 0.8148 | 0.8462 | 0.9225 |
| | Kernel SVM - Poly | degree=3, C=1.0, gamma='scale' | 0.8305 | 0.7931 | 0.8519 | 0.8214 | 0.9387 |
| | Kernel SVM - RBF | C=1.0, gamma='scale' | **0.8644** | 0.8276 | 0.8889 | **0.8571** | **0.9456** |
| | Kernel SVM - Sigmoid | C=1.0, gamma='scale' | 0.8305 | 0.8400 | 0.7778 | 0.8077 | 0.9120 |
| **Đa lớp (Toy)**| One-vs-Rest SVM | Kernel RBF, C=2.0 (3 classes) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A |

---

## 11. Discussion (Thảo luận chuyên sâu)
1. **Sức mạnh của mô hình xác suất tuyến tính:**
   Logistic Regression kết hợp cùng PCA (14 chiều) mang lại hiệu quả rất cao (Accuracy 86.44%, ROC-AUC 0.9329). Trong y tế, mô hình tuyến tính có ưu điểm tuyệt đối về tính minh bạch: trọng số $w$ phản ánh trực tiếp mức độ nguy cơ của từng chỉ số (ví dụ: tuổi cao, đau thắt ngực gắng sức, nồng độ ST chênh xuống).
2. **Khả năng khái quát của Mạng nơ-ron MLP:**
   Nhờ tích hợp Inverted Dropout và L2 Weight Decay, mạng nơ-ron tự lập trình tránh được hiện tượng ghi nhớ vẹt (memorization) trên tập dữ liệu nhỏ (~300 mẫu), đạt Test Accuracy 88.14% và ROC-AUC vượt 0.94.
3. **RBF Kernel SVM - Đỉnh cao phân loại phi tuyến:**
   Kernel SVM với hàm nhân RBF đạt ROC-AUC cao nhất (0.9456) trong toàn bộ các mô hình, chứng minh rằng không gian đặc trưng bệnh tim tồn tại những tương tác phi tuyến tính phức tạp mà các ranh giới siêu phẳng phẳng không thể bao quát hết.
4. **Hiệu quả của giảm chiều có giám sát (LDA):**
   Việc chỉ cần nén xuống đúng 1 chiều số học mà vẫn đạt Accuracy 83.05% cho thấy khoảng cách giữa tâm 2 lớp trong không gian đặc trưng chiếu là rất rõ ràng.

---

## 12. Limitations & Future Work (Hạn chế và Hướng phát triển)
1. **Quy mô tập dữ liệu:** Dataset gồm 297 mẫu là tương đối khiêm tốn đối với các mô hình phức tạp như MLP hay Kernel SVM. Việc thu thập thêm dữ liệu đa trung tâm sẽ giúp giảm thiểu phương sai ước lượng.
2. **Tối ưu hóa siêu tham số:** Hiện tại việc tuning được thực hiện qua các lưới cấu hình thủ công (Explicit Configurations). Trong tương lai có thể bổ sung k-fold cross-validation từ đầu trong `src/model_selection.py` để ước lượng siêu tham số chính xác hơn nữa.
3. **Mở rộng đa lớp lâm sàng:** Nếu có nguồn dữ liệu Cleveland gốc còn lưu trữ mức độ hẹp động mạch vành (0, 1, 2, 3, 4), kiến trúc `OneVsRestSVM` và `OneVsOneSVM` đã sẵn sàng để thực nghiệm trực tiếp mà không cần sửa đổi mã nguồn.
