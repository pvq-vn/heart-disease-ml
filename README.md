# Heart Disease ML Project

Dự án Machine Learning phân loại bệnh tim dựa trên tập dữ liệu Heart Cleveland. Dự án này áp dụng các mô hình phân loại: Logistic Regression, Perceptron, KNN Classifier và sử dụng PCA để giảm chiều dữ liệu.

## Cấu trúc thư mục

- `data/`: Chứa file dữ liệu (ví dụ: `heart_cleveland_upload.csv`).
- `src/`: Mã nguồn bao gồm tiền xử lý dữ liệu (`preprocessing.py`), tính toán độ đo (`metrics.py`), và các mô hình machine learning (`models/`).
- `experiments/`: Chứa kết quả chạy mô hình (`results.csv`) và biểu đồ so sánh (`figures/`).
- `run_experiments.py`: Script chính để chạy toàn bộ quá trình huấn luyện và đánh giá.

## Cách chạy

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy file experiment:
   ```bash
   python run_experiments.py
   ```

## Kết quả
Kết quả chi tiết được lưu trong `experiments/results.csv` và biểu đồ so sánh ở `experiments/figures/`.
