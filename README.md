# IT Salary Analysis

## Giới thiệu

Dự án này nhằm phân tích các yếu tố ảnh hưởng đến mức lương trong lĩnh vực Khoa học dữ liệu và Công nghệ thông tin. Việc phân tích được thực hiện bằng Python kết hợp với các mô hình học máy.

## Các bước thực hiện

* Tiền xử lý dữ liệu: làm sạch dữ liệu, mã hóa biến phân loại và chuẩn hóa dữ liệu.
* Phân tích khám phá dữ liệu (EDA): trực quan hóa và tìm mối quan hệ giữa các biến.
* Xây dựng mô hình: sử dụng Linear Regression, Random Forest và XGBoost.
* Đánh giá mô hình: sử dụng chỉ số R² để so sánh hiệu suất.

## Hướng dẫn chạy chương trình

```bash
python main.py
```

## Nguồn dữ liệu

https://www.kaggle.com/datasets/sazidthe1/data-science-salaries

## Công nghệ sử dụng

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

## Kết quả

Mô hình XGBoost cho kết quả tốt nhất, tuy nhiên độ chính xác chưa cao, cho thấy còn nhiều yếu tố ảnh hưởng đến mức lương chưa được đưa vào mô hình.
