# Phát Hiện Đối Tượng Với EfficientDet

Repository này chứa triển khai của mô hình EfficientDet cho việc phát hiện đối tượng.

## Cài Đặt và Thiết Lập

1. Clone repository:
```bash
git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git
```

2. Chạy dự đoán:
```bash
python predict.py
```

## Xem Kết Quả Huấn Luyện

Để xem kết quả huấn luyện bằng TensorBoard:

```bash
tensorboard --logdir=20250523-063143
```

Sau khi chạy lệnh trên, TensorBoard sẽ khởi động và bạn có thể xem các chỉ số huấn luyện bằng cách mở trình duyệt web và truy cập:
```
http://localhost:6006
```

Các log huấn luyện bao gồm:
- Các chỉ số loss
- Loss phân loại
- Loss hồi quy 