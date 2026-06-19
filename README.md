# 🇻🇳 Nhận Diện Tiền Tệ Việt Nam

## � Thành Viên Thực Hiện

| MSSV | Họ và Tên |
|------|-----------|
| 24521725 | Đoàn Nguyễn Minh Thư |
| 24520980 | Huỳnh Nguyễn Hoài Lộc |
| 24521871 | Trần Thanh Trúc |

## �📋 Mô Tả Dự Án

Dự án này phát triển hệ thống **nhận diện tiền giấy Việt Nam** bằng cách sử dụng kỹ thuật học sâu. Hệ thống có khả năng:
- Xác định **mệnh giá tiền** (200đ, 500đ, 1,000đ, ..., 500,000đ)
- Phân biệt **mặt trước** và **mặt sau** của tờ tiền
- Hiển thị thông tin chi tiết về tiền nhân vật, công trình lịch sử trên tờ tiền
- Cung cấp giao diện web thân thiện qua **Gradio**

## 🎯 Các Phương Pháp Được Sử Dụng

### 1. **Deep Learning (CNN)**
- **Model**: ResNet18 Fine-tuning
- **Framework**: PyTorch
- **Notebook**: `CNN/CNN.ipynb`
- Đạt độ chính xác cao trên tập test

### 2. **Machine Learning (Classical)**
- **Phương pháp**: HOG (Histogram of Oriented Gradients) + Color Features + SVM
- **Notebook**: `ml/HOG+Color+SVM.ipynb`
- Cung cấp giải pháp thay thế cho những trường hợp tài nguyên hạn chế

### 3. **Feature Detection**
- **Phương pháp**: ORB (Oriented FAST and Rotated BRIEF)
- **Notebook**: `orb/orb.ipynb`
- Kỹ thuật phát hiện đặc điểm bất biến tỷ lệ

## 📂 Cấu Trúc Thư Mục

```
├── CNN/                          # Notebook CNN với ResNet18
│   └── CNN.ipynb
├── ml/                           # Machine Learning approach
│   ├── HOG+Color+SVM.ipynb       # SVM classifier
│   ├── train_ml.py               # Script huấn luyện
│   └── evaluate_ml.py            # Đánh giá mô hình
├── orb/                          # ORB feature detection
│   └── orb.ipynb
├── data_processing/              # Xử lý dữ liệu
│   ├── preprocess.py             # Tiền xử lý ảnh
│   ├── augment_data.py           # Tăng cường dữ liệu
│   └── result_check_data.ipynb   # Kiểm tra dữ liệu
├── dataset/                      # Dữ liệu huấn luyện/kiểm tra
│   └── DATASET/
│       ├── train/                # Tập huấn luyện
│       ├── val/                  # Tập validation
│       └── test/                 # Tập test
└── Demo/                         # Ứng dụng demo
    ├── demo_gradio.py            # Giao diện web
    ├── best_model.pth            # Mô hình đã huấn luyện
    └── mapping.json              # Thông tin mệnh giá tiền
```

## 🚀 Bắt Đầu Nhanh

### Yêu Cầu
```bash
pip install torch torchvision gradio pillow opencv-python scikit-image scikit-learn
```

### Chạy Demo
```bash
cd Demo
python demo_gradio.py
```

Giao diện Gradio sẽ mở trên `http://127.0.0.1:7860`

### Debug
Bật mode debug để xem chi tiết prediction:
```bash
DEBUG_DEMO=1 python demo_gradio.py
```

## 📊 Dataset

Dataset bao gồm **12 mệnh giá tiền**:
- 200 VND, 500 VND, 1,000 VND, 2,000 VND
- 5,000 VND, 10,000 VND, 20,000 VND, 50,000 VND
- 100,000 VND, 200,000 VND, 500,000 VND

Mỗi mệnh giá có **2 mặt**: Mặt trước (truoc) và Mặt sau (sau)

##  Ghi Chú

- Mô hình đã huấn luyện được lưu tại `Demo/best_model.pth`
- Thông tin chi tiết về các mệnh giá tiền được lưu trong `Demo/mapping.json`
- Dữ liệu được tiền xử lý và tăng cường để cải thiện độ chính xác
