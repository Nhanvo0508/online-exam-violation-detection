# Online Exam Violation Detection

Hệ thống phát hiện vật thể có khả năng vi phạm quy định trong thi online
thông qua camera, kết hợp Computer Vision, Object Tracking và Machine
Learning để đánh giá mức độ rủi ro của từng sự kiện.

## 1. Cài đặt và chạy project

### Bước 1: Cài Python

Cài Python từ trang chính thức của Python và đảm bảo Python đã được thêm
vào PATH.

Kiểm tra:

``` bash
python --version
```

Nếu máy sử dụng lệnh `py` thay cho `python` thì có thể dùng:

``` bash
py --version
```

### Bước 2: Cài các thư viện

Mở Terminal tại thư mục project và chạy:

``` bash
pip install -r requirements.txt
```

Nếu máy không nhận `pip`, có thể dùng:

``` bash
python -m pip install -r requirements.txt
```

### Bước 3: Chạy project

Mở file:

``` text
main.py
```

Sau đó bấm **Run** trong PyCharm.

### Bước 4: Mở giao diện

Sau khi server khởi động, mở trình duyệt và truy cập:

``` text
http://127.0.0.1:8010/
```

Hệ thống sẽ sử dụng camera để phát hiện và theo dõi các vật thể.

------------------------------------------------------------------------

## 2. Giới thiệu

Project xây dựng một hệ thống hỗ trợ giám sát thi trực tuyến. Camera
cung cấp hình ảnh, hệ thống phát hiện các vật thể được định nghĩa là có
khả năng vi phạm, theo dõi chúng theo thời gian và tạo event để đánh giá
mức độ rủi ro.

Các thành phần chính:

-   YOLOv8: phát hiện vật thể.
-   ByteTrack: theo dõi đối tượng.
-   Preprocessing: xử lý hình ảnh và vùng quan tâm.
-   Temporal Filter: lọc detection theo thời gian.
-   Machine Learning pipeline: trích xuất đặc trưng và tính Risk Score.
-   FastAPI + WebSocket: cung cấp giao diện và cập nhật dữ liệu
    realtime.

## 3. Các đối tượng được phát hiện

Project hiện hỗ trợ các class:

-   `book` - Sách
-   `calculator` - Máy tính
-   `cheat_sheet` - Tài liệu/phao
-   `earphone` - Tai nghe
-   `notebook` - Vở/sổ
-   `phone` - Điện thoại

## 4. Kiến trúc hệ thống

``` text
Camera
  ↓
Preprocessing
  ↓
YOLOv8
  ↓
ByteTrack
  ↓
Temporal Filter
  ↓
Feature Extraction
  ↓
MobileNetV2 Embedding
  ↓
PCA
  ↓
KMeans
  ↓
Random Forest
  ↓
Logistic Regression
  ↓
Risk Score
  ↓
FastAPI + WebSocket
  ↓
Web Interface
```

## 5. Công nghệ sử dụng

### Computer Vision / Deep Learning

-   Python
-   OpenCV
-   YOLOv8
-   PyTorch
-   Torchvision
-   ByteTrack

### Machine Learning

-   MobileNetV2
-   StandardScaler
-   PCA
-   KMeans
-   Random Forest
-   Logistic Regression
-   Scikit-learn

### Backend / Web

-   FastAPI
-   Uvicorn
-   WebSocket
-   HTML / CSS / JavaScript

## 6. Cấu trúc project

``` text
online-exam-violation-detection/
├── README.md
├── requirements.txt
├── main.py
│
├── app/
│   ├── check.py
│   ├── config.py
│   ├── detector.py
│   ├── draw.py
│   ├── preprocess.py
│   ├── temporal.py
│   └── tracker.py
│
├── MachineLearning/
│   ├── pipeline_infer.py
│   ├── pipeline_train.py
│   ├── mobilenetv2_embeddings.py
│   ├── schema.py
│   ├── utils_io.py
│   ├── pca.py
│   ├── kmeans.py
│   ├── random_forest.py
│   ├── logistic_meta.py
│   ├── ml_artifacts/
│   │   ├── scaler.pkl
│   │   ├── pca.pkl
│   │   ├── kmeans.pkl
│   │   ├── rf.pkl
│   │   └── lr_meta.pkl
│   └── tools/
│
├── models/
│   └── best.pt
│
└── snapshots/
```

## 7. Vai trò các thành phần

### `main.py`

File chính điều phối toàn bộ hệ thống:

-   Camera
-   Preprocessing
-   YOLO detection
-   Tracking
-   Temporal filtering
-   Feature extraction
-   Machine Learning inference
-   Snapshot
-   FastAPI
-   WebSocket
-   Web interface

### `app/detector.py`

Sử dụng YOLOv8 để phát hiện vật thể từ camera.

### `app/tracker.py`

Sử dụng ByteTrack để theo dõi các đối tượng qua nhiều frame và duy trì
`tracking_id`.

### `app/preprocess.py`

Thực hiện các bước xử lý trước khi detection như:

-   Gray-World
-   CLAHE
-   ROI
-   Motion Gate

### `app/temporal.py`

Xử lý thông tin theo thời gian, giúp hạn chế các detection ngắn hoặc
nhiễu và tạo event.

### `MachineLearning/pipeline_infer.py`

Thực hiện pipeline Machine Learning:

``` text
Features
   ↓
StandardScaler
   ↓
PCA
   ↓
KMeans
   ↓
Random Forest
   ↓
Logistic Regression
   ↓
Risk Score
```

## 8. YOLO Confidence và Risk Score

Hai giá trị này có ý nghĩa khác nhau.

### YOLO Confidence

Confidence thể hiện mức độ tin cậy của YOLO đối với một detection.

Ví dụ:

``` text
book 0.89
```

có nghĩa YOLO đang đánh giá detection `book` với confidence 0.89.

### Risk Score

Risk Score được tính bởi pipeline Machine Learning dựa trên các đặc
trưng của event.

Risk Score được sử dụng để quyết định event có đạt ngưỡng ghi nhận vi
phạm/rủi ro của hệ thống hay không.

## 9. Machine Learning Artifacts

Pipeline inference sử dụng các artifact đã train:

``` text
scaler.pkl
pca.pkl
kmeans.pkl
rf.pkl
lr_meta.pkl
```

Các file này chứa trạng thái của các mô hình/bộ biến đổi đã được huấn
luyện và cần có khi chạy Machine Learning inference.

## 10. Dữ liệu và snapshots

Thư mục `snapshots/` chứa các ảnh được tạo trong quá trình hệ thống
chạy.

Trong quá trình phát triển, các ảnh snapshot có thể được sử dụng để kiểm
tra detection và event.


## 11. Quy trình hoạt động

1.  Camera nhận hình ảnh.
2.  Preprocessing xử lý hình ảnh.
3.  YOLOv8 phát hiện vật thể.
4.  ByteTrack theo dõi đối tượng.
5.  Temporal Filter xác định event.
6.  Hệ thống trích xuất các đặc trưng.
7.  MobileNetV2 tạo image embedding.
8.  Pipeline Machine Learning tính Risk Score.
9.  Event đạt ngưỡng được ghi nhận.
10. FastAPI/WebSocket cập nhật giao diện.
11. Snapshot và thông tin event được lưu lại.

## 12. Giao diện

Web interface cung cấp:

-   Camera stream
-   Bounding box
-   Tên vật thể
-   Confidence
-   Risk Score
-   Bảng vi phạm
-   Tracking ID
-   Thời lượng event
-   Snapshot
-   Chi tiết event

## 13. Training Machine Learning

Nếu muốn train lại pipeline Machine Learning bằng dataset phù hợp với
schema của project:

``` bash
python -m MachineLearning.pipeline_train
```

Sau khi train, các artifact được lưu trong:

``` text
MachineLearning/ml_artifacts/
```

## 14. Model

YOLO model được đặt tại:

``` text
models/best.pt
```

Đây là model được sử dụng cho object detection trong hệ thống.

## 15. Mục tiêu project

Project hướng tới việc kết hợp:

``` text
Computer Vision
+
Object Tracking
+
Temporal Analysis
+
Machine Learning
+
Realtime Web Interface
```

để xây dựng một hệ thống hỗ trợ phát hiện và đánh giá các sự kiện có khả
năng vi phạm trong thi trực tuyến.
