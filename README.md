
# Nhận diện chữ viết tay sử dụng CNN và ResNet

## Giới thiệu
Dự án này tập trung vào việc phát triển hệ thống nhận diện chữ viết tay bằng **Mạng Nơ-ron Tích Chập (CNN)** kết hợp với **Mạng Nơ-ron Residual (ResNet)**. Mục tiêu là xây dựng một mô hình mạnh mẽ có khả năng nhận diện cả chữ cái và chữ số từ các hình ảnh đầu vào viết tay.

Giải pháp giải quyết các thách thức như sự biến thể phong cách, nhiễu ảnh và biến dạng thông qua việc áp dụng các kỹ thuật học sâu hiện đại.

## Các thành viên trong nhóm
- **Tống Đức Duy** - 3121411043
- **Dương Văn Sìnl** - 3121411182
- **Lê Trung Kiên** - 3121411110
- **Lê Bùi Minh Khoa** - 3121411103
- **Giảng viên hướng dẫn:** Ts. Trịnh Tiến Đạt

## Mục tiêu dự án
- Thiết kế và triển khai một kiến trúc kết hợp CNN-ResNet để cải thiện độ chính xác nhận diện chữ viết tay.
- Huấn luyện mô hình trên các bộ dữ liệu đa dạng, bao gồm MNIST và bộ dữ liệu chữ viết tay A-Z.
- Giải quyết các vấn đề học sâu phổ biến như gradient biến mất và overfitting bằng cách sử dụng các kết nối bỏ qua (skip connections) và các kỹ thuật điều chuẩn.

## Bộ dữ liệu
- **MNIST:** Bao gồm 60,000 hình ảnh huấn luyện và 10,000 hình ảnh kiểm tra của các chữ số (0–9), mỗi hình ảnh có kích thước 28x28 pixel.
- **Bộ dữ liệu chữ viết tay A-Z:** Gồm 372,451 mẫu chữ cái viết tay in hoa.

Cả hai bộ dữ liệu được tiền xử lý:
1. Thay đổi kích thước hình ảnh thành 32x32 pixel.
2. Chuẩn hóa giá trị pixel về phạm vi [0, 1].
3. Kết hợp thành một bộ dữ liệu duy nhất để huấn luyện và kiểm tra.

## Kiến trúc mô hình
- **Lớp Tích Chập (Convolutional Layers):** Trích xuất các đặc trưng không gian từ hình ảnh đầu vào.
- **Khối Residual (Residual Blocks):** Thêm các kết nối bỏ qua để giảm thiểu vấn đề gradient biến mất.
- **Chuẩn hóa theo lô (Batch Normalization):** Giúp ổn định và tăng tốc quá trình huấn luyện.
- **Hàm kích hoạt (ReLU):** Thêm tính phi tuyến cho mô hình.
- **Toàn cầu trung bình pooling (Global Average Pooling):** Giảm các chiều không gian trước khi phân loại.
- **Các lớp kết nối đầy đủ (Fully Connected Layers):** Xuất ra xác suất lớp cho 36 lớp (A-Z và 0-9).

## Huấn luyện
- **Framework:** TensorFlow và Keras.
- **Tối ưu hóa:** Adam với độ suy giảm học tập theo hàm mũ.
- **Hàm mất mát:** Entropy phân loại (Categorical Cross-Entropy).
- **Tăng cường dữ liệu (Augmentation):** Áp dụng các phép quay, dịch chuyển và tỉ lệ ngẫu nhiên để cải thiện độ chính xác của mô hình.

## Kết quả
Mô hình đạt được độ chính xác cao trên cả bộ dữ liệu kiểm tra và bộ dữ liệu xác thực. Đánh giá bao gồm:
1. Các chỉ số mất mát và độ chính xác qua các epoch.
2. Phân tích độ tương đồng cosine đối với các embedding được tạo ra bởi mô hình.

## Hướng dẫn sử dụng
### Yêu cầu hệ thống
- Python 3.8 hoặc phiên bản cao hơn.
- Các thư viện yêu cầu: TensorFlow, Keras, NumPy, OpenCV, Matplotlib, scikit-learn.

### Thiết lập
1. Clone kho mã nguồn:
   ```bash
   git clone https://github.com/your_username/handwriting-recognition.git
   cd handwriting-recognition
   ```
2. Cài đặt các phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

### Huấn luyện mô hình
Chạy script huấn luyện:
```bash
python train_ocr_model.py --az a_z_handwritten_data.csv --model handwriting.model
```

### Chạy mô hình với hình ảnh trong thư mục `images` và đánh giá kết quả
```bash
python ocr_handwriting.py --model handwriting.model --image images/umbc_address.png
```

## Công việc tương lai
- Mở rộng mô hình để nhận diện chữ viết tay trong chuỗi.
- Tối ưu hóa kiến trúc cho các ứng dụng thời gian thực trên các thiết bị di động và nhúng.

## Tài liệu tham khảo
1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [Tài liệu TensorFlow](https://www.tensorflow.org/)
3. [Tài liệu Keras](https://keras.io/)
4. Bộ dữ liệu MNIST: [Trang của Yann LeCun](http://yann.lecun.com/exdb/mnist/)
5. Bộ dữ liệu chữ viết tay A-Z: [Kaggle Link](https://www.kaggle.com/)

## Lời cảm ơn
Chúng tôi xin cảm ơn giảng viên **Ts. Trịnh Tiến Đạt** đã hướng dẫn và hỗ trợ trong suốt dự án. Chúng tôi cũng cảm ơn sự đóng góp ý kiến từ các bạn đồng nghiệp, giúp hoàn thiện phương pháp và triển khai mô hình.

## Tác giả
- **Dương Văn Sìnl**  
  Sinh viên tại Đại học Sài Gòn, Khoa Công nghệ Thông tin  
  Liên hệ: tinbalon3@gmail.com
``` 

