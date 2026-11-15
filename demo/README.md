# ResNet-101 WSSS Web Demo

Ứng dụng web demo sử dụng mô hình **ResNet-101** đã huấn luyện trên Pascal VOC 2012 để:
- Chọn ảnh, kéo thả ảnh mới (hoặc dán ảnh bằng `Ctrl+V`).
- Nhận diện đa nhãn (Multi-label Classification) trên 20 lớp Pascal VOC.
- Sinh bản đồ kích hoạt **Vanilla CAM** / **LayerCAM** (có tùy chọn Multi-scale + Flip TTA).
- Tạo pseudo-mask phân đoạn bằng **Vanilla CAM**, **CAM + DenseCRF**, và **CAM + SAM (ViT-B)**.

## Cấu trúc thư mục

- `app.py`: Backend Flask phục vụ giao diện web và API suy luận (`/api/status`, `/api/samples`, `/api/predict`).
- `templates/index.html`: Giao diện web.
- `static/style.css`: CSS giao diện.
- `static/app.js`: Xử lý kéo thả, chọn ảnh, gọi API và hiển thị trực quan.

## Cách chạy

Từ thư mục gốc dự án (`Do_an/`):

```powershell
python demo/app.py
```

Hoặc tùy chỉnh cổng / checkpoint:

```powershell
python demo/app.py --host 127.0.0.1 --port 5000
```

Sau đó mở trình duyệt tại `http://127.0.0.1:5000`.

