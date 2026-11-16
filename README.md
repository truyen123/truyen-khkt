# StudyGuard AI — Giám sát tư thế & hỗ trợ học tập

Nguyên mẫu: camera USB + Raspberry Pi 3 Model B (1GB RAM) hoặc PC.
Xử lý ảnh **tại thiết bị**; chatbot gọi **Gemini API** qua Internet.

## Chức năng

| Khối | Triển khai |
|---|---|
| Thu nhận hình ảnh | `open_camera()` 320×240, quét camera thật trên web |
| Xử lý tư thế | MediaPipe Pose lite + `vision.py`: góc lưng (vai–hông), độ cúi đầu chuẩn hoá theo vai, trung bình trượt + hysteresis |
| Làm việc riêng (điện thoại) | YOLOv8n (PC/Pi 4+) hoặc MediaPipe EfficientDet (Pi 3, nhẹ); xác nhận sau 30s; lưu clip AVI 2fps + 4s pre-roll |
| Quản lý phiên học | hiện diện trước camera, idle-grace 60s, thời gian tính đến giây |
| Chatbot | văn bản + giọng nói (wake-word "trợ lý ảo"), Gemini, TTS tiếng Việt Edge Neural (cần Internet), fallback pyttsx3 |
| Màn hình thiết bị | `/kiosk`: 4 tab — Giám sát, Trò chuyện, Cài đặt, Hệ thống (chạy kiosk toàn màn hình) |
| Trang quản lý | `/`: live + trạng thái + test cam/mic/loa + cài đặt + thống kê + log + bằng chứng |

## Cài đặt trên Raspberry Pi (khuyến nghị dùng script)

```bash
git clone https://github.com/truyen123/truyen-khkt.git
cd truyen-khkt
bash install_pi.sh
sudo reboot
```

Script sẽ: cài gói hệ thống, tạo venv, tải 3 model MediaPipe, đăng ký service
`studyguard` (chạy nền) và service `studyguard-kiosk` (mở Chromium toàn màn hình vào `/kiosk`).

Kiểm tra:
```bash
sudo systemctl status studyguard
systemctl --user status studyguard-kiosk
journalctl -u studyguard -f
```

## Cài đặt thủ công

```bash
sudo apt update && sudo apt install -y python3-venv libportaudio2 portaudio19-dev flac espeak-ng chromium
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt psutil

mkdir -p models
curl -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task -o models/pose_landmarker_lite.task
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o models/hand_landmarker.task
curl -L https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite -o models/efficientdet_lite0.tflite

python app.py
```

PC: dùng `requirements.txt` (có thêm YOLO) thay cho `requirements-pi.txt`.

## Địa chỉ truy cập

- Trang quản lý: `http://<IP>:5000`
- Màn hình kiosk: `http://<IP>:5000/kiosk`

Server phục vụ `0.0.0.0` và **chưa có đăng nhập**. Chỉ chạy trong mạng tin cậy, không forward cổng ra Internet.

## Chế độ thiết bị RAM thấp (Pi 3, 1GB)

- YOLO tự tắt khi RAM < 1.5GB. Khi đó dùng MediaPipe EfficientDet (`efficientdet_lite0.tflite`) làm bộ nhận diện điện thoại.
- Bật YOLO thủ công: `pip install ultralytics && export STUDY_ENABLE_YOLO=1`.
- **MediaPipe chưa có bản cho Python 3.13.** Nếu Pi dùng Python 3.13, `pip` sẽ bỏ qua dòng `mediapipe` và phần nhận diện tư thế/tay sẽ không chạy (app vẫn chạy, chỉ stream camera thô). Muốn đầy đủ vision, hãy dùng Python 3.11 hoặc 3.12 trên Pi.

## Chạy lâu dài và tự phục hồi

- **Camera:** vòng xử lý chạy nền (`vision_loop()`), không phụ thuộc người mở trình duyệt. Tự mở lại camera sau 5 lần đọc lỗi.
- **MJPEG:** nhiều client xem cùng lúc được; mỗi client chỉ nhận khung hình mới nhất.
- **MediaPipe:** lỗi khởi tạo → app vẫn stream hình + báo AI chưa sẵn sàng.
- **Trợ lý thoại:** worker tự restart; nhận dạng giọng + Edge TTS cần mạng.
- **Dữ liệu:** log giữ tối đa 10.000 dòng; chat 500 dòng; clip evidence xoay vòng (100 file / 512MB).
- **Database:** SQLite WAL + busy_timeout 5s; recovery tự tắt session/event chưa đóng nếu app crash.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py study_store.py voice_assistant.py vision.py
```

## Quyền riêng tư

- `data.sqlite3`: phiên hiện diện, sự kiện phone, cảnh báo tư thế, nhiệm vụ, hội thoại.
- Không lưu video liên tục; chỉ lưu clip khi đã xác nhận phone (thư mục `evidence/`).
- Nhật ký thoại lưu text, không lưu audio.
