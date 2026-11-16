#!/usr/bin/env python3
"""
Test Chromium + Webcam trên Pi.
Mở lên trang đơn giản có video từ webcam và nút bấm.
Chạy: python3 test_kiosk.py
Sau đó mở Chromium: http://127.0.0.1:5001
"""
import cv2
import threading
import time
from flask import Flask, render_template_string

app = Flask(__name__)

# Biến toàn cục
camera = None
current_frame = None
lock = threading.Lock()

def read_camera():
    """Đọc khung hình từ webcam liên tục."""
    global current_frame, camera
    while True:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1)
            continue
        
        ret, frame = camera.read()
        if not ret:
            print("[Camera] Lỗi đọc khung hình")
            camera.release()
            camera = None
            time.sleep(1)
            continue
        
        # Resize để tránh quá nặng
        frame = cv2.resize(frame, (640, 480))
        
        # Lưu khung hình
        with lock:
            current_frame = frame
        
        time.sleep(0.033)  # ~30fps

def generate_jpeg():
    """Generator MJPEG từ webcam."""
    while True:
        with lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            ret, buf = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
            frame_bytes = buf.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
               + frame_bytes + b'\r\n')
        
        time.sleep(0.033)

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Test Kiosk</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background: #222;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            h1 {
                font-size: 2em;
                margin: 10px 0;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 2px solid #0f0;
                margin: 20px 0;
            }
            button {
                padding: 20px 40px;
                font-size: 1.5em;
                background: #0f0;
                color: #000;
                border: none;
                cursor: pointer;
                margin: 10px;
                border-radius: 5px;
            }
            button:hover {
                background: #0a0;
            }
            .info {
                background: #333;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Test Kiosk - Webcam OK</h1>
        <div class="info">
            <p>Nếu thấy hình webcam bên dưới = webcam hoạt động</p>
            <p>Nếu thấy trang này trên Chromium Pi = kiosk hoạt động</p>
        </div>
        <img src="/video" alt="Webcam Stream" style="width: 100%; max-width: 640px;">
        <br>
        <button onclick="alert('Nút bấm hoạt động!')">Kiểm tra nút</button>
        <button onclick="location.reload()">Làm tươi</button>
        <div class="info" id="time"></div>
    </body>
    <script>
        function updateTime() {
            document.getElementById('time').innerText = new Date().toLocaleTimeString();
        }
        setInterval(updateTime, 1000);
        updateTime();
    </script>
    </html>
    '''
    return render_template_string(html)

@app.route('/video')
def video():
    """Phát luồng MJPEG."""
    return app.response_class(
        generate_jpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # Khởi động thread đọc camera
    camera_thread = threading.Thread(target=read_camera, daemon=True)
    camera_thread.start()
    
    print("=" * 60)
    print("Test Kiosk đang chạy!")
    print("Web:      http://127.0.0.1:5001")
    print("Trên Pi:  http://192.168.1.13:5001")
    print("=" * 60)
    print("Chạy Chromium kiosk trên Pi bằng:")
    print('  XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \\')
    print('  chromium --kiosk --ozone-platform=wayland \\')
    print('    --disable-gpu --user-data-dir=/tmp/test \\')
    print('    http://127.0.0.1:5001')
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
