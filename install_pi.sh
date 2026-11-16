#!/bin/bash
# StudyGuard - cai dat tren Raspberry Pi 3B/3B+ (Raspberry Pi OS 64-bit)
# Chay:  bash install_pi.sh
set -e
cd "$(dirname "$0")"
APP_DIR="$PWD"

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 portaudio19-dev
sudo apt-get install -y flac espeak-ng ffmpeg v4l-utils chromium unclutter cage
sudo apt-get install -y fonts-dejavu-core fonts-noto-core

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements-pi.txt psutil

mkdir -p models
dl() { [ -s "models/$1" ] || curl -fsSL -o "models/$1" "$2"; }
dl pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
dl hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
dl efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

# ---- Dich vu web chay nen ----
sudo tee /etc/systemd/system/studyguard.service >/dev/null <<EOF
[Unit]
Description=StudyGuard AI
After=network-online.target sound.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=PYTHONUNBUFFERED=1
ExecStart=$APP_DIR/.venv/bin/python app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now studyguard

# ---- Kiosk: systemd user service (dung Cage + Chromium) ----
chmod +x kiosk-cage.sh
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/studyguard-kiosk.service <<EOF
[Unit]
Description=StudyGuard Kiosk (Cage)
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/kiosk-cage.sh
Restart=always
RestartSec=5
TimeoutStopSec=5
Environment=XDG_RUNTIME_DIR=/run/user/%U
Environment=WAYLAND_DISPLAY=wayland-0

[Install]
WantedBy=default.target
EOF

# File autostart: bat service user khi user dang nhap vao desktop.
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/studyguard-kiosk.desktop <<EOF
[Desktop Entry]
Type=Application
Name=StudyGuard Kiosk
Comment=Mo giao dien StudyGuard
Exec=systemctl --user start studyguard-kiosk.service
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

# Linger: service user chay ke ca khi khong co phien dang nhap.
sudo loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user daemon-reload 2>/dev/null || true
systemctl --user enable --now studyguard-kiosk.service 2>/dev/null || true

IP="$(hostname -I | awk '{print $1}')"
echo "Xong. Khoi dong lai Pi de man hinh tu mo giao dien."
echo "Web quan ly: http://${IP}:5000  |  Kiosk: http://${IP}:5000/kiosk"
echo "Kiem tra kiosk: systemctl --user status studyguard-kiosk"
