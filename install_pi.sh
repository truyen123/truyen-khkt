#!/bin/bash
# StudyGuard - cai dat tren Raspberry Pi 3B (Raspberry Pi OS Bookworm 64-bit, Python 3.11 mac dinh)
# Chay:  bash install_pi.sh
set -e
cd "$(dirname "$0")"
APP_DIR="$PWD"

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 portaudio19-dev flac espeak-ng \
    ffmpeg v4l-utils chromium-browser unclutter

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements-pi.txt psutil

mkdir -p models
dl() { [ -s "models/$1" ] || curl -fsSL -o "models/$1" "$2"; }
dl pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
dl hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
dl efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

# Dich vu chay nen
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

# Tu mo man hinh kiosk khi vao desktop
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/studyguard-kiosk.desktop <<EOF
[Desktop Entry]
Type=Application
Name=StudyGuard Kiosk
Exec=bash -c 'sleep 8; unclutter -idle 1 & $(command -v chromium-browser || command -v chromium) --kiosk --noerrdialogs --disable-infobars --incognito http://127.0.0.1:5000/kiosk'
EOF

echo "Xong. Khoi dong lai Pi de man hinh tu mo giao dien. Web quan ly: http://$(hostname -I | awk '{print $1}'):5000"
