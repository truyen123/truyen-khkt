#!/bin/bash
# Chạy kiosk qua Cage (Wayland kiosk compositor)
# Ưu điểm: nhẹ, ổn định, chạy Chromium toàn màn hình tốt trên Pi 3

export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

CHROME="$(command -v chromium || command -v chromium-browser)"
if [ -z "$CHROME" ]; then
    echo "Chromium không tìm thấy" >&2
    exit 1
fi

# Chờ Flask khởi động
for i in $(seq 1 30); do
    if curl -sf -o /dev/null http://127.0.0.1:5000/api/kiosk; then
        break
    fi
    sleep 1
done

# Chạy Chromium qua Cage
# Cage sẽ hiển thị Chromium toàn màn hình
exec cage "$CHROME" \
    --kiosk \
    --disable-gpu \
    --disable-gpu-compositing \
    --noerrdialogs \
    --disable-infobars \
    --incognito \
    --disable-features=Translate,MediaRouter \
    --no-first-run \
    --no-default-browser-check \
    --disable-session-crashed-bubble \
    --check-for-update-interval=31536000 \
    --user-data-dir=/tmp/studyguard-chrome \
    http://127.0.0.1:5000/kiosk
