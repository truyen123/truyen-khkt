#!/bin/bash
# Mo giao dien kiosk StudyGuard. Dung chung cho ca labwc (Wayland) va LXDE (X11).
# Chay boi systemd user service: studyguard-kiosk.service
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-wayland}"

CHROME="$(command -v chromium || command -v chromium-browser)"
if [ -z "$CHROME" ]; then
    echo "Khong tim thay chromium. Cai: sudo apt-get install -y chromium" >&2
    sleep 5
    exit 1
fi

# Cho may chay lai 30 giay de dung web.
for i in $(seq 1 30); do
    if curl -sf -o /dev/null http://127.0.0.1:5000/api/kiosk; then break; fi
    sleep 1
done

# An con tro chuot.
unclutter -idle 1 -root >/dev/null 2>&1 &

# Wayland: --ozone-platform=wayland. X11: --ozone-platform=x11.
if [ "$XDG_SESSION_TYPE" = "x11" ]; then
    OZONE="--ozone-platform=x11"
else
    OZONE="--ozone-platform=wayland"
fi

exec "$CHROME" \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --incognito \
    --disable-features=Translate,MediaRouter \
    --no-first-run \
    --no-default-browser-check \
    --disable-session-crashed-bubble \
    --check-for-update-interval=31536000 \
    --user-data-dir=/tmp/studyguard-chrome \
    "$OZONE" \
    "http://127.0.0.1:5000/kiosk"
