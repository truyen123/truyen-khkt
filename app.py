"""StudyGuard AI — server giam sat tu the + dien thoai + tro ly ao.

Chay duoc tren PC va Raspberry Pi 3 Model B (Python 3.10 - 3.13).
Xu ly anh tai thiet bi (MediaPipe Pose + YOLOv8 CPU), chatbot qua Gemini API.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import logging
import math
import socket
import time
from collections import deque
from pathlib import Path
from threading import Condition, RLock, Lock, Thread

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from study_store import StudyStore, local_timestamp
from vision import AngleSmoother, StatusSmoother, analyze_posture, phone_held
from voice_assistant import VoiceAssistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
# Chong cache trinh duyet cu (nguyen nhan "van thay ban cu")
@app.after_request
def _no_cache(resp):
    if request.path in ("/", "/kiosk", "/video_feed") or request.path.endswith((".html", ".js", ".css")):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
    return resp
cv2.setNumThreads(1)

# ---------------------------------------------------------------- config
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
DB_PATH = Path(os.environ.get("STUDY_DB_PATH", BASE_DIR / "data.sqlite3"))
EVIDENCE_DIR = Path(os.environ.get("STUDY_EVIDENCE_DIR", BASE_DIR / "evidence"))
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_enabled": True,
    "microphone_device": None,
    "microphone_enabled": False,
    "speaker_device": None,
    "speaker_enabled": False,
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-flash",
    "tts_voice": "vi-VN-NamAnhNeural",
    "voice_alerts": True,  # doc canh bao bang loa khi gu lung / cui dau / dung dien thoai
    # Nguong tu the (chinh duoc tren web)
    "back_angle_deg": 25,
    "head_dx_ratio": 0.45,
    "alert_after_seconds": 5,
    "good_after_seconds": 2,
    "smooth_window": 8,
    # Dien thoai
    "phone_confirm_seconds": 30,
    "phone_miss_grace": 6,
    "phone_end_grace": 6,
    "phone_conf": 0.55,
    "wrist_dist_px": 65,
    "phone_every_seconds": 2.0,
}
device_lock = RLock()
camera_io_lock = Lock()
mediapipe_lock = Lock()  # MediaPipe not thread-safe


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg.update({k: v for k, v in loaded.items() if k in DEFAULT_CONFIG})
        except (OSError, ValueError) as e:
            logger.warning("config.json loi, dung mac dinh: %s", e)
    return cfg


def save_config(cfg):
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CONFIG_PATH)


device_config = load_config()

# ---------------------------------------------------------------- state
camera_active = bool(device_config["camera_enabled"])
cap = None
camera_error = ""
camera_last_frame_monotonic = None
camera_last_frame_jpeg = None
system_status = "Dang khoi dong..."
status_type = "success"  # success | warn | danger
assistant_status = {"state": "Chua khoi dong", "error": "", "last_question": "", "last_answer": ""}
vision_error = ""
phone_detector_name = "chua khoi dong"
last_voice_alert = 0.0
VOICE_ALERT_COOLDOWN = 60.0

study_store = StudyStore(DB_PATH)
EVIDENCE_FPS = 2
EVIDENCE_PRE_ROLL = 4
CAMERA_INTERVAL = 0.2
YOLO_RETRY_SECONDS = 300
EVIDENCE_MAX_FILES = 100
EVIDENCE_MAX_BYTES = 512 * 1024 * 1024


# ---- log cua server (console) -> do vao bang System log tren web ----
class _DbLogHandler(logging.Handler):
    _NOISY_PATHS = ("GET /get_status", "GET /api/logs", "GET /api/system", "GET /api/dashboard")

    def emit(self, record):
        try:
            msg = self.format(record)[:2000]
            # Access log cua werkzeug bo qua hoan toan: ghi DB tren the SD moi request rat hai the.
            if record.name == "werkzeug":
                return
            lvl = record.levelname if record.levelname in ("INFO", "WARN", "WARNING", "ERROR", "CRITICAL") else "INFO"
            if lvl == "WARNING":
                lvl = "WARN"
            study_store.add_log(lvl, record.name or "server", msg)
        except Exception:
            self.handleError(record)


_db_handler = _DbLogHandler()
_db_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_db_handler)


def open_camera(index, w=320, h=240):
    # Open/read timeout hints are supported by FFmpeg/GStreamer; ignored by some camera backends.
    params = []
    if os.name == "nt":
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    try:
        if os.name != "nt":
            # V4L2 tren Linux khong ho tro tham so timeout (gay loi "unsupported parameter").
            c = cv2.VideoCapture(index, cv2.CAP_V4L2)
        elif params:
            c = cv2.VideoCapture(index, cv2.CAP_ANY, params)
        else:
            c = cv2.VideoCapture(index)
    except (TypeError, cv2.error):
        c = cv2.VideoCapture(index)
    if not c.isOpened():
        c.release()
        raise RuntimeError(f"Khong mo duoc camera {index}.")
    c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # tranh khung hinh cu bi don trong bo dem
    return c


def scan_cameras(max_index=4):
    found = []
    for i in range(max_index + 1):
        c = None
        if cap is not None and camera_active and i == device_config["camera_index"]:
            # Camera dang dung: khong mo lai lan 2 (Linux tu choi / lam giat hinh).
            found.append({"id": i, "name": f"Camera {i} (dang dung)"})
            continue
        if os.name != "nt" and not Path(f"/dev/video{i}").exists():
            continue
        try:
            c = cv2.VideoCapture(i)
            if c.isOpened():
                ok, _ = c.read()
                if ok:
                    found.append({"id": i, "name": f"Camera {i}" + (" (dang dung)" if i == device_config["camera_index"] else "")})
        except cv2.error as e:
            logger.debug("Camera probe %s failed: %s", i, e)
        finally:
            if c is not None:
                c.release()
    if not found:
        found = [{"id": device_config["camera_index"], "name": f"Camera {device_config['camera_index']} (chua phat hien)"}]
    return found


def list_audio():
    import sounddevice as sd
    devs = sd.query_devices()
    import re as _re
    mics = [{"id": "default", "name": "Micro mac dinh he thong"}]
    spks = [{"id": "default", "name": "Loa mac dinh he thong"}]
    cand_in, cand_out = [], []
    for i, d in enumerate(devs):
        name = (d.get("name") or "").strip()
        if not name or "Mapper" in name or "Primary Sound" in name:
            continue  # an driver ao / trung gian
        base = _re.sub(r"\s*\(\d+-\s*", " (", name).strip().lower()
        if d["max_input_channels"] > 0:
            cand_in.append((base, i, name))
        if d["max_output_channels"] > 0:
            cand_out.append((base, i, name))

    def _dedupe(cands):
        # uu tien ten day du (WASAPI/DirectSound thuong dai hon MME bi cat 31 ky tu)
        cands.sort(key=lambda x: -len(x[2]))
        kept = []
        for base, i, name in cands:
            dup = False
            for kb, _, kn in kept:
                short = base[:24]
                if short and (base.startswith(kb[:24]) or kb.startswith(short)):
                    dup = True
                    break
            if not dup:
                kept.append((base, i, name))
        kept.sort(key=lambda x: x[2].lower())
        return [{"id": i, "name": name} for _, i, name in kept]

    mics += _dedupe(cand_in)
    spks += _dedupe(cand_out)
    return mics, spks


def validate_audio(device_id, is_input):
    import sounddevice as sd
    kind = "input" if is_input else "output"
    d = sd.query_devices(device_id, kind)
    key = "max_input_channels" if is_input else "max_output_channels"
    if d[key] < 1:
        raise ValueError(f"Thiet bi am thanh khong co kenh {kind} phu hop.")


if camera_active:
    try:
        cap = open_camera(device_config["camera_index"])
    except RuntimeError as e:
        camera_error = str(e)
        logger.warning(camera_error)
        cap = None


def apply_settings(changes):
    global cap, camera_active, camera_error
    with device_lock:
        unknown = set(changes) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Truong cai dat khong hop le: {', '.join(sorted(unknown))}")
        cand = dict(device_config)
        cand.update(changes)
        if type(cand["camera_index"]) is not int or cand["camera_index"] < 0:
            raise ValueError("Chi so camera phai la so nguyen >= 0.")
        for k in ("camera_enabled", "microphone_enabled", "speaker_enabled", "voice_alerts"):
            if type(cand[k]) is not bool:
                raise ValueError(f"{k} phai la true/false.")
        for k, is_in in (("microphone_device", True), ("speaker_device", False)):
            if cand[k] is not None and (type(cand[k]) is not int or cand[k] < 0):
                raise ValueError(f"{k} phai la so nguyen >= 0 hoac null.")
        if not isinstance(cand.get("gemini_api_key", ""), str) or len(cand["gemini_api_key"]) > 200:
            raise ValueError("gemini_api_key phai la chuoi (toi da 200 ky tu).")
        if not isinstance(cand.get("gemini_model", ""), str) or not cand["gemini_model"].strip() or len(cand["gemini_model"]) > 80:
            raise ValueError("gemini_model khong hop le.")
        if cand.get("tts_voice") not in ("vi-VN-NamAnhNeural", "vi-VN-HoaiMyNeural",
                                         "vi-VN-ThanhBinhNeural", "vi-VN-MyAnNeural"):
            raise ValueError("tts_voice khong hop le.")
        for k in ("back_angle_deg", "head_dx_ratio", "alert_after_seconds",
                  "good_after_seconds", "smooth_window", "phone_confirm_seconds",
                  "phone_miss_grace", "phone_end_grace", "phone_conf",
                  "wrist_dist_px", "phone_every_seconds"):
            if not isinstance(cand[k], (int, float)) or cand[k] <= 0:
                raise ValueError(f"{k} phai la so duong.")
        new_cap = None
        try:
            if cand["camera_enabled"] and (not camera_active or cap is None
                                           or cand["camera_index"] != device_config["camera_index"]):
                new_cap = open_camera(cand["camera_index"])
            if cand["microphone_enabled"]:
                validate_audio(cand["microphone_device"], True)
            if cand["speaker_enabled"]:
                validate_audio(cand["speaker_device"], False)
            save_config(cand)
        except Exception:
            if new_cap is not None:
                new_cap.release()
            raise
        old = cap
        with camera_io_lock:
            cap = new_cap if (cand["camera_enabled"] and new_cap is not None) else (old if cand["camera_enabled"] else None)
            if old is not None and old is not cap:
                old.release()
        device_config.update(cand)
        was = camera_active
        camera_active = cand["camera_enabled"]
        camera_error = ""
        if was and not camera_active:
            study_store.close_active_session()
        return dict(device_config)


def current_config():
    with device_lock:
        return dict(device_config)


def on_assistant_status(s):
    with device_lock:
        assistant_status.update(s)


def study_context():
    d = study_store.dashboard()
    acts = "; ".join(t["description"] for t in d["tasks"][:5]) or "chua ghi chu"
    evs = "; ".join(e["details"] for e in d["events"][:3]) or "khong co"
    return (f"da phat hien hien dien hoc khoang {d['study_seconds'] // 60} phut; "
            f"ghi chu: {acts}; su kien gan day: {evs}")


assistant = None
assistant_lock = Lock()

def get_assistant():
    global assistant
    if assistant is None:
        with assistant_lock:
            if assistant is None:
                try:
                    assistant = VoiceAssistant(get_config=current_config, get_context=study_context,
                                               on_message=study_store.add_assistant_message, on_status=on_assistant_status)
                except Exception as e:
                    logger.exception("Voice assistant init failed: %s", e)
                    assistant = None
    return assistant

# Lazy YOLO (khong chan khoi dong server)
yolo_model = None
yolo_error = ""
yolo_retry_after = 0.0
yolo_lock = Lock()


def _low_memory_device():
    if os.environ.get("STUDY_ENABLE_YOLO") == "1" or os.name == "nt":
        return False
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) < 1_500_000  # kB; Pi 3 has about 1 GB
    except (OSError, ValueError, IndexError):
        return False
    return False


def get_yolo():
    global yolo_model, yolo_error, yolo_retry_after
    if yolo_model is not None:
        return yolo_model
    if _low_memory_device():
        yolo_error = "YOLO tat tren thiet bi RAM thap de uu tien MediaPipe."
        return None
    now = time.monotonic()
    if now < yolo_retry_after:
        return None
    with yolo_lock:
        if yolo_model is not None:
            return yolo_model
        if time.monotonic() < yolo_retry_after:
            return None
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(BASE_DIR / "yolov8n.pt"))
            try:
                import torch
                torch.set_num_threads(1)
            except ImportError:
                pass
            yolo_error = ""
            logger.info("Da tai YOLOv8n.")
        except Exception as e:
            yolo_error = str(e)
            yolo_retry_after = time.monotonic() + YOLO_RETRY_SECONDS
            logger.exception("Khong tai duoc YOLO; se thu lai sau %ss", YOLO_RETRY_SECONDS)
            yolo_model = None
        return yolo_model


POSE_MODEL = BASE_DIR / "models" / "pose_landmarker_lite.task"
HAND_MODEL = BASE_DIR / "models" / "hand_landmarker.task"


def _enforce_evidence_quota():
    files = []
    try:
        for path in EVIDENCE_DIR.iterdir():
            if path.is_file() and path.suffix.lower() in (".avi", ".mp4", ".jpg"):
                try:
                    files.append((path.stat().st_mtime, path, path.stat().st_size))
                except OSError:
                    continue
    except OSError as e:
        logger.warning("Khong doc duoc evidence directory: %s", e)
        return
    total = sum(size for _, _, size in files)
    files.sort()
    while files and (len(files) >= EVIDENCE_MAX_FILES or total >= EVIDENCE_MAX_BYTES):
        _, path, size = files.pop(0)
        try:
            path.unlink()
            total -= size
        except OSError as e:
            logger.warning("Khong xoa duoc clip cu %s: %s", path.name, e)
            break


DETECTOR_MODEL = BASE_DIR / "models" / "efficientdet_lite0.tflite"
_mp_detector = None
_mp_detector_failed = False


def get_mp_detector(mp_mod):
    """Bo nhan dien vat the nhe (MediaPipe) thay YOLO tren Pi 1GB RAM."""
    global _mp_detector, _mp_detector_failed
    if _mp_detector is not None or _mp_detector_failed or mp_mod is None:
        return _mp_detector
    if not DETECTOR_MODEL.is_file():
        _mp_detector_failed = True
        logger.warning("Thieu %s: khong nhan dien duoc dien thoai tren thiet bi nay.", DETECTOR_MODEL.name)
        return None
    try:
        with mediapipe_lock:
            _mp_detector = mp_mod.tasks.vision.ObjectDetector.create_from_options(
                mp_mod.tasks.vision.ObjectDetectorOptions(
                    base_options=mp_mod.tasks.BaseOptions(model_asset_path=str(DETECTOR_MODEL)),
                    running_mode=mp_mod.tasks.vision.RunningMode.VIDEO,
                    max_results=3, score_threshold=0.3, category_allowlist=["cell phone"]))
    except Exception:
        _mp_detector_failed = True
        logger.exception("Khong khoi tao duoc MediaPipe ObjectDetector")
    return _mp_detector


def vision_model_error():
    miss = [p.name for p in (POSE_MODEL, HAND_MODEL) if not p.is_file()]
    if miss:
        return "Thieu model MediaPipe trong models/: " + ", ".join(miss) + ". Xem README.md."
    return ""


# ---------------------------------------------------------------- video loop
frame_cond = Condition()
frame_seq = 0


def _cache_frame(buf):
    """Luu khung hinh moi nhat va danh thuc cac client dang xem /video_feed."""
    global camera_last_frame_jpeg, frame_seq
    with frame_cond:
        camera_last_frame_jpeg = buf.tobytes()
        frame_seq += 1
        frame_cond.notify_all()


def announce(text):
    """Doc canh bao qua loa (nen, co cooldown), khong chan vong xu ly hinh."""
    global last_voice_alert
    cfg = current_config()
    now = time.monotonic()
    if not (cfg.get("voice_alerts") and cfg.get("speaker_enabled")):
        return
    if now - last_voice_alert < VOICE_ALERT_COOLDOWN:
        return
    asst = get_assistant()
    if asst is None:
        return
    last_voice_alert = now
    Thread(target=asst._speak, args=(text,), daemon=True, name="voice-alert").start()


def vision_loop():
    """Vong xu ly nen: doc camera, phan tich tu the/dien thoai. Chay doc lap voi trinh duyet."""
    global system_status, status_type, cap, camera_error, camera_last_frame_monotonic, vision_error, phone_detector_name
    err = vision_model_error()
    degraded = bool(err)  # thieu model -> van stream hinh, tam dung AI
    if err:
        logger.warning("Vision degraded: %s", err)
        system_status = "Camera OK - dang cho model AI (tai 2 file .task theo README)."
        status_type = "warn"
    pre_roll = deque(maxlen=EVIDENCE_PRE_ROLL * EVIDENCE_FPS)
    last_pre = 0.0
    last_ev = 0.0
    last_phone = 0.0
    cand_start = cand_at = last_seen = None
    phone_event = None
    ev_path = None
    writer = None
    posture_alert_id = None
    posture_start = 0.0

    def finish_phone(now_ts):
        nonlocal phone_event, ev_path, writer
        if writer is not None:
            writer.release()
            writer = None
        if phone_event is not None:
            study_store.finish_phone_event(
                phone_event, now_ts,
                ev_path.name if ev_path is not None and ev_path.exists() else None)
            phone_event = None
            ev_path = None

    pose = hands = mp_mod = None
    if not degraded:
        try:
            with mediapipe_lock:
                import mediapipe as mp
                mp_mod = mp
                running = mp.tasks.vision.RunningMode.VIDEO
                pose = mp.tasks.vision.PoseLandmarker.create_from_options(
                    mp.tasks.vision.PoseLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL)),
                        running_mode=running, num_poses=1,
                        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5))
                hands = mp.tasks.vision.HandLandmarker.create_from_options(
                    mp.tasks.vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
                        running_mode=running, num_hands=2,
                        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5))
        except Exception as e:
            degraded = True
            vision_error = f"Khong khoi tao duoc MediaPipe: {e}"
            logger.exception("Vision init failed; falling back to raw camera stream")
            for task in (hands, pose):
                try:
                    if task is not None:
                        with mediapipe_lock:
                            task.close()
                except Exception:
                    pass
            pose = hands = mp_mod = None
            system_status = "Camera OK - AI tam thoi khong san sang."
            status_type = "warn"

    smoother = StatusSmoother(bad_seconds=float(current_config()["alert_after_seconds"]),
                              good_seconds=float(current_config()["good_after_seconds"]))
    ang_sm = AngleSmoother(window=int(current_config()["smooth_window"]))
    try:
        fail_count = 0
        last_reopen = 0.0
        while True:
            t0 = time.monotonic()
            with device_lock:
                active = cap if camera_active else None
                cfg = dict(device_config)
            ok, frame = False, None
            if active is not None:
                # Khong giu device_lock khi doc: read() co the cho den 3s va se treo moi API.
                with camera_io_lock:
                    if active is cap and camera_active:
                        ok, frame = active.read()
            now = time.monotonic()
            if not ok:
                fail_count += 1
                # Tu mo lai camera neu bi app khac chiem tam thoi (VD Zalo/Meet vua tat)
                if camera_active and fail_count >= 5 and now - last_reopen > 5:
                    last_reopen = now
                    fail_count = 0
                    try:
                        # Thu tu khoa giong apply_settings (device_lock -> camera_io_lock) de khong deadlock.
                        with device_lock:
                            with camera_io_lock:
                                old = cap
                                cap = None
                                if old is not None:
                                    try:
                                        old.release()
                                    except Exception as e:
                                        logger.warning("release camera failed: %s", e)
                                cap = open_camera(cfg["camera_index"])
                                camera_error = ""
                        study_store.add_log("INFO", "camera", "Tu mo lai camera thanh cong.")
                    except Exception as e:
                        with device_lock:
                            camera_error = str(e)
                            cap = None
                        study_store.add_log("ERROR", "camera", f"Mo lai camera that bai: {e}")
                        logger.warning("camera reopen failed: %s", e)
                study_store.update_presence(False, now)
                finish_phone(local_timestamp())
                cand_start = cand_at = last_seen = None
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                if camera_active:
                    system_status = camera_error or "Khong doc duoc hinh anh tu camera."
                    status_type = "danger"
                else:
                    fail_count = 0
                    system_status = "Camera da tat."
                    status_type = "success"
                _, buf = cv2.imencode(".jpg", blank)
                _cache_frame(buf)
                time.sleep(CAMERA_INTERVAL)
                continue

            fail_count = 0
            camera_last_frame_monotonic = time.monotonic()
            camera_error = ""
            
            # Detect corrupted frame (V4L2 issue on RPi): check if extreme low variation (uniform color)
            if frame is not None and frame.size > 0:
                # Uniform color frame = corrupted. Check std dev across channels
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    std_val = gray.std()
                    # Normal frame has std > 20; corrupted (uniform color) has std < 5
                    if std_val < 5:
                        logger.debug("Skipping corrupted frame (std=%.1f)", std_val)
                        blank = np.zeros((240, 320, 3), dtype=np.uint8)
                        _, buf = cv2.imencode(".jpg", blank)
                        _cache_frame(buf)
                        time.sleep(CAMERA_INTERVAL)
                        continue
                except Exception:
                    pass
            
            if frame.shape[1] != 320 or frame.shape[0] != 240:
                frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

            if degraded or pose is None:
                # Chua co model: stream hinh tho + van dem gio hien dien
                label, ltype = ("Camera OK - dang cho model AI (tai 2 file .task).", "warn")
                system_status, status_type = label, ltype
                study_store.update_presence(True, now)
                cv2.putText(frame, "Cho model AI...", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                _cache_frame(buf)
                time.sleep(max(0, CAMERA_INTERVAL - (time.monotonic() - t0)))
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mp_img = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
            ts = int(now * 1000)
            with mediapipe_lock:
                pose_res = pose.detect_for_video(mp_img, ts)
            rgb.flags.writeable = True
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]

            lm = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
            ana = analyze_posture(lm, w, h, cfg)
            back_sm = ang_sm.update(ana["back_angle"]) if ana["present"] else 0.0
            raw = ana["status"] if ana["present"] else "good"
            stable = smoother.update(raw, now)
            study_store.update_presence(ana["present"], now)

            if stable in ("hunch", "head_down") and posture_alert_id is None:
                posture_alert_id = study_store.start_posture_alert(
                    stable, "%s (goc lung %.1f)" % (ana["label"], back_sm))
                posture_start = now
                study_store.add_log("WARN", "vision", "Tu the: %s (goc lung %.0f)." % (stable, back_sm))
                announce("Bạn đang gù lưng, hãy ngồi thẳng lên nhé." if stable == "hunch"
                         else "Bạn đang cúi đầu quá thấp, hãy ngẩng đầu lên nhé.")
            elif stable == "good" and posture_alert_id is not None:
                study_store.finish_posture_alert(posture_alert_id, now - posture_start)
                posture_alert_id = None

            if stable == "good":
                if ana["present"]:
                    label, ltype = ("Dang hoc - tu the chuan", "success")
                else:
                    label, ltype = ("Khong co nguoi truoc camera", "success")
            elif stable == "hunch":
                label = "Canh bao: Gu lung (%.0f > %s) - %ds" % (back_sm, cfg["back_angle_deg"], int(now - posture_start))
                ltype = "danger"
            elif stable == "head_down":
                label = "Canh bao: Cui dau (%.2f) - %ds" % (ana["head_dx_norm"], int(now - posture_start))
                ltype = "danger"
            else:
                label, ltype = ("Dang xac minh tu the...", "warn")

            if lm and ana["present"]:
                try:
                    def xs(i):
                        return (int(lm[i].x * w), int(lm[i].y * h))
                    left = 7 if lm[7].visibility > lm[8].visibility else 8
                    ls = 11 if lm[11].visibility > lm[12].visibility else 12
                    cv2.line(img, xs(left), xs(ls), (0, 255, 255), 2)
                    cv2.line(img, xs(12), xs(24), (255, 0, 0), 2)
                    cv2.line(img, xs(11), xs(23), (255, 0, 0), 2)
                except (IndexError, AttributeError):
                    pass

            phone_now = False
            if now - last_phone >= float(cfg["phone_every_seconds"]):
                last_phone = now
                try:
                    boxes = []  # (x1, y1, x2, y2) cua dien thoai
                    yolo = get_yolo()
                    if yolo is not None:
                        phone_detector_name = "YOLOv8n"
                        res = yolo.predict(frame, imgsz=320, conf=float(cfg["phone_conf"]),
                                           max_det=3, device="cpu", verbose=False)
                        for r in res:
                            for b in r.boxes:
                                if yolo.names[int(b.cls[0])] == "cell phone":
                                    boxes.append(tuple(map(int, b.xyxy[0])))
                    else:
                        det = get_mp_detector(mp_mod)
                        if det is not None:
                            phone_detector_name = "MediaPipe EfficientDet (nhe cho Pi)"
                            with mediapipe_lock:
                                dres = det.detect_for_video(mp_img, ts)
                            for d_ in dres.detections:
                                cat = d_.categories[0]
                                if cat.category_name == "cell phone" and cat.score >= float(cfg["phone_conf"]) * 0.8:
                                    bb = d_.bounding_box
                                    boxes.append((bb.origin_x, bb.origin_y,
                                                  bb.origin_x + bb.width, bb.origin_y + bb.height))
                        else:
                            phone_detector_name = "tat (thieu model)"
                    if boxes:
                        with mediapipe_lock:
                            hr = hands.detect_for_video(mp_img, ts)
                        wrists = [(int(hand[0].x * w), int(hand[0].y * h))
                                  for hand in (hr.hand_landmarks or [])]
                        for x1, y1, x2, y2 in boxes:
                            held = phone_held(((x1 + x2) / 2, (y1 + y2) / 2), wrists,
                                              threshold_px=float(cfg["wrist_dist_px"]))
                            cv2.rectangle(img, (x1, y1), (x2, y2),
                                          (0, 0, 255) if held else (0, 165, 255), 2)
                            if held:
                                phone_now = True
                except Exception as e:
                    logger.exception("Loi nhan dien dien thoai: %s", e)

            if phone_now:
                last_seen = now
                if cand_start is None:
                    cand_start, cand_at = now, local_timestamp()
            elif last_seen is not None and now - last_seen > float(cfg["phone_miss_grace"]):
                if phone_event is None:
                    cand_start = cand_at = last_seen = None

            if phone_event is None and cand_start is not None:
                wait = float(cfg["phone_confirm_seconds"])
                left = wait - (now - cand_start)
                if left <= 0:
                    phone_event = study_store.start_phone_event(
                        cand_at, "Nghi ngo su dung dien thoai (tay gan dien thoai on dinh)")
                    study_store.add_log("WARN", "vision", "Xac nhan su dung dien thoai (event %s)." % phone_event)
                    announce("Bạn đang dùng điện thoại, hãy tập trung học bài nhé.")
                    _enforce_evidence_quota()
                    fn = "phone_%s_%s.avi" % (time.strftime("%Y%m%d_%H%M%S"), phone_event)
                    ev_path = EVIDENCE_DIR / fn
                    try:
                        writer = cv2.VideoWriter(str(ev_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                                 EVIDENCE_FPS, (w, h))
                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            ev_path = None
                    except cv2.error:
                        logger.exception("Khong tao duoc evidence writer")
                        writer = None
                        ev_path = None
                    if writer is not None:
                        for _, fr in pre_roll:
                            dec = cv2.imdecode(np.frombuffer(fr, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if dec is not None:
                                writer.write(dec)
                else:
                    label = "Dang xac minh dien thoai... %ds" % int(left)
                    ltype = "warn"

            if phone_event is not None:
                held_for = int(now - cand_start) if cand_start else 0
                label = "Da xac nhan su dung dien thoai (%ds)" % held_for
                ltype = "danger"
                if last_seen is not None and now - last_seen >= float(cfg["phone_end_grace"]):
                    finish_phone(local_timestamp())
                    cand_start = cand_at = last_seen = None
                elif writer is not None and now - last_ev >= 1 / EVIDENCE_FPS:
                    writer.write(img)
                    last_ev = now

            system_status, status_type = label, ltype
            cv2.putText(img, "%.0fdeg %02ds" % (back_sm, int(now % 60)), (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
            _cache_frame(buf)
            if now - last_pre >= 1 / EVIDENCE_FPS:
                pre_roll.append((now, buf.tobytes()))
                last_pre = now
            time.sleep(max(0, CAMERA_INTERVAL - (time.monotonic() - t0)))
    finally:
        try:
            if pose is not None:
                pose.close()
        except Exception:
            pass
        try:
            if hands is not None:
                hands.close()
        except Exception:
            pass
        finish_phone(local_timestamp())
        if posture_alert_id is not None:
            study_store.finish_posture_alert(posture_alert_id, time.monotonic() - posture_start)


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/kiosk")
def kiosk():
    """Giao dien toan man hinh cho man hinh gan truc tiep vao Pi (HDMI/DSI)."""
    return render_template("kiosk-full.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG: nhieu client xem cung luc; vong xu ly chay nen, khong phu thuoc nguoi xem."""
    def stream():
        last = -1
        while True:
            with frame_cond:
                frame_cond.wait_for(lambda: frame_seq != last, timeout=5)
                data, last = camera_last_frame_jpeg, frame_seq
            if data is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"

    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


_vision_thread = None


def start_vision():
    global _vision_thread
    if _vision_thread is not None and _vision_thread.is_alive():
        return

    def runner():
        global system_status, status_type
        while True:
            try:
                vision_loop()
            except Exception:
                logger.exception("Vision loop crashed; restart in 3s")
                system_status, status_type = "Loi xu ly hinh anh, dang khoi dong lai...", "danger"
            time.sleep(3)

    _vision_thread = Thread(target=runner, name="vision", daemon=True)
    _vision_thread.start()


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


@app.route("/api/kiosk")
def api_kiosk():
    """Mot goi du lieu nho cho man hinh Pi: trang thai, thong so, hoi thoai gan day."""
    d = study_store.dashboard()
    with device_lock:
        cfg = dict(device_config)
        cam_ok = bool(camera_active and camera_last_frame_monotonic is not None
                      and time.monotonic() - camera_last_frame_monotonic < 3.0)
    info = {"ip": local_ip(), "port": 5000}
    try:
        import psutil
        info["cpu"] = psutil.cpu_percent(interval=None)
        info["ram"] = psutil.virtual_memory().percent
        temp = psutil.sensors_temperatures().get("cpu_thermal") if hasattr(psutil, "sensors_temperatures") else None
        if temp:
            info["temp"] = round(temp[0].current, 1)
    except Exception:
        pass
    return jsonify({
        "status": system_status, "type": status_type,
        "camera_ok": cam_ok, "mic_on": cfg["microphone_enabled"], "speaker_on": cfg["speaker_enabled"],
        "assistant": dict(assistant_status), "phone_detector": phone_detector_name,
        "study_text": d["study_text"], "week_text": d["week_text"],
        "phone_count": len(d["events"]), "posture_count": len(d["posture_alerts"]),
        "tasks": d["tasks"][:5], "chat": study_store.recent_messages(6),
        "system": info,
    })


@app.route("/api/vision_status")
def vision_status():
    err = vision_model_error()
    return jsonify({"ok": not err, "error": err,
                    "pose_model": POSE_MODEL.exists(), "hand_model": HAND_MODEL.exists()})


@app.route("/snapshot.jpg")
def snapshot():
    with device_lock:
        frame_data = camera_last_frame_jpeg
        last_frame = camera_last_frame_monotonic
    if frame_data is None or last_frame is None or time.monotonic() - last_frame > 3.0:
        return jsonify({"error": "Khong co khung hinh camera moi."}), 503
    return Response(frame_data, mimetype="image/jpeg")


@app.route("/get_status")
def get_status():
    with device_lock:
        cfg = dict(device_config)
        asst = dict(assistant_status)
        cam_act, cam_err = camera_active, camera_error
        last_frame = camera_last_frame_monotonic
    frame_fresh = last_frame is not None and time.monotonic() - last_frame < 3.0
    # Khong goi cap.grab() tai endpoint: frame chi doc boi video loop.
    cam_conn = bool(cam_act and cap is not None and cap.isOpened() and frame_fresh)
    spk_conn = bool(cfg.get("speaker_enabled"))
    mic_conn = asst.get("state") in ("Dang nghe", "Dang tro chuyen")
    mic_err = asst.get("error", "") if cfg["microphone_enabled"] and asst.get("error") else ""
    return jsonify({"status": system_status, "type": status_type,
                    "camera": {"enabled": cam_act, "connected": cam_conn,
                               "grab_ok": frame_fresh if cam_act else None, "error": cam_err},
                    "microphone": {"enabled": cfg["microphone_enabled"], "connected": mic_conn, "error": mic_err},
                    "speaker": {"enabled": cfg["speaker_enabled"], "connected": spk_conn, "error": ""},
                    "assistant": asst})


@app.route("/api/dashboard")
def dashboard():
    return jsonify(study_store.dashboard())


@app.route("/api/tasks", methods=["POST"])
def create_task():
    if not request.is_json:
        return jsonify({"error": "Can Content-Type application/json."}), 400
    d = request.get_json(silent=True)
    desc = d.get("description") if isinstance(d, dict) else None
    if not isinstance(desc, str) or not desc.strip() or len(desc) > 300:
        return jsonify({"error": "Mo ta 1-300 ky tu."}), 400
    return jsonify({"task": study_store.add_task(desc.strip())}), 201


@app.route("/api/tasks/<int:tid>/toggle", methods=["POST"])
def toggle_task(tid):
    t = study_store.toggle_task(tid)
    if t is None:
        return jsonify({"error": "Khong tim thay nhiem vu."}), 404
    return jsonify({"task": t})


@app.route("/api/tasks/<int:tid>", methods=["DELETE"])
def delete_task(tid):
    study_store.delete_task(tid)
    return jsonify({"ok": True})


@app.route("/api/gemini/test", methods=["POST"])
def test_gemini():
    if not request.is_json:
        return jsonify({"error": "Yêu cầu JSON."}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Dữ liệu kiểm tra không hợp lệ."}), 400
    model = body.get("model", current_config().get("gemini_model", "gemini-2.5-flash"))
    api_key = body.get("api_key", "")
    if not isinstance(model, str) or not model.strip() or len(model) > 80:
        return jsonify({"error": "Tên model phải dài 1-80 ký tự."}), 400
    if not isinstance(api_key, str) or len(api_key) > 200:
        return jsonify({"error": "API key không hợp lệ."}), 400
    cfg = current_config()
    cfg["gemini_model"] = model.strip()
    if api_key.strip():
        cfg["gemini_api_key"] = api_key.strip()
    tester = VoiceAssistant(get_config=lambda: cfg, get_context=lambda: "",
                            on_message=lambda q, a: None, on_status=lambda s: None)
    try:
        answer = tester.answer("Trả lời đúng một từ: OK")
        return jsonify({"ok": True, "model": cfg["gemini_model"],
                        "message": f"Gemini hoạt động với {cfg['gemini_model']}.",
                        "preview": answer[:80]})
    except Exception as e:
        logger.warning("Gemini config test failed for model %s: %s", cfg["gemini_model"], e)
        detail = str(e).lower()
        if "api key" in detail or "unauthenticated" in detail or "401" in detail:
            message = "API key không hợp lệ hoặc đã hết hiệu lực."
        elif "not found" in detail or "404" in detail or "model" in detail:
            message = "Không tìm thấy model này hoặc model không được API key hỗ trợ."
        elif "429" in detail or "resource_exhausted" in detail or "quota" in detail:
            message = "Gemini đã hết quota hoặc đang giới hạn tần suất; kiểm tra quota/billing."
        elif "timeout" in detail or "timed out" in detail:
            message = "Kết nối Gemini quá thời gian chờ; kiểm tra Internet rồi thử lại."
        else:
            message = "Không kết nối hoặc không gọi được Gemini; xem System log để biết chi tiết."
        return jsonify({"error": f"{message} (model: {cfg['gemini_model']})"}), 502


@app.route("/api/assistant/ask", methods=["POST"])
def ask_assistant():
    if not request.is_json:
        return jsonify({"error": "Can Content-Type application/json."}), 400
    d = request.get_json(silent=True)
    q = d.get("question") if isinstance(d, dict) else None
    if not isinstance(q, str) or not q.strip() or len(q) > 1000:
        return jsonify({"error": "Cau hoi 1-1000 ky tu."}), 400
    try:
        asst = get_assistant()
        if asst is None:
            return jsonify({"error": "Tro ly chua san sang."}), 503
        return jsonify({"answer": asst.ask(q.strip())})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Gemini failed")
        return jsonify({"error": f"Khong goi duoc tro ly: {e}"}), 502


@app.route("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename, as_attachment=False)


@app.route("/api/system", methods=["GET"])
def api_system():
    """CPU/RAM/disk + dung luong DB + so clip evidence."""
    import shutil
    info = {"platform": os.name, "db_bytes": study_store.db_size_bytes(),
            "db_text": "%.1f KB" % (study_store.db_size_bytes() / 1024)}
    try:
        du = shutil.disk_usage(str(BASE_DIR))
        info["disk_free_gb"] = round(du.free / 1e9, 1)
        info["disk_total_gb"] = round(du.total / 1e9, 1)
    except OSError:
        pass
    try:  # psutil neu co (PC); Pi khong bat buoc
        import psutil
        info["cpu_percent"] = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        info["ram_percent"] = mem.percent
        info["ram_used_mb"] = round(mem.used / 1e6)
        info["ram_total_mb"] = round(mem.total / 1e6)
        info["uptime_s"] = int(time.time() - psutil.boot_time())
    except ImportError:
        info["note"] = "cai psutil de xem CPU/RAM (pip install psutil)"
    clips = []
    try:
        clips = [f for f in EVIDENCE_DIR.iterdir() if f.is_file()]
    except OSError:
        pass
    info["evidence_files"] = len(clips)
    info["evidence_mb"] = round(sum(f.stat().st_size for f in clips if f.exists()) / 1e6, 1)
    info["yolo"] = {"loaded": yolo_model is not None, "error": yolo_error,
                    "retry_in_s": max(0, int(yolo_retry_after - time.monotonic()))}
    info["assistant"] = dict(assistant_status)
    return jsonify(info)


@app.route("/api/logs", methods=["GET"])
def api_logs():
    return jsonify({"logs": study_store.get_logs(request.args.get("limit", 200))})


@app.route("/api/logs", methods=["DELETE"])
def api_clear_logs():
    study_store.clear_logs()
    study_store.add_log("INFO", "web", "Da xoa system logs.")
    return jsonify({"ok": True})


@app.route("/api/data", methods=["DELETE"])
def api_clear_data():
    """Xoa toan bo du lieu hoc tap. ?evidence=1 de xoa ca clip."""
    want_evidence = request.args.get("evidence") == "1"
    ev = {"removed": 0, "kept": []}
    if want_evidence:
        ev = study_store.delete_evidence_files(EVIDENCE_DIR)
    study_store.clear_all_data()
    study_store.add_log("WARN", "web", "Da xoa toan bo du lieu (evidence=%s)." % want_evidence)
    return jsonify({"ok": True, "evidence": ev})


@app.route("/api/system/clear_cache", methods=["POST"])
def api_clear_cache():
    cleared = []
    for d in (BASE_DIR / "__pycache__",):
        if d.is_dir():
            import shutil as _sh
            try:
                _sh.rmtree(d)
                cleared.append(str(d))
            except OSError:
                pass
    study_store.add_log("INFO", "web", "Da xoa cache: %s" % (cleared or "khong co"))
    return jsonify({"ok": True, "cleared": cleared})


def mask_settings(cfg):
    out = dict(cfg)
    key = out.get("gemini_api_key") or ""
    out["gemini_api_key_set"] = bool(key)
    out["gemini_api_key"] = ("••••••••" + key[-4:]) if key else ""
    return out


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        if not request.is_json:
            return jsonify({"error": "Can Content-Type application/json."}), 400
        changes = request.get_json(silent=True)
        if not isinstance(changes, dict):
            return jsonify({"error": "Body phai la JSON object."}), 400
        # convert select "default" -> None for audio
        for k in ("microphone_device", "speaker_device"):
            if changes.get(k) == "default":
                changes[k] = None
        # empty key field = keep old key; "CLEAR" = xoa key
        if changes.get("gemini_api_key", None) == "":
            changes.pop("gemini_api_key")
        elif changes.get("gemini_api_key") == "CLEAR":
            changes["gemini_api_key"] = ""
        try:
            updated = apply_settings(changes)
        except (ValueError, RuntimeError, OSError) as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"settings": mask_settings(updated)})
    try:
        mics, spks = list_audio()
    except Exception as e:
        return jsonify({"error": f"Khong lay duoc thiet bi am thanh: {e}"}), 503
    return jsonify({"settings": mask_settings(current_config()),
                    "devices": {"cameras": scan_cameras(), "microphones": mics, "speakers": spks}})


@app.route("/toggle_camera", methods=["POST"])
def toggle_camera():
    try:
        with device_lock:
            updated = apply_settings({"camera_enabled": not camera_active})
        return jsonify({"active": updated["camera_enabled"]})
    except (ValueError, RuntimeError, OSError) as e:
        return jsonify({"error": str(e)}), 400


# ---- device self-tests (web-based, no extra hardware) ----
@app.route("/api/test/camera", methods=["POST"])
def test_camera():
    idx = (request.get_json(silent=True) or {}).get("camera_index", current_config()["camera_index"])
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "camera_index khong hop le."}), 400
    if camera_active and idx == current_config()["camera_index"]:
        # Camera dang do vong xu ly nen giu: dung khung hinh moi nhat thay vi mo lan 2.
        data = camera_last_frame_jpeg
        fresh = camera_last_frame_monotonic is not None and time.monotonic() - camera_last_frame_monotonic < 3.0
        if not data or not fresh:
            return jsonify({"ok": False, "error": camera_error or "Camera dang bat nhung khong co khung hinh moi."}), 400
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        ok = frame is not None
    else:
        try:
            c = open_camera(idx)
            try:
                ok, frame = c.read()
            finally:
                c.release()
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
    if not ok or frame is None:
        return jsonify({"ok": False, "error": "Mo duoc camera nhung khong doc duoc khung hinh."}), 400
    h, w = frame.shape[:2]
    return jsonify({"ok": True, "message": f"Camera {idx} OK — anh {w}x{h}.", "snapshot_url": "/snapshot.jpg"})


@app.route("/api/test/microphone", methods=["POST"])
def test_microphone():
    import sounddevice as sd
    body = request.get_json(silent=True) or {}
    dev = body.get("microphone_device", current_config()["microphone_device"])
    if dev == "default":
        dev = None
    seconds = min(max(float(body.get("seconds", 2.0)), 1.0), 5.0)
    try:
        validate_audio(dev, True)
        info = sd.query_devices(dev, "input")
        rate = int(info.get("default_samplerate") or 16000)
        rec = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32", device=dev)
        sd.wait()
        level = float(np.abs(rec).mean())
        if level < 1e-4:
            return jsonify({"ok": False, "error": "Micro khong thu duoc tin hieu (qua yen tinh hoac sai thiet bi).",
                            "level": level})
        return jsonify({"ok": True, "message": f"Micro OK — muc tin hieu {level:.4f}.", "level": level})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/test/speaker", methods=["POST"])
def test_speaker():
    import sounddevice as sd
    body = request.get_json(silent=True) or {}
    raw_dev = body.get("speaker_device", current_config()["speaker_device"])
    dev = None if raw_dev == "default" else raw_dev
    try:
        validate_audio(dev, False)
        info = sd.query_devices(dev, "output")
        rate = int(info.get("default_samplerate") or 48000)
        secs = 1.5  # keo dai de kip nghe + doi loa Bluetooth
        t = np.linspace(0, secs, int(rate * secs), endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
        # fade in/out chong "tac" loa
        n_fade = min(len(tone), int(rate * 0.05))
        fade = np.ones(len(tone), dtype=np.float32)
        fade[:n_fade] = np.linspace(0, 1, n_fade)
        fade[-n_fade:] = np.linspace(1, 0, n_fade)
        tone = tone * fade
        sd.play(tone, samplerate=rate, device=dev)
        sd.wait()
        name = info.get("name", "?")
        return jsonify({"ok": True,
                        "message": f"Da phat 660Hz / {secs}s qua loa '{name}' (rate {rate}Hz). "
                                   "Neu khong nghe: tang volume Windows + kiem tra dung loa dang phat nhac."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/test/voice", methods=["POST"])
def test_voice():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "Xin chào! Tôi là trợ lý học tập StudyGuard. Bạn muốn hỏi bài gì nào?").strip()[:300]
    voice = body.get("tts_voice") or current_config().get("tts_voice") or "vi-VN-NamAnhNeural"
    try:
        from voice_assistant import VoiceAssistant
        va = VoiceAssistant(get_config=current_config, get_context=lambda: "",
                            on_message=lambda q, a: None, on_status=lambda s: None)
        cfg = dict(current_config())
        cfg["speaker_enabled"] = True
        cfg["tts_voice"] = voice
        if body.get("speaker_device", None) not in (None, ""):
            raw = body["speaker_device"]
            cfg["speaker_device"] = None if raw == "default" else int(raw)
        va.get_config = lambda: cfg
        va._speak(text)
        return jsonify({"ok": True, "message": f"Da doc thu bang giong {voice}."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    start_vision()
    asst = get_assistant()
    if asst is not None:
        asst.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
