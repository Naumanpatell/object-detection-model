import argparse
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

TH = {
    "calib_frames": 40,
    "sleepy_eye_ratio": 0.55,
    "happy_width_ratio": 1.10,
    "shocked_mouth_delta": 0.08,
    "shocked_brow_delta": 0.025,
    "sad_curve_delta": -0.012,
    "smooth_window": 8,
}

# baseline for --image mode (no calibration)
NEUTRAL_DEFAULTS = {
    "mouth_open": 0.010,
    "mouth_width": 0.350,
    "smile_curve": 0.015,
    "eye_open": 0.070,
    "brow": 0.130,
}

FEATURE_NAMES = ["mouth_open", "mouth_width", "smile_curve", "eye_open", "brow"]
LABELS = ["happy", "shocked", "sad", "sleepy", "neutral"]

COLOURS = {
    "happy": (0, 215, 255),
    "shocked": (0, 140, 255),
    "sad": (230, 130, 40),
    "sleepy": (190, 90, 160),
    "neutral": (190, 190, 190),
}

LM = {
    "face_left": 234, "face_right": 454,
    "lip_top": 13, "lip_bottom": 14,
    "mouth_left": 61, "mouth_right": 291,
    "l_eye_top": 159, "l_eye_bottom": 145,
    "r_eye_top": 386, "r_eye_bottom": 374,
    "l_brow": 105, "r_brow": 334,
}

BASE_DIR = Path(__file__).resolve().parent
STICKER_DIR = BASE_DIR / "stickers"
SCREENSHOT_DIR = BASE_DIR / "screenshots"
FONT = cv2.FONT_HERSHEY_SIMPLEX


# 1. DETECT
def create_face_mesh(static_image=False):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def detect_face(face_mesh, frame):
    h, w = frame.shape[:2]
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, None

    face = results.multi_face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h) for p in face.landmark], dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bbox = (int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2)))
    return pts, bbox


# 2. FEATURES
def dist(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))


def extract_features(pts):
    face_w = dist(pts, LM["face_left"], LM["face_right"])
    if face_w < 1e-6:
        return None

    lip_centre_y = (pts[LM["lip_top"], 1] + pts[LM["lip_bottom"], 1]) / 2
    corner_y = (pts[LM["mouth_left"], 1] + pts[LM["mouth_right"], 1]) / 2

    return {
        "mouth_open": dist(pts, LM["lip_top"], LM["lip_bottom"]) / face_w,
        "mouth_width": dist(pts, LM["mouth_left"], LM["mouth_right"]) / face_w,
        "smile_curve": float(lip_centre_y - corner_y) / face_w,  # positive = corners up
        "eye_open": (dist(pts, LM["l_eye_top"], LM["l_eye_bottom"])
                     + dist(pts, LM["r_eye_top"], LM["r_eye_bottom"])) / 2 / face_w,
        "brow": (dist(pts, LM["l_brow"], LM["l_eye_top"])
                 + dist(pts, LM["r_brow"], LM["r_eye_top"])) / 2 / face_w,
    }


# 3. CALIBRATE
class Calibrator:
    def __init__(self, n_frames=TH["calib_frames"]):
        self.n_frames = n_frames
        self.reset()

    def reset(self):
        self.samples = []
        self.baseline = None

    @property
    def done(self):
        return self.baseline is not None

    def add(self, features):
        if self.done:
            return False
        self.samples.append(features)
        if len(self.samples) >= self.n_frames:
            self.baseline = {k: float(np.mean([s[k] for s in self.samples])) for k in FEATURE_NAMES}
            return True
        return False


# 4. CLASSIFY
def classify_expression(f, base):
    if f["eye_open"] < TH["sleepy_eye_ratio"] * base["eye_open"]:
        return "sleepy"
    if f["mouth_width"] > TH["happy_width_ratio"] * base["mouth_width"]:
        return "happy"
    if (f["mouth_open"] - base["mouth_open"] > TH["shocked_mouth_delta"]
            or f["brow"] - base["brow"] > TH["shocked_brow_delta"]):
        return "shocked"
    if f["smile_curve"] - base["smile_curve"] < TH["sad_curve_delta"]:
        return "sad"
    return "neutral"


# 5. SMOOTH
class LabelSmoother:
    def __init__(self, window=TH["smooth_window"]):
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def update(self, label):
        self.history.append(label)
        return Counter(self.history).most_common(1)[0][0]


# 6. OVERLAY
def make_placeholder(label, size=300):
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[:, :, :3] = COLOURS[label]
    img[:, :, 3] = 210
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (255, 255, 255, 255), 6)
    text = f"{label} cat"
    scale, thick = 1.4, 3
    tw = cv2.getTextSize(text, FONT, scale, thick)[0][0]
    scale *= min(1.0, size * 0.85 / tw)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, ((size - tw) // 2, (size + th) // 2), FONT, scale,
                (20, 20, 20, 255), thick, cv2.LINE_AA)
    return img


def load_stickers():
    stickers = {}
    for label in LABELS:
        path = STICKER_DIR / f"{label}.png"
        img = None
        if path.exists():
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[stickers] {path.name} not found - using placeholder")
            img = make_placeholder(label)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        stickers[label] = img
    return stickers


def blend_rgba(frame, img, x, y):
    fh, fw = frame.shape[:2]
    h, w = img.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, fw), min(y + h, fh)
    if x1 <= x0 or y1 <= y0:
        return
    crop = img[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (alpha * crop[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)


def draw_sticker(frame, sticker, bbox):
    x1, y1, x2, y2 = bbox
    new_w = max(1, x2 - x1)
    new_h = max(1, round(sticker.shape[0] * new_w / sticker.shape[1]))
    resized = cv2.resize(sticker, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gap = int(0.1 * new_w)
    sx = x2 + gap
    if sx + new_w // 2 > frame.shape[1] and x1 - gap - new_w >= 0:
        sx = x1 - gap - new_w  # no room on the right, use the left
    sy = (y1 + y2) // 2 - new_h // 2
    blend_rgba(frame, resized, sx, sy)


# UI
def put_label(frame, text, org, colour, scale=0.7, thick=2):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org[0], max(th + 6, org[1])
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y + base), colour, -1)
    cv2.putText(frame, text, (x + 4, y - 3), FONT, scale, (20, 20, 20), thick, cv2.LINE_AA)


def draw_face_box(frame, bbox, label):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOURS[label], 2)
    put_label(frame, label.upper(), (x1, y1 - 4), COLOURS[label])


def draw_debug(frame, features, baseline, raw_label=None):
    lines = []
    for k in FEATURE_NAMES:
        v = features[k]
        if baseline is None:
            lines.append(f"{k:<12}{v:+.3f}")
            continue
        b = baseline[k]
        if k in ("eye_open", "mouth_width"):
            rel = f"x{v / b:.2f}" if abs(b) > 1e-6 else "x--"
        else:
            rel = f"{v - b:+.3f}"
        lines.append(f"{k:<12}{v:+.3f}  base {b:+.3f}  {rel}")
    if raw_label is not None:
        lines.append(f"frame label: {raw_label}")

    lh = 20
    w = max(cv2.getTextSize(s, FONT, 0.5, 1)[0][0] for s in lines) + 20
    h = lh * len(lines) + 12
    y0 = frame.shape[0] - h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, y0), (10 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (20, y0 + 22 + i * lh), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_fps(frame, fps):
    text = f"FPS {fps:4.1f}"
    tw = cv2.getTextSize(text, FONT, 0.6, 2)[0][0]
    cv2.putText(frame, text, (frame.shape[1] - tw - 10, 25), FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if ok:
        buf.tofile(str(path))
    return ok


def annotate(frame, face_mesh, stickers, baseline, debug, smoother=None, calibrator=None):
    pts, bbox = detect_face(face_mesh, frame)
    if pts is None:
        put_label(frame, "No face detected", (10, 30), (80, 80, 255))
        return None, None

    features = extract_features(pts)
    if features is None:
        return None, None

    if calibrator is not None and not calibrator.done:
        if calibrator.add(features):
            print("[calibration] baseline:", {k: round(v, 4) for k, v in calibrator.baseline.items()})
        cv2.rectangle(frame, bbox[:2], bbox[2:], (255, 255, 255), 2)
        put_label(frame, f"Calibrating - neutral face ({len(calibrator.samples)}/{calibrator.n_frames})",
                  (10, 30), (255, 255, 255))
        if debug:
            draw_debug(frame, features, None)
        return None, features

    raw = classify_expression(features, baseline)
    label = smoother.update(raw) if smoother else raw

    draw_sticker(frame, stickers[label], bbox)
    draw_face_box(frame, bbox, label)
    if debug:
        draw_debug(frame, features, baseline, raw)
    return label, features


def open_camera(index):
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def run_webcam(camera_index=0):
    cap = open_camera(camera_index)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam {camera_index}. Try --camera 1.")

    stickers = load_stickers()
    calibrator = Calibrator()
    smoother = LabelSmoother()
    debug = True
    fps, last = 0.0, time.perf_counter()
    window = "Cat Sticker Face Detector"
    print("Keys: q = quit, r = recalibrate, d = toggle debug, s = screenshot")

    with create_face_mesh() as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break
            frame = cv2.flip(frame, 1)

            annotate(frame, face_mesh, stickers, calibrator.baseline, debug, smoother, calibrator)

            now = time.perf_counter()
            dt, last = now - last, now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
            draw_fps(frame, fps)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                calibrator.reset()
                smoother.reset()
                print("[calibration] restarting - hold a neutral face")
            elif key == ord("d"):
                debug = not debug
            elif key == ord("s"):
                path = SCREENSHOT_DIR / f"cat_faces_{datetime.now():%Y%m%d_%H%M%S}.png"
                if save_image(frame, path):
                    print(f"[screenshot] saved {path}")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path, out_path=None):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if frame is None:
        sys.exit(f"Could not read image: {image_path}")

    stickers = load_stickers()
    with create_face_mesh(static_image=True) as face_mesh:
        label, features = annotate(frame, face_mesh, stickers, NEUTRAL_DEFAULTS, debug=True)

    if features is None:
        print("No face detected.")
    else:
        print("Features:", {k: round(v, 4) for k, v in features.items()})
        print("Label:   ", label)

    out_path = out_path or SCREENSHOT_DIR / f"{Path(image_path).stem}_annotated.png"
    if save_image(frame, out_path):
        print(f"Saved annotated image to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cat Sticker Face Detector")
    parser.add_argument("--image", help="run on a single image instead of the webcam")
    parser.add_argument("--out", help="output path for --image mode")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    args = parser.parse_args()

    if args.image:
        run_image(args.image, args.out)
    else:
        run_webcam(args.camera)


if __name__ == "__main__":
    main()
