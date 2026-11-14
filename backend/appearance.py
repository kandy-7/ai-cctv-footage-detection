"""
appearance.py – Person crop extraction and colour estimation for CCTV footage.

Design rules:
- Does NOT load any YOLO model. Receives `result` and `frame` from processing.py.
- Saves person upper-body crops to backend/person_crops/{video_id}/
- Estimates dominant upper-body clothing colour via HSV analysis.
- Rejects tiny, blurry, low-confidence crops.
- Avoids saving near-identical frames from the same track (min 15-frame gap).
- Thread-safe per-video state via per-video-id dicts.
"""

import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

CROPS_DIR = os.path.join(BACKEND_DIR, "person_crops")
os.makedirs(CROPS_DIR, exist_ok=True)

# Minimum acceptable crop dimensions
MIN_CROP_WIDTH = 30
MIN_CROP_HEIGHT = 40

# Minimum Laplacian variance (blur rejection)
MIN_BLUR_SCORE = 20.0

# Minimum frames between saves for the same track (avoids near-duplicates)
MIN_TRACK_FRAME_GAP = 15

# Minimum person confidence to extract crop
MIN_PERSON_CONFIDENCE = 0.4

# Upper-body fraction (top portion of the bbox)
UPPER_BODY_FRACTION = 0.55

# --------------------------------------------------
# Per-video tracker state
# --------------------------------------------------

# Maps video_id -> { track_id -> last_saved_frame_number }
_track_state: dict[str, dict[int, int]] = {}

# Counter for tracks that have no assigned ID (fallback)
_fallback_counter: dict[str, int] = {}


def reset_track_state(video_id: str) -> None:
    """
    Reset per-video track state.
    Call this ONCE at the start of processing each video.
    """
    _track_state[video_id] = {}
    _fallback_counter[video_id] = 0
    logger.debug("Appearance track state reset for video_id=%s", video_id)


# --------------------------------------------------
# Colour estimation
# --------------------------------------------------

# HSV ranges: (lower_h, upper_h, lower_s_min, label)
# H in OpenCV is 0–179, S/V are 0–255
_COLOR_RANGES = [
    # name         H_low  H_high  S_min  V_min
    ("red",           0,    10,   80,   60),   # red (lower)
    ("orange",       10,    25,   80,   80),
    ("yellow",       25,    38,   80,   80),
    ("green",        38,    85,   50,   50),
    ("blue",         85,   130,   50,   50),
    ("purple",      130,   155,   50,   50),
    ("pink",        155,   175,   40,   80),
    ("red_upper",   165,   179,   80,   60),   # red (upper wrap)
]

_WHITE_S_MAX = 60    # S < 60 → white/grey/black (not chromatic)
_BLACK_V_MAX = 60    # V < 60 → black
_WHITE_V_MIN = 160   # V > 160 AND S < 60 → white


def _mask_skin(hsv_img: np.ndarray) -> np.ndarray:
    """Return a boolean mask where True = likely skin pixel (to exclude)."""
    # Skin tone: H 0–25 (and 160–179), S 25–170, V 100–255
    lower_skin1 = np.array([0,  25, 100], dtype=np.uint8)
    upper_skin1 = np.array([25, 170, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 25, 100], dtype=np.uint8)
    upper_skin2 = np.array([179, 170, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_img, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv_img, lower_skin2, upper_skin2)
    return (mask1 | mask2).astype(bool)


def estimate_upper_color(crop_bgr: np.ndarray) -> tuple[str, float]:
    """
    Estimate dominant upper-body clothing colour via HSV analysis.

    Returns:
        (color_name, confidence)   where confidence is fraction of
        non-skin pixels matching the dominant colour bin.
        Returns ("unknown", 0.0) when too few valid pixels are found.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return ("unknown", 0.0)

    # Resize to fixed size for consistent analysis
    h_px, w_px = crop_bgr.shape[:2]
    if h_px > 100 or w_px > 80:
        crop_bgr = cv2.resize(crop_bgr, (80, 100))

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    skin_mask = _mask_skin(hsv)

    # Non-skin pixels
    valid_mask = ~skin_mask
    total_valid = int(valid_mask.sum())

    if total_valid < 20:
        return ("unknown", 0.0)

    # Classify valid pixels
    color_counts: dict[str, int] = {
        "red": 0, "orange": 0, "yellow": 0,
        "green": 0, "blue": 0, "purple": 0,
        "pink": 0, "white": 0, "black": 0, "other": 0,
    }

    h_valid = H[valid_mask]
    s_valid = S[valid_mask]
    v_valid = V[valid_mask]

    for i in range(len(h_valid)):
        h, s, v = int(h_valid[i]), int(s_valid[i]), int(v_valid[i])

        if v < _BLACK_V_MAX:
            color_counts["black"] += 1
            continue

        if s < _WHITE_S_MAX:
            if v >= _WHITE_V_MIN:
                color_counts["white"] += 1
            else:
                color_counts["other"] += 1   # grey
            continue

        matched = False
        for name, h_lo, h_hi, s_min, v_min in _COLOR_RANGES:
            if h_lo <= h <= h_hi and s >= s_min and v >= v_min:
                canonical = "red" if name == "red_upper" else name
                color_counts[canonical] += 1
                matched = True
                break

        if not matched:
            color_counts["other"] += 1

    # Find dominant colour
    best_color = max(color_counts, key=lambda c: color_counts[c])
    best_count = color_counts[best_color]
    confidence = best_count / total_valid

    # Require at least 20% dominance; otherwise report "other"
    if confidence < 0.20:
        return ("other", float(confidence))

    return (best_color, float(round(confidence, 3)))


# --------------------------------------------------
# Blur detection
# --------------------------------------------------

def _blur_score(gray_img: np.ndarray) -> float:
    """Return Laplacian variance (higher = sharper)."""
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


# --------------------------------------------------
# Main extraction function
# --------------------------------------------------

def extract_appearances_from_result(
    result,                 # ultralytics Results object from processing.py
    frame: np.ndarray,      # original BGR frame
    video_id: str,
    frame_number: int,
    timestamp: float,
) -> list[dict]:
    """
    Extract person appearance observations from a single YOLO result.

    Parameters
    ----------
    result      : ultralytics.engine.results.Results
    frame       : BGR numpy array (original frame, not annotated)
    video_id    : str
    frame_number: int (global frame index from processing loop)
    timestamp   : float (seconds)

    Returns
    -------
    List of dicts, each containing:
        video_id, track_id, frame_number, timestamp,
        crop_path, upper_color, upper_color_confidence
    """

    # Ensure state exists for this video
    if video_id not in _track_state:
        reset_track_state(video_id)

    track_seen = _track_state[video_id]
    observations: list[dict] = []

    boxes = result.boxes
    if boxes is None:
        return observations

    img_h, img_w = frame.shape[:2]

    video_crops_dir = os.path.join(CROPS_DIR, video_id)
    os.makedirs(video_crops_dir, exist_ok=True)

    for box in boxes:
        # Only process person detections
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]

        if class_name != "person":
            continue

        confidence = float(box.conf[0].item())
        if confidence < MIN_PERSON_CONFIDENCE:
            continue

        # Bounding box (xyxy, pixel coords)
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = (
            int(max(0, xyxy[0])),
            int(max(0, xyxy[1])),
            int(min(img_w, xyxy[2])),
            int(min(img_h, xyxy[3])),
        )

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w < MIN_CROP_WIDTH or bbox_h < MIN_CROP_HEIGHT:
            continue

        # Track ID
        track_id: int
        if box.id is not None:
            try:
                track_id = int(box.id[0].item())
            except Exception:
                _fallback_counter[video_id] += 1
                track_id = -_fallback_counter[video_id]
        else:
            _fallback_counter[video_id] += 1
            track_id = -_fallback_counter[video_id]

        # Throttle: skip if same track saved recently
        last_frame = track_seen.get(track_id, -MIN_TRACK_FRAME_GAP - 1)
        if frame_number - last_frame < MIN_TRACK_FRAME_GAP:
            continue

        # Upper-body crop: top UPPER_BODY_FRACTION of bbox
        upper_y2 = y1 + int(bbox_h * UPPER_BODY_FRACTION)
        upper_y2 = min(upper_y2, img_h)

        crop = frame[y1:upper_y2, x1:x2]

        if crop.size == 0:
            continue

        # Blur rejection
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray_crop)
        if blur < MIN_BLUR_SCORE:
            logger.debug(
                "Rejecting crop track=%s frame=%s blur=%.1f",
                track_id, frame_number, blur
            )
            continue

        # Colour estimation
        upper_color, upper_color_confidence = estimate_upper_color(crop)

        # Build crop filename: track id may be negative (fallback); use abs
        tid_str = f"t{abs(track_id)}" if track_id >= 0 else f"unk{abs(track_id)}"
        crop_filename = (
            f"vid{video_id[:8]}_{tid_str}_f{frame_number}.jpg"
        )
        crop_path = os.path.join(video_crops_dir, crop_filename)

        # Save crop
        saved = cv2.imwrite(crop_path, crop)
        if not saved:
            logger.warning("Failed to write crop: %s", crop_path)
            continue

        # Update tracker state
        track_seen[track_id] = frame_number

        observations.append({
            "video_id": video_id,
            "track_id": track_id if track_id >= 0 else None,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "crop_path": crop_path,
            "upper_color": upper_color,
            "upper_color_confidence": upper_color_confidence,
        })

        logger.debug(
            "Appearance: track=%s frame=%s color=%s conf=%.2f",
            track_id, frame_number, upper_color, upper_color_confidence
        )

    return observations
