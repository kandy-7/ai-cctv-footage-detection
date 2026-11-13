"""
processing.py – Video analysis pipeline using YOLOv8s + ByteTrack.

Improvements over original:
- YOLOv8s model (better accuracy than yolov8n)
- ByteTrack multi-object tracking (eliminates duplicate detections)
- Accurate timestamp via CAP_PROP_POS_MSEC (not frame_count / fps)
- Processes every 3rd frame (better coverage than every 5th)
- Blur detection + denoising/CLAHE for degraded frames
- Annotated debug frames saved for first 10 events per video
- Duplicate guard via SQLite unique constraint (video_id, frame_number, class_name)
- Video duration stored in Video table after processing
"""

import cv2
from ultralytics import YOLO
import logging
import os
from sqlalchemy.exc import IntegrityError
from database import SessionLocal, Event, Video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch

# ---------------------------------------------------------------------------
# Global model & device configuration
# ---------------------------------------------------------------------------
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFERRED_MODEL = os.getenv("YOLO_MODEL", "yolov8s.pt")
FALLBACK_MODEL = os.path.join(os.path.dirname(__file__), "yolov8n.pt")

model = None
for candidate in [PREFERRED_MODEL, "yolov8s.pt", FALLBACK_MODEL, "yolov8n.pt"]:
    try:
        model = YOLO(candidate)
        logger.info(f"Successfully loaded YOLO model: {candidate} on default device: {DEFAULT_DEVICE}")
        break
    except Exception as exc:
        logger.warning(f"Could not load candidate model '{candidate}': {exc}")

if model is None:
    raise RuntimeError("Could not load any YOLO model. Ensure weights (e.g. yolov8n.pt or yolov8s.pt) are available.")


# ---------------------------------------------------------------------------
# Simple IoU Tracker (fallback when C-extension 'lap' is unavailable)
# ---------------------------------------------------------------------------
class SimpleIoUTracker:
    def __init__(self, iou_thresh: float = 0.3):
        self.tracks = {}  # track_id -> {'cls': int, 'bbox': [x1, y1, x2, y2], 'lost': int}
        self.next_id = 1
        self.iou_thresh = iou_thresh

    @staticmethod
    def _compute_iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections) -> list:
        """
        detections: list of dict(cls=int, bbox=[x1, y1, x2, y2])
        Returns list of track_ids matching each detection.
        """
        assigned_ids = [None] * len(detections)
        used_tracks = set()

        for i, det in enumerate(detections):
            best_iou = 0.0
            best_tid = None
            for tid, tinfo in self.tracks.items():
                if tid in used_tracks:
                    continue
                if tinfo['cls'] == det['cls']:
                    iou = self._compute_iou(tinfo['bbox'], det['bbox'])
                    if iou > best_iou and iou >= self.iou_thresh:
                        best_iou = iou
                        best_tid = tid

            if best_tid is not None:
                assigned_ids[i] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid] = {'cls': det['cls'], 'bbox': det['bbox'], 'lost': 0}
            else:
                new_tid = self.next_id
                self.next_id += 1
                assigned_ids[i] = new_tid
                self.tracks[new_tid] = {'cls': det['cls'], 'bbox': det['bbox'], 'lost': 0}

        # Age tracks
        to_delete = []
        for tid in self.tracks:
            if tid not in used_tracks and tid not in assigned_ids:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > 15:  # Remove track after 15 frames
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return assigned_ids

_HAS_BYTETRACK = None  # Cached flag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_blur_score(image) -> float:
    """Return Laplacian variance (lower = blurrier)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def enhance_frame(frame):
    """
    Apply denoising + CLAHE for blurry / low-light frames.
    Returns an enhanced copy of the frame.
    """
    # Denoising
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    # CLAHE on L channel (LAB colour space)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge((l_ch, a_ch, b_ch))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_video_file(
    video_path: str,
    video_id: str,
    threshold: float = 0.5,
    imgsz: int = 640,
    iou: float = 0.45,
    device: str = None
):
    """
    Read *video_path*, run YOLO + ByteTrack on every 3rd frame,
    store detections in the database, and update the Video record.

    Args:
        video_path:  Absolute or relative path to the video file.
        video_id:    UUID string that identifies this upload.
        threshold:   Minimum confidence to record a detection.
        imgsz:       Inference input resolution (default 640).
        iou:         NMS IoU threshold for tracking (default 0.45).
        device:      Inference device: 'cpu' or 'cuda' (auto-detected if None).
    """
    if device is None:
        device = DEFAULT_DEVICE

    logger.info(f"[{video_id}] Processing started: {video_path} (model={model.model_name if hasattr(model, 'model_name') else 'YOLO'}, imgsz={imgsz}, iou={iou}, device={device})")

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[{video_id}] Cannot open video: {video_path}")
        _mark_done(video_id, duration=None)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = (total_frames / fps) if fps > 0 and total_frames > 0 else None
    logger.info(f"[{video_id}] FPS={fps:.2f}, total_frames={total_frames}, "
                f"duration={video_duration:.2f}s" if video_duration else
                f"[{video_id}] FPS={fps:.2f}, total_frames={total_frames}, duration=unknown")

    # Annotated frame output directory
    debug_dir = os.path.join(os.path.dirname(__file__), "debug_frames", video_id)
    os.makedirs(debug_dir, exist_ok=True)

    db = SessionLocal()
    frame_index = 0        # Counts every frame read
    events_saved = 0       # Total DB rows written

    iou_tracker = SimpleIoUTracker()
    global _HAS_BYTETRACK
    PROCESS_EVERY_NTH = 3  # Sample rate (process every 3rd frame)
    BLUR_THRESHOLD = 50.0  # Below this → enhance frame
    MAX_DEBUG_FRAMES = 10  # Save annotated images for first N events

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Read accurate timestamp AFTER cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Only process every 3rd frame
            if frame_index % PROCESS_EVERY_NTH != 0:
                frame_index += 1
                continue

            # ----------------------------------------------------------
            # Preprocessing: enhance blurry / low-light frames
            # ----------------------------------------------------------
            blur_score = get_blur_score(frame)
            if blur_score < BLUR_THRESHOLD:
                frame = enhance_frame(frame)
                logger.debug(f"[{video_id}] frame {frame_index} enhanced "
                             f"(blur={blur_score:.1f})")

            # ----------------------------------------------------------
            # Detection + Tracking (ByteTrack or robust IoU fallback)
            # ----------------------------------------------------------
            results = None
            if _HAS_BYTETRACK is not False:
                try:
                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=threshold,
                        iou=iou,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                    )
                    _HAS_BYTETRACK = True
                except Exception as track_err:
                    logger.warning(f"ByteTrack tracker unavailable ({track_err}); using integrated IoU Tracker fallback.")
                    _HAS_BYTETRACK = False

            if results is None:
                results = model(
                    frame,
                    conf=threshold,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                )

            frame_has_detection = False

            for result in results:
                # If using fallback tracker, compute track IDs via IoU
                boxes_to_track = []
                valid_boxes = []
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= threshold:
                        valid_boxes.append(box)
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        boxes_to_track.append({'cls': int(box.cls[0]), 'bbox': xyxy})

                assigned_track_ids = []
                if _HAS_BYTETRACK is False and boxes_to_track:
                    assigned_track_ids = iou_tracker.update(boxes_to_track)

                for idx, box in enumerate(valid_boxes):
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]

                    track_id = None
                    if hasattr(box, "id") and box.id is not None:
                        track_id = int(box.id[0])
                    elif assigned_track_ids and idx < len(assigned_track_ids):
                        track_id = assigned_track_ids[idx]

                    # Save annotated debug frame for first few events
                    frame_path = ""
                    if events_saved < MAX_DEBUG_FRAMES:
                        annotated = result.plot()
                        debug_path = os.path.join(debug_dir, f"frame_{frame_index}.jpg")
                        cv2.imwrite(debug_path, annotated)
                        frame_path = debug_path
                        frame_has_detection = True

                    # --------------------------------------------------
                    # Insert – skip on duplicate (video_id, frame, class)
                    # --------------------------------------------------
                    new_event = Event(
                        video_id=video_id,
                        class_name=class_name,
                        confidence=conf,
                        timestamp=timestamp,
                        frame_number=frame_index,
                        track_id=track_id,
                        video_duration=video_duration,
                        frame_path=frame_path,
                    )
                    db.add(new_event)
                    try:
                        db.commit()
                        events_saved += 1
                    except IntegrityError:
                        db.rollback()  # Duplicate – already in DB, ignore

            frame_index += 1

    finally:
        cap.release()
        db.close()

    # ------------------------------------------------------------------
    # Update Video record (duration + mark done)
    # ------------------------------------------------------------------
    _mark_done(video_id, duration=video_duration)
    logger.info(f"[{video_id}] Finished. frames_read={frame_index}, "
                f"events_saved={events_saved}")


def _mark_done(video_id: str, duration):
    """Update the Video row to status='done' and persist duration."""
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = "done"
            if duration is not None:
                video.duration = duration
            db.commit()
        else:
            logger.warning(f"[{video_id}] No Video row found to mark done.")
    finally:
        db.close()
