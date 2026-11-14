
import os
import cv2
import logging
import torch

from ultralytics import YOLO
from database import SessionLocal, Event, Video, AppearanceObservation
from appearance import (
    extract_appearances_from_result,
    reset_track_state,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

DEBUG_DIR = os.path.join(
    BACKEND_DIR,
    "debug_frames"
)

os.makedirs(DEBUG_DIR, exist_ok=True)

MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8s.pt")

FRAME_INTERVAL = int(
    os.getenv("FRAME_INTERVAL", "3")
)

IMAGE_SIZE = int(
    os.getenv("YOLO_IMAGE_SIZE", "640")
)

DEFAULT_CONFIDENCE = float(
    os.getenv("YOLO_CONFIDENCE", "0.5")
)

IOU_THRESHOLD = float(
    os.getenv("YOLO_IOU", "0.45")
)

DEVICE = "0" if torch.cuda.is_available() else "cpu"

model = None
tracker_enabled = True


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------

def get_model():

    global model

    if model is not None:
        return model

    try:
        logger.info("Loading YOLO model: %s", MODEL_NAME)
        model = YOLO(MODEL_NAME)

    except Exception:
        logger.exception(
            "Could not load %s. Falling back to yolov8n.pt",
            MODEL_NAME
        )

        model = YOLO("yolov8n.pt")

    logger.info("YOLO model loaded successfully.")
    logger.info("Inference device: %s", DEVICE)

    return model


# --------------------------------------------------
# VIDEO METADATA
# --------------------------------------------------

def get_video_metadata(cap):

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0
        logger.warning("Invalid FPS. Using fallback 30 FPS.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        total_frames / fps
        if total_frames > 0
        else 0.0
    )

    return fps, total_frames, duration


# --------------------------------------------------
# DETECTION
# --------------------------------------------------

def run_detection(frame, threshold, imgsz, iou):

    global tracker_enabled

    detector = get_model()

    if tracker_enabled:

        try:
            results = detector.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=threshold,
                iou=iou,
                imgsz=imgsz,
                device=DEVICE,
                verbose=False
            )

            return results

        except Exception:
            logger.exception(
                "Tracking unavailable. Falling back to prediction."
            )

            tracker_enabled = False

    return detector.predict(
        source=frame,
        conf=threshold,
        iou=iou,
        imgsz=imgsz,
        device=DEVICE,
        verbose=False
    )


# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------

def process_video_file(
    video_path: str,
    video_id: str,
    threshold: float = DEFAULT_CONFIDENCE,
    imgsz: int = IMAGE_SIZE,
    iou: float = IOU_THRESHOLD,
):

    db = SessionLocal()
    cap = None

    try:

        logger.info(
            "Starting video processing: %s",
            video_path
        )

        video_record = (
            db.query(Video)
            .filter(Video.id == video_id)
            .first()
        )

        if video_record is None:
            raise ValueError(
                f"Video record not found: {video_id}"
            )

        video_record.status = "processing"
        db.commit()

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {video_path}"
            )

        fps, total_frames, duration = get_video_metadata(cap)

        logger.info(
            "Video metadata: FPS=%s, Frames=%s, Duration=%.3fs",
            fps,
            total_frames,
            duration
        )

        video_record.duration = duration
        db.commit()

        video_debug_dir = os.path.join(
            DEBUG_DIR,
            video_id
        )

        os.makedirs(video_debug_dir, exist_ok=True)

        frame_count = 0
        processed_count = 0

        # Reset appearance tracker state for this video
        reset_track_state(video_id)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Process every Nth frame.
            if frame_count % FRAME_INTERVAL == 0:

                # OpenCV position in milliseconds.
                timestamp_ms = cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )

                timestamp = (
                    timestamp_ms / 1000.0
                    if timestamp_ms >= 0
                    else frame_count / fps
                )

                # Fallback if the decoder returns zero
                # for a non-first frame.
                if frame_count > 0 and timestamp <= 0:
                    timestamp = frame_count / fps

                results = run_detection(
                    frame,
                    threshold,
                    imgsz,
                    iou
                )

                for result in results:

                    boxes = result.boxes

                    if boxes is None:
                        continue

                    annotated_frame = result.plot()

                    for box in boxes:

                        confidence = float(
                            box.conf[0].item()
                        )

                        if confidence < threshold:
                            continue

                        class_id = int(
                            box.cls[0].item()
                        )

                        class_name = result.names[class_id]

                        track_id = None

                        if box.id is not None:
                            track_id = int(
                                box.id[0].item()
                            )

                        event = Event(
                            video_id=video_id,
                            class_name=class_name,
                            confidence=confidence,
                            timestamp=float(timestamp),
                            frame_number=frame_count,
                            track_id=track_id,
                            video_duration=duration,
                            frame_path=""
                        )

                        db.add(event)

                        logger.info(
                            "Detected %s | Confidence %.2f | "
                            "Timestamp %.3f | Track %s",
                            class_name,
                            confidence,
                            timestamp,
                            track_id
                        )

                    # Save one annotated image per sampled frame.
                    debug_path = os.path.join(
                        video_debug_dir,
                        f"frame_{frame_count}.jpg"
                    )

                    cv2.imwrite(
                        debug_path,
                        annotated_frame
                    )

                    # ── Appearance extraction ──────────────────────────
                    # Reuse the same result; no second YOLO call.
                    try:
                        app_obs_list = extract_appearances_from_result(
                            result,
                            frame,
                            video_id,
                            frame_count,
                            float(timestamp),
                        )

                        for obs in app_obs_list:
                            app_row = AppearanceObservation(
                                video_id=obs["video_id"],
                                track_id=obs["track_id"],
                                timestamp=obs["timestamp"],
                                frame_number=obs["frame_number"],
                                upper_color=obs["upper_color"],
                                upper_color_confidence=obs[
                                    "upper_color_confidence"
                                ],
                                crop_path=obs["crop_path"],
                            )
                            db.add(app_row)

                    except Exception:
                        logger.exception(
                            "Appearance extraction error at frame %s",
                            frame_count
                        )

                # Commit events + appearance rows for this frame.
                db.commit()

                processed_count += 1

            frame_count += 1

        # ------------------------------------------
        # PROCESSING COMPLETED
        # ------------------------------------------

        video_record = (
            db.query(Video)
            .filter(Video.id == video_id)
            .first()
        )

        if video_record:

            video_record.duration = duration
            video_record.status = "done"

            db.commit()

        logger.info(
            "Processing completed. Total frames=%s, "
            "Processed frames=%s",
            frame_count,
            processed_count
        )

    except Exception:

        db.rollback()

        logger.exception(
            "Video processing failed: %s",
            video_id
        )

        try:

            video_record = (
                db.query(Video)
                .filter(Video.id == video_id)
                .first()
            )

            if video_record:
                video_record.status = "failed"
                db.commit()

        except Exception:
            db.rollback()
            logger.exception("Could not update failed status.")

        raise

    finally:

        if cap is not None:
            cap.release()

        db.close()
