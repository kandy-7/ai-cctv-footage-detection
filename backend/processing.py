import cv2
from ultralytics import YOLO
import logging
import os
from database import SessionLocal, Event
import uuid

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
# Note: The model will be downloaded on first run if not present
model = YOLO('yolov8n.pt') 

def get_blur_score(image):
    """
    Compute the Laplacian Variance of the image.
    Lower value = More blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_video_file(video_path: str, video_id: str, threshold: float = 0.5):
    """
    Reads video, runs detection, and logs events to DB.
    """
    logger.info(f"Starting processing for {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file.")
        return
    
    db = SessionLocal()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning("Could not get valid FPS from video metadata. Defaulting to 30.0")
        fps = 30.0
        
    logger.info(f"Video FPS: {fps}")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / fps
        
        # Optimization: Process every Nth frame (e.g., every 5th frame) to speed up
        # logging every single frame might be too much for DB
        if frame_count % 5 == 0:
            # Blur detection
            blur_score = get_blur_score(frame)
            
            # Run Object Detection
            results = model(frame, verbose=False) # verbose=False to reduce log noise
            
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= threshold:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id]
                        
                        # Log to DB
                        new_event = Event(
                            video_id=video_id,
                            class_name=class_name,
                            confidence=conf,
                            timestamp=timestamp,
                            frame_path="" # Can save frame if needed
                        )
                        db.add(new_event)
            db.commit()

        frame_count += 1
    
    cap.release()
    db.close()
    logger.info(f"Finished processing {video_path}. Total frames: {frame_count}")
