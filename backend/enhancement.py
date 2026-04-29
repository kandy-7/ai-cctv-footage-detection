import cv2
from cv2 import dnn_superres
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_super_res():
    """
    Initialize the Super Resolution model.
    Using ESPCN or FSRCNN for speed, or EDSR for quality.
    For this project, we'll try to use a model included or download one.
    """
    sr = dnn_superres.DnnSuperResImpl_create()
    
    # Check if model exists, if not, we warn (user needs to ensure model file is present)
    model_path = "ESPCN_x4.pb" # Example model path
    
    if os.path.exists(model_path):
        sr.readModel(model_path)
        sr.setModel("espcn", 4) # Scale factor 4
    else:
        logger.warning(f"Super Resolution model {model_path} not found. Using bicubic upscaling fallback.")
        return None
    
    return sr

def upscale_clip(input_path: str, output_path: str, start_time: float, end_time: float):
    """
    Extracts a clip and applies 4K upscaling.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 4K resolution
    target_width = 3840
    target_height = 2160
    
    # Setup Video Writer with H.264 if possible
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        if not out.isOpened():
             logger.warning("avc1 codec failed, falling back to mp4v")
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    except Exception as e:
        logger.error(f"Error initializing VideoWriter: {e}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    sr = init_super_res()
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if sr:
            # AI Upscale
            upscaled = sr.upsample(frame)
            # Resize to exact 4K if model output is different
            final_frame = cv2.resize(upscaled, (target_width, target_height))
        else:
            # Fallback: Bicubic Resize
            final_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            
        out.write(final_frame)
        current_frame += 1
        
    cap.release()
    out.release()
    logger.info(f"Created 4K clip at {output_path}")
