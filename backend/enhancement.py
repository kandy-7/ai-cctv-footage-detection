"""
enhancement.py — Smart clip extraction and upscaling for CCTV footage.

Key improvements over the original:
- Time-based seeking (CAP_PROP_POS_MSEC) instead of frame-number-based.
- Adaptive output resolution (no forced 4K); respects a 1280px wide cap.
- Aspect-ratio preservation.
- Optional Super Resolution via ESPCN model; falls back to bicubic gracefully.
- Robust codec selection (mp4v → XVID fallback).
- Proper error handling and detailed logging.
"""

import os
import logging
import cv2
from cv2 import dnn_superres

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Super-resolution helpers
# ---------------------------------------------------------------------------

def init_super_res(model_path: str = "ESPCN_x4.pb"):
    """
    Try to load the ESPCN x4 super-resolution model.

    Returns a DnnSuperResImpl instance on success, or None when the model
    file is absent (bicubic upscaling will be used instead).
    """
    if not os.path.exists(model_path):
        logger.warning(
            "SR model '%s' not found — bicubic upscaling will be used.", model_path
        )
        return None

    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("espcn", 4)  # scale factor 4
        logger.info("Super-resolution model loaded from '%s'.", model_path)
        return sr
    except Exception as exc:
        logger.warning("Failed to load SR model (%s) — using bicubic.", exc)
        return None


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

MAX_OUTPUT_WIDTH = 1280   # cap for browser playback performance
MAX_OUTPUT_HEIGHT = 720

def _compute_target_resolution(width: int, height: int) -> tuple[int, int]:
    """
    Decide the output (target_width, target_height) given the source dimensions.

    Rules:
      - area <= 640x480  → try 2x upscale
      - area <= 1280x720 → try 1.5x upscale
      - area >  1280x720 → keep original (never downscale)
      - Hard cap: output width <= MAX_OUTPUT_WIDTH, height <= MAX_OUTPUT_HEIGHT
      - Aspect ratio is always preserved.
    """
    area = width * height

    if area <= 640 * 480:
        desired_scale = 2.0
    elif area <= 1280 * 720:
        desired_scale = 1.5
    else:
        desired_scale = 1.0  # already large enough

    # Scale factor is limited so we never exceed the browser-playback cap.
    scale = min(
        desired_scale,
        MAX_OUTPUT_WIDTH / width,
        MAX_OUTPUT_HEIGHT / height,
    )

    # Never shrink the video — if the source is already above the cap we
    # do want to cap it, but do not go below the cap if scale < 1.
    # Spec says "if original is already > 1280 wide, keep it at original
    # but cap at 1280" — so a scale < 1 is allowed only to enforce the cap.
    # However, if the original is smaller and desired_scale would not increase
    # the size, still apply it (edge case: scale already < 1 from div alone).

    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))

    # Make dimensions even (required by most codecs).
    target_width += target_width % 2
    target_height += target_height % 2

    return target_width, target_height


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def upscale_clip(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> bool:
    """
    Extract a time segment from *input_path* and write it to *output_path*,
    applying smart upscaling.

    Parameters
    ----------
    input_path  : source video file path
    output_path : destination .mp4 file path
    start_time  : clip start in seconds (float)
    end_time    : clip end   in seconds (float)

    Returns
    -------
    True on success, False on failure.
    """
    logger.info(
        "upscale_clip: input='%s' start=%.3fs end=%.3fs",
        input_path, start_time, end_time,
    )

    # ------------------------------------------------------------------
    # Open source video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Cannot open video file: '%s'", input_path)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # sane default if unreadable
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Source: %dx%d @ %.2f fps", src_width, src_height, fps)

    # ------------------------------------------------------------------
    # Seek to start using millisecond-based position (more reliable than
    # frame-number seeking for variable-FPS or B-frame videos).
    # ------------------------------------------------------------------
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)
    actual_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    actual_pos_sec = actual_pos_msec / 1000.0
    logger.info(
        "Seek: requested=%.3fs  actual=%.3fs  (delta=%.3fs)",
        start_time, actual_pos_sec, actual_pos_sec - start_time,
    )

    # ------------------------------------------------------------------
    # Compute output resolution
    # ------------------------------------------------------------------
    target_width, target_height = _compute_target_resolution(src_width, src_height)
    logger.info("Output resolution: %dx%d", target_width, target_height)

    # ------------------------------------------------------------------
    # Initialise VideoWriter — try mp4v first, fall back to XVID.
    # ------------------------------------------------------------------
    if not output_path.lower().endswith(".mp4"):
        output_path = os.path.splitext(output_path)[0] + ".mp4"

    out = None
    for codec in ("mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            output_path, fourcc, fps, (target_width, target_height)
        )
        if writer.isOpened():
            out = writer
            logger.info("VideoWriter opened with codec '%s'.", codec)
            break
        logger.warning("Codec '%s' failed, trying next…", codec)

    if out is None:
        logger.error("Could not open VideoWriter for '%s'.", output_path)
        cap.release()
        return False

    # ------------------------------------------------------------------
    # Load optional super-resolution model
    # ------------------------------------------------------------------
    sr = init_super_res()

    # ------------------------------------------------------------------
    # Read & write frames until end_time
    # ------------------------------------------------------------------
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of file or read error

        # Time-based stop condition — no frame-count arithmetic.
        current_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_pos_sec >= end_time:
            break

        # --- Upscale / resize the frame ---
        if sr is not None:
            # AI super-resolution (x4), then cap to target size.
            upscaled = sr.upsample(frame)
            final_frame = cv2.resize(
                upscaled, (target_width, target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            # Bicubic interpolation — fast and decent quality.
            final_frame = cv2.resize(
                frame, (target_width, target_height),
                interpolation=cv2.INTER_CUBIC,
            )

        out.write(final_frame)
        frames_written += 1

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------
    cap.release()
    out.release()

    logger.info(
        "upscale_clip done: output='%s'  resolution=%dx%d  frames=%d",
        output_path, target_width, target_height, frames_written,
    )
    return True
