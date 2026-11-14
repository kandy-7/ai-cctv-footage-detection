
"""
enhancement.py
CCTV clip extraction, adaptive resizing, optional ESPCN
super-resolution, and browser-compatible H.264 MP4 encoding.
"""

import os
import cv2
import math
import logging
import shutil
import subprocess

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

try:
    from cv2 import dnn_superres
except ImportError:
    dnn_superres = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MAX_OUTPUT_WIDTH = 1280
MAX_OUTPUT_HEIGHT = 720

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SR_MODEL_PATH = os.path.join(BASE_DIR, "ESPCN_x4.pb")


# ---------------------------------------------------------
# FFMPEG DETECTION
# ---------------------------------------------------------

def get_ffmpeg_path():
    """
    Locate FFmpeg using imageio-ffmpeg first,
    then fall back to the system PATH.
    """

    if imageio_ffmpeg is not None:
        try:
            path = imageio_ffmpeg.get_ffmpeg_exe()

            if path and os.path.isfile(path):
                logger.info("Using imageio FFmpeg: %s", path)
                return path

        except Exception:
            logger.exception("Could not locate imageio FFmpeg.")

    path = shutil.which("ffmpeg")

    if path:
        logger.info("Using system FFmpeg: %s", path)
        return path

    logger.error(
        "FFmpeg not found. Install it using: "
        "python -m pip install imageio-ffmpeg"
    )

    return None


# ---------------------------------------------------------
# SUPER RESOLUTION
# ---------------------------------------------------------

def init_super_res(model_path=SR_MODEL_PATH):
    """Load optional ESPCN x4 super-resolution model."""

    if dnn_superres is None:
        logger.warning("OpenCV super-resolution module unavailable.")
        return None

    if not os.path.exists(model_path):
        logger.warning(
            "ESPCN model not found. Using bicubic resizing."
        )
        return None

    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("espcn", 4)

        logger.info("ESPCN super-resolution model loaded.")
        return sr

    except Exception:
        logger.exception("Could not load ESPCN model.")
        return None


# ---------------------------------------------------------
# RESOLUTION HANDLING
# ---------------------------------------------------------

def _compute_target_resolution(width, height):
    """Calculate output dimensions while preserving aspect ratio."""

    if width <= 0 or height <= 0:
        raise ValueError("Invalid source video dimensions.")

    if width * height <= 640 * 480:
        desired_scale = 2.0

    elif width * height <= 1280 * 720:
        desired_scale = 1.5

    else:
        desired_scale = 1.0

    scale = min(
        desired_scale,
        MAX_OUTPUT_WIDTH / width,
        MAX_OUTPUT_HEIGHT / height
    )

    target_width = max(2, int(round(width * scale)))
    target_height = max(2, int(round(height * scale)))

    # Ensure even dimensions for video encoding.
    target_width -= target_width % 2
    target_height -= target_height % 2

    return target_width, target_height


# ---------------------------------------------------------
# MAIN CLIP EXTRACTION
# ---------------------------------------------------------

def upscale_clip(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float
) -> bool:

    cap = None
    out = None
    temp_path = None

    logger.info(
        "Starting clip extraction: %s | %.2f - %.2f seconds",
        input_path,
        start_time,
        end_time
    )

    try:

        # 1. Validate input
        if not os.path.isfile(input_path):
            logger.error("Input video does not exist: %s", input_path)
            return False

        if (
            not math.isfinite(start_time)
            or not math.isfinite(end_time)
            or start_time < 0
            or end_time <= start_time
        ):
            logger.error("Invalid clip timestamps.")
            return False

        # 2. Find FFmpeg
        ffmpeg_path = get_ffmpeg_path()

        if not ffmpeg_path:
            return False

        # 3. Prepare output paths
        output_path = os.path.abspath(output_path)

        if not output_path.lower().endswith(".mp4"):
            output_path = os.path.splitext(output_path)[0] + ".mp4"

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        temp_path = os.path.splitext(output_path)[0] + "_temp.avi"

        # Remove an old output so a failed conversion
        # cannot accidentally return a stale video.
        if os.path.exists(output_path):
            os.remove(output_path)

        # 4. Open source video
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            logger.error("Cannot open input video.")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0

        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if src_width <= 0 or src_height <= 0:
            logger.error("Invalid source video dimensions.")
            return False

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        duration = (
            frame_count / fps
            if frame_count > 0
            else 0
        )

        if duration > 0:
            end_time = min(end_time, duration)

        if end_time <= start_time:
            logger.error("Requested clip is outside video duration.")
            return False

        logger.info(
            "Source: %dx%d | %.2f FPS | %.2f seconds",
            src_width,
            src_height,
            fps,
            duration
        )

        # 5. Seek to requested start time
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            start_time * 1000.0
        )

        actual_position = cap.get(
            cv2.CAP_PROP_POS_MSEC
        ) / 1000.0

        logger.info(
            "Requested start: %.3f | Actual seek: %.3f",
            start_time,
            actual_position
        )

        # 6. Calculate output dimensions
        target_width, target_height = _compute_target_resolution(
            src_width,
            src_height
        )

        logger.info(
            "Output resolution: %dx%d",
            target_width,
            target_height
        )

        # 7. Initialize temporary writer
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        out = cv2.VideoWriter(
            temp_path,
            fourcc,
            fps,
            (target_width, target_height)
        )

        if not out.isOpened():
            logger.error("Could not initialize temporary video writer.")
            return False

        # 8. Load optional super-resolution model
        sr = init_super_res()

        # 9. Process frames
        frames_written = 0
        previous_timestamp = -1.0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            timestamp = (
                cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            )

            # Fallback for unreliable video timestamps.
            if (
                not math.isfinite(timestamp)
                or timestamp <= previous_timestamp
                or timestamp <= 0
            ):
                current_frame = cap.get(
                    cv2.CAP_PROP_POS_FRAMES
                )

                timestamp = max(
                    0.0,
                    (current_frame - 1) / fps
                )

            previous_timestamp = timestamp

            if timestamp >= end_time:
                break

            # Optional AI super-resolution
            if sr is not None:
                try:
                    frame = sr.upsample(frame)

                except Exception:
                    logger.exception(
                        "Super-resolution failed. "
                        "Switching to normal resizing."
                    )
                    sr = None

            # Resize while preserving aspect ratio.
            interpolation = (
                cv2.INTER_LANCZOS4
                if target_width > src_width
                else cv2.INTER_AREA
            )

            frame = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=interpolation
            )

            out.write(frame)
            frames_written += 1

        logger.info("Frames extracted: %d", frames_written)

        if frames_written == 0:
            logger.error("No frames extracted.")
            return False

        # 10. Release OpenCV resources
        cap.release()
        cap = None

        out.release()
        out = None

        # 11. Convert to browser-compatible H.264 MP4
        command = [
            ffmpeg_path,
            "-y",
            "-i", temp_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            output_path
        ]

        logger.info("Converting video to H.264 MP4...")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            logger.error(
                "FFmpeg conversion failed:\n%s",
                result.stderr[-3000:]
            )
            return False

        # 12. Verify output
        if (
            not os.path.isfile(output_path)
            or os.path.getsize(output_path) == 0
        ):
            logger.error("Generated MP4 is missing or empty.")
            return False

        logger.info(
            "SUCCESS: Generated browser-compatible video: %s",
            output_path
        )

        return True

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out.")
        return False

    except Exception:
        logger.exception("Clip extraction failed.")
        return False

    finally:

        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)

            except OSError:
                logger.warning(
                    "Could not remove temporary file: %s",
                    temp_path
                )
