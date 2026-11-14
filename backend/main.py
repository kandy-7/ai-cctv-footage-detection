
import os
import uuid
import shutil
import logging

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Request,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from processing import process_video_file
from search import search_events, search_appearances
from enhancement import upscale_clip

from database import (
    init_db,
    SessionLocal,
    Event,
    Video,
    AppearanceObservation,
    BACKEND_DIR,
)

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# DIRECTORY CONFIGURATION
# --------------------------------------------------

UPLOAD_DIR = os.path.join(
    BACKEND_DIR,
    "uploads"
)

PROCESSED_DIR = os.path.join(
    BACKEND_DIR,
    "processed"
)

CROPS_DIR = os.path.join(
    BACKEND_DIR,
    "person_crops"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}

# --------------------------------------------------
# FASTAPI LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting CCTV Analysis System")

    init_db()

    yield

    logger.info("Shutting down CCTV Analysis System")


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="CCTV Analysis System",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=PROCESSED_DIR),
    name="static",
)

app.mount(
    "/crops",
    StaticFiles(directory=CROPS_DIR),
    name="crops",
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class QueryRequest(BaseModel):

    video_id: str

    query: str

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0
    )


class AppearanceQueryRequest(BaseModel):

    query: str

    video_id: str | None = None

    confidence_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100
    )


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def read_root():

    return {
        "message": "CCTV AI System Ready",
        "version": "2.1.0"
    }


# --------------------------------------------------
# UPLOAD VIDEO
# --------------------------------------------------

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    original_filename = (
        Path(
            (file.filename or "video.mp4").replace("\\", "/")
        ).name
    )

    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    video_id = str(uuid.uuid4())

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{video_id}_{original_filename}"
    )

    try:

        file_size = 0

        with open(file_path, "wb") as buffer:

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                file_size += len(chunk)

                buffer.write(chunk)

        if file_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # Create database record.
        db = SessionLocal()

        try:

            video = Video(
                id=video_id,
                filename=original_filename,
                duration=None,
                status="processing"
            )

            db.add(video)
            db.commit()

        finally:
            db.close()

        # Start background processing.
        background_tasks.add_task(
            process_video_file,
            file_path,
            video_id
        )

        logger.info(
            "Video uploaded successfully: %s",
            video_id
        )

        return {
            "video_id": video_id,
            "filename": original_filename,
            "size_bytes": file_size,
            "status": "processing",
            "message": "Video uploaded and processing started."
        }

    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    except Exception:

        if os.path.exists(file_path):
            os.remove(file_path)

        logger.exception("Video upload failed.")

        raise HTTPException(
            status_code=500,
            detail="Video upload failed."
        )

    finally:
        await file.close()


# --------------------------------------------------
# PROCESSING STATUS
# --------------------------------------------------

@app.get("/status/{video_id}")
def get_status(video_id: str):

    db = SessionLocal()

    try:

        video = (
            db.query(Video)
            .filter(Video.id == video_id)
            .first()
        )

        if not video:

            raise HTTPException(
                status_code=404,
                detail="Video not found."
            )

        event_count = (
            db.query(Event)
            .filter(Event.video_id == video_id)
            .count()
        )

        return {
            "video_id": video_id,
            "status": video.status,
            "event_count": event_count,
            "duration": video.duration
        }

    finally:
        db.close()


# --------------------------------------------------
# LIST VIDEOS
# --------------------------------------------------

@app.get("/videos")
def list_videos():

    db = SessionLocal()

    try:

        videos = (
            db.query(Video)
            .order_by(Video.created_at.desc())
            .all()
        )

        return [
            {
                "video_id": video.id,
                "filename": video.filename,
                "status": video.status,
                "duration": video.duration,
                "created_at": (
                    video.created_at.isoformat()
                    if video.created_at
                    else None
                )
            }
            for video in videos
        ]

    finally:
        db.close()


# --------------------------------------------------
# SEARCH CCTV EVENTS
# --------------------------------------------------

@app.post("/query")
async def query_video(
    request: QueryRequest,
    http_request: Request
):

    db = SessionLocal()

    try:

        video = (
            db.query(Video)
            .filter(Video.id == request.video_id)
            .first()
        )

        if not video:

            raise HTTPException(
                status_code=404,
                detail="Video not found."
            )

        if video.status == "processing":

            raise HTTPException(
                status_code=409,
                detail="Video is still processing. Please wait."
            )

        if video.status == "failed":

            raise HTTPException(
                status_code=409,
                detail="Video processing failed. Please upload again."
            )

        video_duration = video.duration or 0.0

    finally:
        db.close()

    # ----------------------------------------------
    # SEARCH EVENTS
    # ----------------------------------------------

    try:

        event_results = search_events(
            request.query,
            request.video_id,
            request.threshold
        )

    except Exception:

        logger.exception("Event search failed.")

        raise HTTPException(
            status_code=500,
            detail="Event search failed."
        )

    if not event_results:

        return {
            "found": False,
            "message": "No matching events found.",
            "events": [],
            "total_matches": 0
        }

    # ----------------------------------------------
    # FIND ORIGINAL VIDEO
    # ----------------------------------------------

    matching_files = [
        path for path in Path(UPLOAD_DIR).glob(
            f"{request.video_id}_*"
        )
        if path.is_file()
    ]

    if not matching_files:

        raise HTTPException(
            status_code=404,
            detail="Original video file not found."
        )

    input_file = str(matching_files[0])

    # ----------------------------------------------
    # EXTRACT MATCHING CLIPS
    # ----------------------------------------------

    processed_events = []

    for event in event_results[:10]:

        timestamp = float(event["timestamp"])

        start_time = max(
            0.0,
            timestamp - 2.0
        )

        end_time = (
            timestamp + 3.0
        )

        if video_duration > 0:
            end_time = min(
                end_time,
                video_duration
            )

        if end_time <= start_time:
            continue

        ts_tag = (
            f"{timestamp:.3f}".replace(".", "_")
        )

        output_filename = (
            f"{request.video_id}_{ts_tag}_720p.mp4"
        )

        output_path = os.path.join(
            PROCESSED_DIR,
            output_filename
        )

        try:

            if not os.path.exists(output_path):

                upscale_clip(
                    input_file,
                    output_path,
                    start_time,
                    end_time
                )

            if not os.path.exists(output_path):
                logger.error(
                    "Clip extraction did not create %s",
                    output_path
                )
                continue

            video_url = (
                str(http_request.base_url).rstrip("/")
                + f"/static/{output_filename}"
            )

            processed_events.append({
                "timestamp": timestamp,
                "start_time": start_time,
                "end_time": end_time,
                "video_url": video_url,
                "classes": event.get("classes", []),
                "count": event.get("count", 0),
                "confidence": event.get("confidence"),
            })

        except Exception:

            logger.exception(
                "Failed to extract event clip at %.3f seconds",
                timestamp
            )

    return {
        "found": len(processed_events) > 0,
        "events": processed_events,
        "total_matches": len(event_results),
        "clips_generated": len(processed_events)
    }


# --------------------------------------------------
# APPEARANCE SEARCH – supported colours
# --------------------------------------------------

@app.get("/appearance-search/colors")
def get_supported_colors():
    """Return the list of colour labels the HSV classifier can return."""
    from search import SUPPORTED_COLORS
    return {
        "colors": sorted(SUPPORTED_COLORS),
        "note": (
            "Colours are estimated via HSV analysis. "
            "A trained MobileNetV3 classifier can be substituted later."
        ),
    }


# --------------------------------------------------
# APPEARANCE SEARCH
# --------------------------------------------------

@app.post("/appearance-search")
def appearance_search(
    request: AppearanceQueryRequest,
    http_request: Request,
):
    """
    Search for people by upper-body clothing colour.

    Query examples:
      "orange T-shirt"
      "blue shirt"
      "black jacket"

    If video_id is None the search spans all uploaded videos.
    """
    try:
        raw_results = search_appearances(
            query=request.query,
            video_id=request.video_id,
            color_confidence_min=request.confidence_min,
            limit=request.limit,
        )
    except Exception:
        logger.exception("Appearance search failed.")
        raise HTTPException(
            status_code=500,
            detail="Appearance search failed.",
        )

    if not raw_results:
        return {
            "found": False,
            "results": [],
            "total": 0,
            "query": request.query,
        }

    base_url = str(http_request.base_url).rstrip("/")

    formatted: list[dict] = []
    for obs in raw_results:

        # Build crop URL
        crop_url: str | None = None
        if obs.get("crop_path"):
            # crop_path is an absolute filesystem path;
            # derive a relative path from CROPS_DIR
            try:
                rel = os.path.relpath(
                    obs["crop_path"], CROPS_DIR
                ).replace("\\", "/")
                crop_url = f"{base_url}/crops/{rel}"
            except ValueError:
                # Cross-drive relpath on Windows raises ValueError
                crop_url = None

        # Build clip URL for the video at this timestamp
        clip_url: str | None = None
        if obs.get("timestamp") is not None and obs.get("video_id"):
            ts_tag = f"{obs['timestamp']:.3f}".replace(".", "_")
            clip_filename = (
                f"{obs['video_id']}_{ts_tag}_720p.mp4"
            )
            clip_path = os.path.join(PROCESSED_DIR, clip_filename)

            if os.path.exists(clip_path):
                clip_url = f"{base_url}/static/{clip_filename}"

        formatted.append({
            "video_id": obs["video_id"],
            "filename": obs.get("filename"),
            "track_id": obs.get("track_id"),
            "timestamp": obs.get("timestamp"),
            "frame_number": obs.get("frame_number"),
            "upper_color": obs.get("upper_color"),
            "upper_color_confidence": obs.get("upper_color_confidence"),
            "crop_url": crop_url,
            "clip_url": clip_url,
        })

    return {
        "found": True,
        "results": formatted,
        "total": len(formatted),
        "query": request.query,
    }

