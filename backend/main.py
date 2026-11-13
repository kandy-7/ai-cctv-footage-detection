import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import logging

from processing import process_video_file
from search import search_events
from enhancement import upscale_clip
from database import init_db, SessionLocal, Event, Video

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

# Initialize DB (creates tables if missing)
init_db()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="CCTV Analysis System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="processed"), name="static")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    video_id: str
    query: str
    threshold: float = 0.5


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "CCTV AI System Ready", "version": "2.0.0"}


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accept a video file, persist it, create a Video record, and kick off
    background processing.  Returns video_id and original filename.
    """
    video_id = str(uuid.uuid4())
    original_filename = file.filename or "video.mp4"
    file_path = f"uploads/{video_id}_{original_filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create a Video row so /status can be queried immediately
    db = SessionLocal()
    try:
        new_video = Video(
            id=video_id,
            filename=original_filename,
            status="processing",
        )
        db.add(new_video)
        db.commit()
    finally:
        db.close()

    background_tasks.add_task(process_video_file, file_path, video_id)

    return {
        "video_id": video_id,
        "filename": original_filename,
        "message": "Video uploaded and processing started.",
    }


@app.get("/status/{video_id}")
def get_status(video_id: str):
    """
    Poll processing status for a given video.

    Returns:
        status      – "processing" | "done"
        event_count – number of detection events stored so far
        duration    – total video duration in seconds (None until done)
    """
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        event_count = db.query(Event).filter(Event.video_id == video_id).count()

        return {
            "video_id": video_id,
            "status": video.status,
            "event_count": event_count,
            "duration": video.duration,
        }
    finally:
        db.close()


@app.get("/videos")
def list_videos():
    """
    List all uploaded videos with their processing status.
    """
    db = SessionLocal()
    try:
        videos = db.query(Video).order_by(Video.created_at.desc()).all()
        return [
            {
                "video_id": v.id,
                "filename": v.filename,
                "status": v.status,
                "duration": v.duration,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in videos
        ]
    finally:
        db.close()


@app.post("/query")
async def query_video(request: QueryRequest):
    """
    Search for events matching *query* in *video_id* and return
    short extracted clips (capped at 720p) for browser playback.
    """
    event_results = search_events(request.query, request.video_id, request.threshold)

    if not event_results:
        return {"found": False, "message": "The requested event did not occur in the uploaded video."}

    # Locate original upload
    input_file = ""
    for f in os.listdir("uploads"):
        if f.startswith(request.video_id):
            input_file = os.path.join("uploads", f)
            break

    if not input_file:
        raise HTTPException(status_code=404, detail="Video file not found")

    processed_events = []

    for event in event_results[:10]:  # Limit to top 10 results
        # Use the ACTUAL float timestamp (not an integer key)
        timestamp: float = float(event["timestamp"])
        start_time = max(0.0, timestamp - 2.0)
        end_time = timestamp + 3.0  # 5-second clip

        # Clip filename encodes fractional seconds to avoid collisions
        ts_tag = f"{timestamp:.3f}".replace(".", "_")
        output_filename = f"{request.video_id}_{ts_tag}_720p.mp4"
        output_path = f"processed/{output_filename}"

        if not os.path.exists(output_path):
            # upscale_clip now caps output at 720p (handled in enhancement.py)
            upscale_clip(input_file, output_path, start_time, end_time)

        processed_events.append({
            "timestamp": timestamp,
            "video_url": f"http://localhost:8001/static/{output_filename}",
            "classes": event["classes"],
            "count": event["count"],
            "confidence": event.get("confidence"),
        })

    return {
        "found": True,
        "events": processed_events,
        "total_matches": len(event_results),
    }
