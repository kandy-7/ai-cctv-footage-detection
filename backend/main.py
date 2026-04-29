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
from database import init_db

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

# Initialize DB
init_db()

app = FastAPI(title="CCTV Analysis System", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="processed"), name="static")

class QueryRequest(BaseModel):
    video_id: str
    query: str
    threshold: float = 0.5

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    file_path = f"uploads/{video_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Trigger processing in background
    background_tasks.add_task(process_video_file, file_path, video_id)
    
    return {"video_id": video_id, "message": "Video uploaded and processing started."}

@app.post("/query")
async def query_video(request: QueryRequest):
    event_results = search_events(request.query, request.video_id, request.threshold)
    
    if not event_results:
        return {"found": False, "message": "The requested event did not occur in the uploaded video."}
    
    # Return all found events to let the user select
    # For each event, we generate a video URL
    processed_events = []
    
    # Find original file (naive search)
    input_file = ""
    for f in os.listdir("uploads"):
        if f.startswith(request.video_id):
            input_file = os.path.join("uploads", f)
            break
            
    if not input_file:
        raise HTTPException(status_code=404, detail="Video file not found")
        
    for event in event_results[:10]: # Limit to top 10 results
        timestamp = event["timestamp"]
        start_time = max(0, timestamp - 2)
        end_time = timestamp + 3 # 5 second clip
        
        output_filename = f"{request.video_id}_{int(timestamp)}_4k.mp4"
        output_path = f"processed/{output_filename}"
        
        # Check if we already processed this specific clip
        if not os.path.exists(output_path):
            upscale_clip(input_file, output_path, start_time, end_time)
            
        processed_events.append({
            "timestamp": timestamp,
            "video_url": f"http://localhost:8001/static/{output_filename}",
            "classes": event["classes"],
            "count": event["count"]
        })
    
    return {
        "found": True, 
        "events": processed_events,
        "total_matches": len(event_results)
    }

@app.get("/")
def read_root():
    return {"message": "CCTV AI System Ready"}
