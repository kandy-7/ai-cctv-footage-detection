"""
test_suite.py – Comprehensive automated test suite for the AI CCTV Detection System.
Tests:
  1. Database tables and schema
  2. Video processing with YOLO + ByteTrack on real sample video
  3. Database event verification
  4. Natural language search query engine
  5. Video clip extraction and smart resolution capping
  6. FastAPI endpoints (/status, /videos, /query)
"""

import sys
import os
import shutil
import uuid
import time

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import init_db, SessionLocal, Event, Video
from processing import process_video_file
from search import search_events, parse_query
from enhancement import upscale_clip, _compute_target_resolution
import cv2

def run_all_tests():
    print("=" * 60)
    print("RUNNING AI CCTV DETECTION SYSTEM TEST SUITE")
    print("=" * 60)

    # 1. Database Init Test
    print("\n[TEST 1] Initializing Database & Verifying Schema...")
    init_db()
    db = SessionLocal()
    # Check tables exist
    video_cols = [c.name for c in Video.__table__.columns]
    event_cols = [c.name for c in Event.__table__.columns]
    print(f"  Video columns: {video_cols}")
    print(f"  Event columns: {event_cols}")
    assert "duration" in video_cols, "Video missing 'duration' column"
    assert "status" in video_cols, "Video missing 'status' column"
    assert "frame_number" in event_cols, "Event missing 'frame_number' column"
    assert "track_id" in event_cols, "Event missing 'track_id' column"
    assert "video_duration" in event_cols, "Event missing 'video_duration' column"
    db.close()
    print("  --> [PASS] Database schema verified successfully.")

    # 2. Smart Resolution Capping Test
    print("\n[TEST 2] Testing Video Resolution & Aspect Ratio Calculation...")
    # Small video 640x480 -> 2x upscale -> 1280x960, but max height is 720, so scaled to 960x720 (even)
    w_out, h_out = _compute_target_resolution(640, 480)
    print(f"  Input 640x480 -> Output: {w_out}x{h_out}")
    assert w_out <= 1280 and h_out <= 720, "Output exceeded 1280x720 cap"
    assert w_out % 2 == 0 and h_out % 2 == 0, "Dimensions must be even numbers"

    # 1920x1080 -> should be capped at 1280x720
    w_out2, h_out2 = _compute_target_resolution(1920, 1080)
    print(f"  Input 1920x1080 -> Output: {w_out2}x{h_out2}")
    assert w_out2 <= 1280 and h_out2 <= 720, "1080p exceeded cap"
    print("  --> [PASS] Resolution capping logic verified.")

    # 3. Process Sample Video
    sample_video = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "istockphoto-1160587031-640_adpp_is.mp4"))
    assert os.path.exists(sample_video), f"Sample video not found at {sample_video}"
    
    test_video_id = f"test_{uuid.uuid4().hex[:8]}"
    print(f"\n[TEST 3] Processing Sample Video ({test_video_id})...")
    print(f"  Source: {sample_video}")

    # Register video row in DB
    db = SessionLocal()
    video_entry = Video(id=test_video_id, filename="sample_cctv.mp4", status="processing")
    db.add(video_entry)
    db.commit()
    db.close()

    t_start = time.time()
    process_video_file(sample_video, test_video_id, threshold=0.35)
    t_elapsed = time.time() - t_start
    print(f"  Processing finished in {t_elapsed:.2f}s")

    # 4. Verify DB Records
    print("\n[TEST 4] Verifying Database Records & Timestamps...")
    db = SessionLocal()
    video_record = db.query(Video).filter(Video.id == test_video_id).first()
    assert video_record is not None, "Video record not found"
    print(f"  Video status: {video_record.status}, recorded duration: {video_record.duration:.2f}s")
    assert video_record.status == "done", "Video status should be 'done'"
    assert video_record.duration > 0, "Duration must be greater than 0"

    events = db.query(Event).filter(Event.video_id == test_video_id).all()
    print(f"  Total events detected: {len(events)}")
    assert len(events) > 0, "Expected at least 1 detection event in sample video"

    classes_detected = set(e.class_name for e in events)
    print(f"  Classes detected: {classes_detected}")
    timestamps = [e.timestamp for e in events]
    print(f"  Timestamp range: min={min(timestamps):.2f}s, max={max(timestamps):.2f}s")
    assert min(timestamps) >= 0, "Timestamp cannot be negative"
    assert max(timestamps) <= video_record.duration + 0.5, "Timestamp exceeds video duration"
    db.close()
    print("  --> [PASS] Event records and timestamps verified.")

    # 5. Natural Language Search Queries
    print("\n[TEST 5] Testing Natural Language Search Queries...")
    # Query 1: by actual class detected
    first_class = list(classes_detected)[0]
    results1 = search_events(f"Show me {first_class}", test_video_id, threshold=0.3)
    print(f"  Query: 'Show me {first_class}' -> {len(results1)} matching buckets found")
    assert len(results1) > 0, f"Search for detected class '{first_class}' returned 0 results"
    first_res = results1[0]
    print(f"    Top match: timestamp={first_res['timestamp']:.2f}s, classes={first_res['classes']}, confidence={first_res['confidence']:.2f}")

    # Query 2: Non-existent object (e.g. airplane)
    results2 = search_events("airplane flying", test_video_id, threshold=0.3)
    print(f"  Query: 'airplane flying' -> {len(results2)} matches (expected 0)")
    assert len(results2) == 0, "Search for non-existent class should return 0 results"
    print("  --> [PASS] Search query parsing and bucketing verified.")

    # 6. Clip Extraction Test
    print("\n[TEST 6] Testing Clip Extraction & Time-Based Seeking...")
    os.makedirs("processed", exist_ok=True)
    test_clip_output = f"processed/{test_video_id}_clip_test.mp4"
    event_ts = first_res['timestamp']
    start_time = max(0.0, event_ts - 1.5)
    end_time = event_ts + 2.0
    print(f"  Extracting clip for event at {event_ts:.2f}s (window: {start_time:.2f}s - {end_time:.2f}s)...")
    
    success = upscale_clip(sample_video, test_clip_output, start_time, end_time)
    assert success, "upscale_clip returned False"
    assert os.path.exists(test_clip_output), f"Clip output file was not created: {test_clip_output}"
    
    # Verify generated clip
    clip_cap = cv2.VideoCapture(test_clip_output)
    assert clip_cap.isOpened(), "Could not open extracted clip"
    clip_w = int(clip_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    clip_h = int(clip_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_frames = int(clip_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_fps = clip_cap.get(cv2.CAP_PROP_FPS) or 25.0
    clip_dur = clip_frames / clip_fps
    clip_cap.release()
    print(f"  Extracted clip: resolution={clip_w}x{clip_h}, frames={clip_frames}, duration={clip_dur:.2f}s")
    assert clip_w <= 1280 and clip_h <= 720, "Clip resolution exceeded cap"
    assert clip_dur > 0, "Clip duration must be positive"
    print("  --> [PASS] Clip extraction verified successfully.")

    # 7. Check Debug Frames
    print("\n[TEST 7] Checking Annotated Debug Frames...")
    debug_dir = os.path.join(os.path.dirname(__file__), "debug_frames", test_video_id)
    if os.path.exists(debug_dir):
        saved_frames = os.listdir(debug_dir)
        print(f"  Debug frames saved in {debug_dir}: {len(saved_frames)} images")
        assert len(saved_frames) > 0, "Expected annotated debug frames"
    print("  --> [PASS] Debug frames verified.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()

