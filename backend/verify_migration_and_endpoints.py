"""
verify_migration_and_endpoints.py
Systematic validation of:
1. PRAGMA table_info(events)
2. Required columns presence
3. Accessibility of existing CCTV event records
4. GET /status/{video_id} on existing records (HTTP 200)
5. Video upload and processing
6. Verification of new event records with frame_number and track_id
7. GET /status/{video_id} on newly processed video (HTTP 200)
8. Idempotent repeated migration check
"""

import sys
import os
import sqlite3
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import DB_FILE, init_db, SessionLocal, Event, Video
from starlette.testclient import TestClient
from main import app

def run_checks():
    print("=" * 65)
    print("MIGRATION & ENDPOINT VERIFICATION")
    print("=" * 65)

    # 1. PRAGMA check on actual database
    print("\n[STEP 1] Checking SQLite Schema via PRAGMA...")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(events);")
    columns_info = cur.fetchall()
    col_names = [c[1] for c in columns_info]
    print(f"  Database file: {DB_FILE}")
    print(f"  'events' columns ({len(col_names)}): {col_names}")
    
    # 2. Verify all required columns exist
    print("\n[STEP 2] Verifying Required Columns...")
    required = ["frame_number", "track_id", "video_duration"]
    for req in required:
        assert req in col_names, f"Required column '{req}' is MISSING in events table!"
        print(f"  -> Column '{req}': EXISTS [OK]")

    # 3. Verify old event records are still accessible
    print("\n[STEP 3] Verifying Preservation & Accessibility of Existing Event Records...")
    cur.execute("SELECT count(*) FROM events;")
    total_events = cur.fetchone()[0]
    print(f"  Total events in DB: {total_events}")
    assert total_events > 0, "Database should have existing records"
    
    # Check that old records can be loaded by SQLAlchemy
    db = SessionLocal()
    old_event = db.query(Event).first()
    assert old_event is not None, "Could not fetch event via SQLAlchemy"
    print(f"  Sample existing event:")
    print(f"    id: {old_event.id}, video_id: {old_event.video_id}, class: {old_event.class_name}")
    print(f"    confidence: {old_event.confidence}, timestamp: {old_event.timestamp}")
    print(f"    frame_number: {old_event.frame_number}, track_id: {old_event.track_id}")
    db.close()
    conn.close()
    print("  -> Existing records fully intact and accessible [OK]")

    # 4. Test GET /status/{video_id} on existing video
    print("\n[STEP 4] Testing GET /status/{video_id} on Existing Video...")
    client = TestClient(app)
    existing_video_id = old_event.video_id
    response = client.get(f"/status/{existing_video_id}")
    print(f"  Request: GET /status/{existing_video_id}")
    print(f"  Status code: {response.status_code}")
    print(f"  Response JSON: {response.json()}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert response.json()["status"] in ["processing", "done"], "Invalid status in response"
    assert "event_count" in response.json(), "Missing 'event_count' in response"
    print("  -> Status endpoint returned HTTP 200 successfully! [OK]")

    # 5. Test GET /videos endpoint
    print("\n[STEP 5] Testing GET /videos...")
    v_resp = client.get("/videos")
    assert v_resp.status_code == 200
    videos_list = v_resp.json()
    print(f"  Total registered videos: {len(videos_list)}")
    assert len(videos_list) > 0

    # 6. Test Video Upload and Processing
    print("\n[STEP 6] Testing Video Upload & Processing...")
    sample_video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "istockphoto-948679414-640_adpp_is.mp4"))
    assert os.path.exists(sample_video_path), f"Sample video not found at {sample_video_path}"

    with open(sample_video_path, "rb") as vf:
        upload_resp = client.post(
            "/upload",
            files={"file": ("test_upload_cctv.mp4", vf, "video/mp4")}
        )
    print(f"  Upload response: {upload_resp.status_code} -> {upload_resp.json()}")
    assert upload_resp.status_code == 200
    new_video_id = upload_resp.json()["video_id"]

    # Test GET /status immediately after upload (before background processing finishes)
    stat_immediate = client.get(f"/status/{new_video_id}")
    print(f"  GET /status/{new_video_id} (immediate): {stat_immediate.status_code} -> {stat_immediate.json()}")
    assert stat_immediate.status_code == 200
    assert stat_immediate.json()["status"] == "processing"

    # Now run processing on this uploaded file to verify event insertion with frame_number & track_id
    print("\n[STEP 7] Running Processing on Uploaded Video...")
    from processing import process_video_file
    uploaded_file_path = os.path.join("uploads", f"{new_video_id}_test_upload_cctv.mp4")
    process_video_file(uploaded_file_path, new_video_id, threshold=0.35)

    # 7. Verify new events in database have frame_number and track_id
    print("\n[STEP 8] Verifying New Events Have frame_number and track_id...")
    db = SessionLocal()
    new_events = db.query(Event).filter(Event.video_id == new_video_id).all()
    print(f"  Total events detected for {new_video_id}: {len(new_events)}")
    assert len(new_events) > 0, "No events detected"
    sample_new_event = new_events[0]
    print(f"  Sample newly detected event:")
    print(f"    id: {sample_new_event.id}, class: {sample_new_event.class_name}")
    print(f"    timestamp: {sample_new_event.timestamp:.2f}s")
    print(f"    frame_number: {sample_new_event.frame_number}")
    print(f"    track_id: {sample_new_event.track_id}")
    print(f"    video_duration: {sample_new_event.video_duration}")
    assert sample_new_event.frame_number is not None, "frame_number should NOT be None for new events"
    assert sample_new_event.track_id is not None, "track_id should NOT be None for new events"
    assert sample_new_event.video_duration is not None, "video_duration should NOT be None"
    db.close()
    print("  -> New events correctly store frame_number, track_id, and video_duration! [OK]")

    # 8. Test GET /status/{video_id} after processing
    print("\n[STEP 9] Testing GET /status/{video_id} After Processing...")
    stat_done = client.get(f"/status/{new_video_id}")
    print(f"  GET /status/{new_video_id} (after processing): {stat_done.status_code} -> {stat_done.json()}")
    assert stat_done.status_code == 200
    assert stat_done.json()["status"] == "done"
    assert stat_done.json()["event_count"] == len(new_events)
    assert stat_done.json()["duration"] > 0
    print("  -> Status returns HTTP 200 with status='done' and accurate event_count! [OK]")

    # 9. Test Query Endpoint on the new video
    print("\n[STEP 10] Testing POST /query on Newly Processed Video...")
    q_resp = client.post(
        "/query",
        json={"video_id": new_video_id, "query": sample_new_event.class_name, "threshold": 0.3}
    )
    print(f"  POST /query for class '{sample_new_event.class_name}': {q_resp.status_code} -> found: {q_resp.json().get('found')}")
    assert q_resp.status_code == 200
    assert q_resp.json()["found"] is True

    # 10. Test Idempotency (running init_db() again)
    print("\n[STEP 11] Testing Idempotent Migration Safety (Re-running init_db())...")
    init_db()
    init_db()
    print("  -> init_db() called multiple times without errors [OK]")

    print("\n" + "=" * 65)
    print("ALL MIGRATION AND ENDPOINT TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)

if __name__ == "__main__":
    run_checks()
