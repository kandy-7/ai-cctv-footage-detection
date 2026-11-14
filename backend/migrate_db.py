"""
migrate_db.py – Safe, idempotent SQLite migration script for AI CCTV Footage Detection System.

Performs:
1. Locates SQLite databases (backend/cctv_events.db and ./cctv_events.db if present).
2. Inspects PRAGMA table_info(events).
3. Adds missing columns using ALTER TABLE:
   - frame_number INTEGER (nullable)
   - track_id INTEGER (nullable)
   - video_duration FLOAT (nullable)
4. Checks whether table 'videos' exists and creates it if missing.
5. Backfills 'videos' table with past video_ids found in 'events' so that GET /videos and GET /status/{video_id} work for all existing records.
6. Creates helpful indices without destroying data.
7. Verifies migration and preserves all existing rows.
"""

import sqlite3
import os
import sys

def migrate_database(db_path: str):
    abs_path = os.path.abspath(db_path)
    if not os.path.exists(abs_path):
        print(f"[SKIP] Database not found at: {abs_path}")
        return

    print("=" * 65)
    print(f"[MIGRATION] Target: {abs_path}")
    print(f"            Initial size: {os.path.getsize(abs_path)} bytes")
    print("=" * 65)

    conn = sqlite3.connect(abs_path)
    cursor = conn.cursor()

    # 1. Inspect existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = set(row[0] for row in cursor.fetchall())
    print(f"  Existing tables: {existing_tables}")

    # 2. Ensure 'videos' table exists
    if "videos" not in existing_tables:
        print("  -> Creating missing 'videos' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id VARCHAR PRIMARY KEY,
                filename VARCHAR,
                duration FLOAT,
                status VARCHAR DEFAULT 'processing',
                created_at DATETIME
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_videos_id ON videos (id);")
        conn.commit()
        print("     [OK] 'videos' table created.")
    else:
        print("  -> 'videos' table already exists.")

    # 3. Inspect 'events' table
    if "events" not in existing_tables:
        print("  -> Creating missing 'events' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id VARCHAR,
                class_name VARCHAR,
                confidence FLOAT,
                timestamp FLOAT,
                frame_number INTEGER,
                track_id INTEGER,
                video_duration FLOAT,
                frame_path VARCHAR DEFAULT '',
                created_at DATETIME
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_events_video_id ON events (video_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_events_class_name ON events (class_name);")
        conn.commit()
        print("     [OK] 'events' table created.")
    else:
        cursor.execute("PRAGMA table_info(events);")
        columns_info = cursor.fetchall()
        existing_cols = {col[1]: col[2] for col in columns_info}
        print(f"  Current 'events' columns: {list(existing_cols.keys())}")

        # Count existing rows before migration
        cursor.execute("SELECT count(*) FROM events;")
        initial_event_count = cursor.fetchone()[0]
        print(f"  Existing records in 'events': {initial_event_count}")

        # Required columns and their SQLite types
        required_columns = {
            "frame_number": "INTEGER",
            "track_id": "INTEGER",
            "video_duration": "FLOAT",
        }

        # 4. Add missing columns safely using ALTER TABLE
        for col_name, col_type in required_columns.items():
            if col_name not in existing_cols:
                alter_sql = f"ALTER TABLE events ADD COLUMN {col_name} {col_type};"
                print(f"  -> Adding missing column: {col_name} ({col_type})...")
                cursor.execute(alter_sql)
                conn.commit()
                print(f"     [OK] Column '{col_name}' added successfully.")
            else:
                print(f"  -> Column '{col_name}' already exists. Skipping.")

        # 5. Create helpful indices
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_events_video_id ON events (video_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_events_class_name ON events (class_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_events_video_frame ON events (video_id, frame_number);")
        conn.commit()
        print("  -> Indices verified/created.")

        # Verify record count preserved
        cursor.execute("SELECT count(*) FROM events;")
        final_event_count = cursor.fetchone()[0]
        assert final_event_count == initial_event_count, f"Row count mismatch! {final_event_count} != {initial_event_count}"
        print(f"  -> Record preservation verified: {final_event_count} rows preserved.")

    # 6. Backfill 'videos' table with past video_ids from 'events' if missing
    cursor.execute("SELECT DISTINCT video_id FROM events WHERE video_id IS NOT NULL;")
    distinct_vids = [r[0] for r in cursor.fetchall()]
    print(f"  Distinct video_ids in events: {len(distinct_vids)}")

    backfilled = 0
    uploads_dir = os.path.abspath(os.path.join(os.path.dirname(abs_path), "uploads"))
    if not os.path.exists(uploads_dir):
        uploads_dir = os.path.abspath("uploads")

    upload_files = os.listdir(uploads_dir) if os.path.exists(uploads_dir) else []

    for vid in distinct_vids:
        cursor.execute("SELECT id FROM videos WHERE id = ?;", (vid,))
        if cursor.fetchone() is None:
            # Find matching upload filename if present
            matched_filename = f"{vid}.mp4"
            for uf in upload_files:
                if uf.startswith(vid):
                    matched_filename = uf[len(vid)+1:] if len(uf) > len(vid)+1 else uf
                    break

            # Calculate duration and earliest created_at from events
            cursor.execute("""
                SELECT MAX(timestamp), MIN(created_at) FROM events WHERE video_id = ?;
            """, (vid,))
            row = cursor.fetchone()
            calc_duration = (row[0] + 1.0) if (row and row[0] is not None) else None
            earliest_created = row[1] if (row and row[1] is not None) else None

            cursor.execute("""
                INSERT INTO videos (id, filename, duration, status, created_at)
                VALUES (?, ?, ?, 'done', ?);
            """, (vid, matched_filename, calc_duration, earliest_created))
            backfilled += 1

    if backfilled > 0:
        conn.commit()
        print(f"  -> Backfilled {backfilled} past video record(s) into 'videos' table.")
    else:
        print("  -> 'videos' table is already synchronized with existing events.")

    # 7. Final PRAGMA check
    cursor.execute("PRAGMA table_info(events);")
    post_cols = [c[1] for c in cursor.fetchall()]
    print(f"\n[VERIFIED] Final columns for 'events': {post_cols}")
    cursor.execute("SELECT count(*) FROM events;")
    print(f"[VERIFIED] Final row count for 'events': {cursor.fetchone()[0]}")
    cursor.execute("SELECT count(*) FROM videos;")
    print(f"[VERIFIED] Final row count for 'videos': {cursor.fetchone()[0]}")

    conn.close()
    print("=" * 65 + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Migrate backend/cctv_events.db (primary)
    primary_db = os.path.join(base_dir, "cctv_events.db")
    migrate_database(primary_db)

    # Migrate root cctv_events.db if present
    root_db = os.path.abspath(os.path.join(base_dir, "..", "cctv_events.db"))
    if os.path.exists(root_db) and os.path.normpath(root_db) != os.path.normpath(primary_db):
        migrate_database(root_db)
