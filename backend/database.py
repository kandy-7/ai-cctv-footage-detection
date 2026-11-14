
import os
import sqlite3
import logging
import datetime

from contextlib import closing
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Index,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import URL



logger = logging.getLogger(__name__)

# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BACKEND_DIR, "cctv_events.db")

DB_PATH = URL.create(
    "sqlite",
    database=DB_FILE
)

engine = create_engine(
    DB_PATH,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# --------------------------------------------------
# DATABASE MODELS
# --------------------------------------------------

class Video(Base):
    """Stores uploaded video information and processing status."""

    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    duration = Column(Float, nullable=True)
    status = Column(String, default="processing", nullable=False)
    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )


class Event(Base):
    """Stores object detection events."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)

    video_id = Column(String, index=True)
    class_name = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(Float)

    frame_number = Column(Integer, nullable=True)
    track_id = Column(Integer, nullable=True)
    video_duration = Column(Float, nullable=True)

    frame_path = Column(String, default="")

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    __table_args__ = (
        Index(
            "ix_events_video_frame",
            "video_id",
            "frame_number"
        ),
    )


class AppearanceObservation(Base):
    """
    Stores per-track clothing appearance observations extracted during
    YOLO person detection. One row per saved crop.

    upper_color / lower_color are human-readable colour names
    (e.g. 'orange', 'blue', 'black', 'other', 'unknown').
    model_version records which classifier produced the label.
    """

    __tablename__ = "appearance_observations"

    id = Column(Integer, primary_key=True, index=True)

    video_id = Column(String, index=True, nullable=False)
    track_id = Column(Integer, nullable=True, index=True)

    timestamp = Column(Float, index=True)
    frame_number = Column(Integer, nullable=True)

    upper_color = Column(String, index=True, default="unknown")
    upper_color_confidence = Column(Float, default=0.0)

    lower_color = Column(String, nullable=True)

    crop_path = Column(String, default="")

    model_version = Column(String, default="hsv_baseline_v1")

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    __table_args__ = (
        Index(
            "ix_appearance_video_track",
            "video_id",
            "track_id"
        ),
    )


# --------------------------------------------------
# SQLITE MIGRATIONS
# --------------------------------------------------

def run_migrations(db_file_path: str = DB_FILE):
    """
    Add missing columns to existing SQLite tables.
    Never drop or recreate existing event records.
    """

    if not os.path.exists(db_file_path):
        raise FileNotFoundError(
            f"Database file not found: {db_file_path}"
        )

    logger.info("Checking SQLite database schema...")

    try:
        with closing(sqlite3.connect(db_file_path)) as conn:
            with conn:

                cursor = conn.cursor()

                # ----------------------------------
                # EVENTS TABLE
                # ----------------------------------

                cursor.execute("""
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table' AND name='events';
                """)

                if cursor.fetchone():

                    cursor.execute("PRAGMA table_info(events)")

                    existing_columns = {
                        row[1] for row in cursor.fetchall()
                    }

                    required_columns = {
                        "frame_number": "INTEGER",
                        "track_id": "INTEGER",
                        "video_duration": "FLOAT",
                    }

                    for name, column_type in required_columns.items():

                        if name not in existing_columns:

                            logger.info(
                                "Adding missing events column: %s",
                                name
                            )

                            cursor.execute(
                                f"ALTER TABLE events "
                                f"ADD COLUMN {name} {column_type}"
                            )

                    # Indexes
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS
                        ix_events_video_id
                        ON events(video_id)
                    """)

                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS
                        ix_events_class_name
                        ON events(class_name)
                    """)

                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS
                        ix_events_video_frame
                        ON events(video_id, frame_number)
                    """)

                # ----------------------------------
                # VIDEOS TABLE
                # ----------------------------------

                cursor.execute("""
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table' AND name='videos'
                """)

                if not cursor.fetchone():

                    logger.info("Creating videos table.")

                    cursor.execute("""
                        CREATE TABLE videos (
                            id VARCHAR PRIMARY KEY,
                            filename VARCHAR,
                            duration FLOAT,
                            status VARCHAR DEFAULT 'processing',
                            created_at DATETIME
                        )
                    """)

                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS
                        ix_videos_id
                        ON videos(id)
                    """)

                # ----------------------------------
                # APPEARANCE_OBSERVATIONS TABLE
                # ----------------------------------

                cursor.execute("""
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table' AND name='appearance_observations'
                """)

                if not cursor.fetchone():

                    logger.info("Creating appearance_observations table.")

                    cursor.execute("""
                        CREATE TABLE appearance_observations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            video_id VARCHAR NOT NULL,
                            track_id INTEGER,
                            timestamp FLOAT,
                            frame_number INTEGER,
                            upper_color VARCHAR DEFAULT 'unknown',
                            upper_color_confidence FLOAT DEFAULT 0.0,
                            lower_color VARCHAR,
                            crop_path VARCHAR DEFAULT '',
                            model_version VARCHAR DEFAULT 'hsv_baseline_v1',
                            created_at DATETIME
                        )
                    """)

                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS
                        ix_appearance_video_id
                        ON appearance_observations(video_id)
                    """)

                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS
                        ix_appearance_upper_color
                        ON appearance_observations(upper_color)
                    """)

                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS
                        ix_appearance_video_track
                        ON appearance_observations(video_id, track_id)
                    """)

        logger.info("SQLite migration completed.")

    except Exception:
        logger.exception("SQLite migration failed.")
        raise


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

def init_db():

    logger.info("Initializing CCTV database...")

    # Create missing tables without replacing existing tables.
    Base.metadata.create_all(bind=engine)

    # Migrate existing tables.
    run_migrations(DB_FILE)

    # Verify event schema.
    with closing(sqlite3.connect(DB_FILE)) as conn:

        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(events)")

        actual_columns = {
            row[1] for row in cursor.fetchall()
        }

        required_columns = {
            "id",
            "video_id",
            "class_name",
            "confidence",
            "timestamp",
            "frame_number",
            "track_id",
            "video_duration",
            "frame_path",
            "created_at",
        }

        missing = required_columns - actual_columns

        if missing:
            raise RuntimeError(
                f"Database schema incomplete: {missing}"
            )

        # Verify appearance_observations table exists.
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='appearance_observations'
        """)

        if not cursor.fetchone():
            raise RuntimeError(
                "appearance_observations table missing after migration."
            )

    logger.info("CCTV database initialized successfully.")
