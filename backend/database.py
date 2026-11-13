from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Database path - relative to backend/ directory
DB_PATH = "sqlite:///./cctv_events.db"

engine = create_engine(
    DB_PATH, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Video(Base):
    """Tracks uploaded videos and their processing status."""
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True)  # UUID
    filename = Column(String)
    duration = Column(Float, nullable=True)             # Total video duration in seconds
    status = Column(String, default="processing")       # "processing" | "done"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Event(Base):
    """Stores per-frame detection events."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, index=True)               # UUID of uploaded video
    class_name = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(Float)                           # Actual time in seconds (float)
    frame_number = Column(Integer, nullable=True)       # Actual decoded frame index
    track_id = Column(Integer, nullable=True)           # ByteTrack track ID
    video_duration = Column(Float, nullable=True)       # Denormalised copy of video duration
    frame_path = Column(String, default="")             # Path to saved annotated debug frame
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Prevent duplicate inserts for same video + frame + class
    __table_args__ = (
        UniqueConstraint("video_id", "frame_number", "class_name",
                         name="uq_event_video_frame_class"),
    )


def init_db():
    """Create all tables (safe to call multiple times)."""
    Base.metadata.create_all(bind=engine)
