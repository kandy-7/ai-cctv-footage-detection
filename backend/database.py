from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Ensure backend directory context or absolute path logic
# For this setup, we assume running from backend/ or passing absolute path
DB_PATH = "sqlite:///./cctv_events.db"

engine = create_engine(
    DB_PATH, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, index=True) # UUID of uploaded video
    class_name = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(Float) # Time in seconds from start
    frame_path = Column(String) # Path to saved frame image
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
