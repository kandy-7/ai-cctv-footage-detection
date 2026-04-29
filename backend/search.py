from database import SessionLocal, Event
from sqlalchemy.orm import Session
import logging

# Simple mapping of common terms to COCO classes (YOLOv8 default)
YOLO_CLASSES = {
    "person": ["person", "man", "woman", "human", "pedestrian"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "car": ["car", "vehicle", "taxi", "automobile"],
    "motorcycle": ["motorcycle", "motorbike", "bike"],
    "bus": ["bus"],
    "truck": ["truck", "lorry"],
    "cat": ["cat", "kitten"],
    "dog": ["dog", "puppy"],
    "backpack": ["backpack", "bag"],
    "handbag": ["handbag", "purse"],
    # Add more as needed
}

EVENT_KEYWORDS = {
    "accident": ["accident", "crash", "collision", "dash", "hit", "emergency"],
}

def parse_query(query: str):
    """
    Extracts keywords from query and maps to YOLO classes.
    """
    query = query.lower()
    targets = []
    
    # Reverse lookup
    for coco_class, synonyms in YOLO_CLASSES.items():
        for syn in synonyms:
            if syn in query:
                targets.append(coco_class)
                break # Found one match for this class
            
    return targets

def search_events(query: str, video_id: str, threshold: float = 0.5):
    query_lower = query.lower()
    targets = parse_query(query_lower)
    
    is_event_query = any(kw in query_lower for synonyms in EVENT_KEYWORDS.values() for kw in synonyms)
    
    # If "and" is in query, we might want to ensure multiple targets are present
    require_multiple = " and " in query_lower or is_event_query
    
    db = SessionLocal()
    
    # If it's an event query (accident/dash), focus on vehicles/people
    if is_event_query and not targets:
        targets = ["car", "motorcycle", "bicycle", "truck", "bus", "person"]

    # Get search window - all events for this video
    all_detections = db.query(Event).filter(
        Event.video_id == video_id,
        Event.confidence >= (threshold * 0.8) # Lower threshold slightly for events
    ).order_by(Event.timestamp).all()
    
    db.close()

    if not all_detections:
        return []

    # Group detections by timestamp (1-second buckets)
    buckets = {}
    for det in all_detections:
        t_key = round(det.timestamp)
        if t_key not in buckets:
            buckets[t_key] = []
        buckets[t_key].append(det)

    results = []
    for t_key, dets in buckets.items():
        found_classes = {d.class_name for d in dets}
        
        # Check if this bucket matches the search
        score = 1
        matches_targets = False
        
        if targets:
            matching_classes = [t for t in targets if t in found_classes]
            if matching_classes:
                matches_targets = True
                score += len(matching_classes) * 5
            
            if require_multiple:
                # If we require multiple objects (e.g. "car and bike" or "accident")
                # and only found one thing, check if we have multiple detections in the frame
                if len(dets) < 2:
                    continue
        else:
            # If no specific targets found, but it's an event query, keep it if multiple detections
            if is_event_query and len(dets) >= 2:
                matches_targets = True
            else:
                continue

        if not matches_targets:
            continue
            
        # Boost score if multiple unique classes (likely collision)
        if len(found_classes) >= 2:
            score += 20
        
        # Boost if lots of detections (chaos)
        score += len(dets) * 2

        results.append({
            "timestamp": t_key,
            "classes": list(found_classes),
            "score": score,
            "count": len(dets)
        })

    # Sort results by score (descending) then timestamp
    results.sort(key=lambda x: (-x["score"], x["timestamp"]))
    
    return results
