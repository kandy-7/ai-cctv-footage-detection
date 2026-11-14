"""
search.py – Query engine for CCTV detection events.

Fixes over original:
- 2-second buckets instead of 1-second (groups temporally close detections)
- Returns actual float timestamp (highest-confidence detection in bucket), not rounded int
- require_multiple only triggered when query explicitly has " and " between two objects
- Confidence returned in result for frontend display
"""

from database import SessionLocal, Event, AppearanceObservation, Video
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary: maps COCO class names → query synonyms
# ---------------------------------------------------------------------------
YOLO_CLASSES = {
    "person":     ["person", "man", "woman", "human", "pedestrian"],
    "bicycle":    ["bicycle", "bike", "cycle"],
    "car":        ["car", "vehicle", "taxi", "automobile"],
    "motorcycle": ["motorcycle", "motorbike"],
    "bus":        ["bus"],
    "truck":      ["truck", "lorry"],
    "cat":        ["cat", "kitten"],
    "dog":        ["dog", "puppy"],
    "backpack":   ["backpack", "bag"],
    "handbag":    ["handbag", "purse"],
}

EVENT_KEYWORDS = {
    "accident": ["accident", "crash", "collision", "dash", "hit", "emergency"],
}

BUCKET_WINDOW = 2.0  # seconds – group detections within this window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_query(query: str):
    """Return list of COCO class names mentioned in *query*."""
    query = query.lower()
    targets = []
    for coco_class, synonyms in YOLO_CLASSES.items():
        for syn in synonyms:
            if syn in query:
                targets.append(coco_class)
                break
    return targets


def _bucket_key(timestamp: float) -> int:
    """Map a float timestamp to a 2-second bucket index."""
    return int(timestamp / BUCKET_WINDOW)


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search_events(query: str, video_id: str, threshold: float = 0.5):
    """
    Search stored detection events for *video_id* matching *query*.

    Returns a list of dicts (sorted best-match first), each with:
      timestamp  – float, actual time of best detection in this bucket
      classes    – list of unique class names found
      score      – relevance score (higher = better)
      count      – number of raw detections in bucket
      confidence – highest confidence in bucket
    """
    query_lower = query.lower()
    targets = parse_query(query_lower)
    is_event_query = any(
        kw in query_lower for synonyms in EVENT_KEYWORDS.values() for kw in synonyms
    )

    # require_multiple ONLY when query explicitly says "X and Y"
    # (not for event queries – those may be single-vehicle)
    require_multiple = " and " in query_lower and len(targets) >= 2

    db = SessionLocal()
    try:
        all_detections = (
            db.query(Event)
            .filter(
                Event.video_id == video_id,
                Event.confidence >= threshold * 0.8,
            )
            .order_by(Event.timestamp)
            .all()
        )
    finally:
        db.close()

    if not all_detections:
        return []

    # Fallback targets for event-only queries ("show me accidents")
    if is_event_query and not targets:
        targets = ["car", "motorcycle", "bicycle", "truck", "bus", "person"]

    # ------------------------------------------------------------------
    # Group into 2-second buckets
    # ------------------------------------------------------------------
    buckets: dict[int, list] = {}
    for det in all_detections:
        bk = _bucket_key(det.timestamp)
        buckets.setdefault(bk, []).append(det)

    results = []
    for bk, dets in buckets.items():
        found_classes = {d.class_name for d in dets}
        best_det = max(dets, key=lambda d: d.confidence)   # highest-conf detection
        best_confidence = float(best_det.confidence)
        actual_timestamp = float(best_det.timestamp)        # real float, not rounded

        score = 1
        matches_targets = False

        if targets:
            matching_classes = [t for t in targets if t in found_classes]
            if matching_classes:
                matches_targets = True
                score += len(matching_classes) * 5

                if require_multiple:
                    # Need at least two distinct target classes present
                    if len(matching_classes) < 2:
                        continue
        else:
            # No specific targets – for event queries accept any bucket with detections
            if is_event_query:
                matches_targets = True
            else:
                continue  # Non-event query with no recognisable targets → skip

        if not matches_targets:
            continue

        # Boost for diversity (likely collision scene) and volume
        if len(found_classes) >= 2:
            score += 20
        score += len(dets) * 2
        score += int(best_confidence * 10)

        results.append({
            "timestamp":  actual_timestamp,
            "classes":    list(found_classes),
            "score":      score,
            "count":      len(dets),
            "confidence": best_confidence,
        })

    results.sort(key=lambda x: (-x["score"], x["timestamp"]))
    return results


# ---------------------------------------------------------------------------
# Appearance colour search
# ---------------------------------------------------------------------------

# Supported colour names (canonical, lowercase)
SUPPORTED_COLORS = {
    "red", "orange", "yellow", "green",
    "blue", "purple", "pink", "white",
    "black", "other",
}

# Colour synonyms for natural-language query parsing
_COLOR_SYNONYMS: dict[str, str] = {
    "red": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "blue": "blue",
    "purple": "purple",
    "violet": "purple",
    "pink": "pink",
    "white": "white",
    "black": "black",
    "dark": "black",
    "navy": "blue",
    "teal": "green",
    "cyan": "blue",
    "maroon": "red",
    "crimson": "red",
    "magenta": "pink",
    "grey": "other",
    "gray": "other",
}


def _parse_color_query(query: str) -> list[str]:
    """Extract canonical colour names from a free-text query string."""
    q = query.lower()
    found: list[str] = []
    for word, canonical in _COLOR_SYNONYMS.items():
        if word in q and canonical not in found:
            found.append(canonical)
    return found


def search_appearances(
    query: str,
    video_id: str | None = None,
    color_confidence_min: float = 0.3,
    limit: int = 50,
) -> list[dict]:
    """
    Search AppearanceObservation records for people whose upper clothing
    matches the colour(s) described in *query*.

    Parameters
    ----------
    query               : Free-text query, e.g. "orange T-shirt" or "blue shirt"
    video_id            : Filter to a specific video; None = search all videos
    color_confidence_min: Minimum upper_color_confidence to include
    limit               : Maximum number of results returned

    Returns
    -------
    List of dicts ordered by upper_color_confidence DESC, timestamp ASC.
    Each dict:
        video_id, filename, track_id, timestamp, frame_number,
        upper_color, upper_color_confidence, crop_path
    """
    colors = _parse_color_query(query)

    if not colors:
        logger.warning("search_appearances: no colour recognised in query %r", query)
        return []

    db = SessionLocal()
    try:
        q = db.query(AppearanceObservation, Video).join(
            Video, Video.id == AppearanceObservation.video_id, isouter=True
        )

        q = q.filter(AppearanceObservation.upper_color.in_(colors))
        q = q.filter(
            AppearanceObservation.upper_color_confidence >= color_confidence_min
        )

        if video_id is not None:
            q = q.filter(AppearanceObservation.video_id == video_id)

        q = q.order_by(
            AppearanceObservation.upper_color_confidence.desc(),
            AppearanceObservation.timestamp.asc(),
        )

        rows = q.limit(limit).all()

        results: list[dict] = []
        seen_tracks: set[tuple] = set()

        for obs, video in rows:
            # Deduplicate by (video_id, track_id) — keep highest-confidence
            dedup_key = (obs.video_id, obs.track_id)
            if obs.track_id is not None and dedup_key in seen_tracks:
                continue
            seen_tracks.add(dedup_key)

            results.append({
                "video_id": obs.video_id,
                "filename": video.filename if video else None,
                "track_id": obs.track_id,
                "timestamp": obs.timestamp,
                "frame_number": obs.frame_number,
                "upper_color": obs.upper_color,
                "upper_color_confidence": obs.upper_color_confidence,
                "crop_path": obs.crop_path,
            })

        return results

    finally:
        db.close()

