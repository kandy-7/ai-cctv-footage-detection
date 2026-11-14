
import csv
import cv2
import time
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "backend" / "uploads"

OUTPUT_DIR = (
    ROOT / "datasets" / "appearance" /
    "upper_color" / "incoming"
)

# Prefer your existing model weights.
WEIGHT_OPTIONS = [
    ROOT / "backend" / "yolov8s.pt",
    ROOT / "backend" / "yolov8n.pt",
    ROOT / "yolov8s.pt",
    ROOT / "yolov8n.pt",
]

FRAME_SKIP = 5
MIN_PERSON_HEIGHT = 100
MIN_CONFIDENCE = 0.40
MIN_SECONDS_BETWEEN_CROPS = 1.5

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv"
}

COLORS = [
    "orange", "pink", "red", "blue",
    "black", "white", "other"
]

# --------------------------------------------------
# FIND MODEL
# --------------------------------------------------

weights = next(
    (p for p in WEIGHT_OPTIONS if p.exists()),
    None
)

if weights is None:
    print("Existing weights not found.")
    print("Using YOLOv8n pretrained weights.")
    weights = "yolov8n.pt"

print(f"Loading model: {weights}")
model = YOLO(str(weights))

# --------------------------------------------------
# ESTIMATE DOMINANT CLOTHING COLOR
# --------------------------------------------------

def estimate_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Ignore very dark and nearly grey pixels
    valid = (s > 35) & (v > 35)

    if valid.sum() < 50:
        return "other"

    hues = h[valid]
    sats = s[valid]
    values = v[valid]

    # Black and white checks
    if (values < 55).mean() > 0.45:
        return "black"

    if ((values > 190) & (sats < 65)).mean() > 0.45:
        return "white"

    # Count pixels in approximate HSV color ranges
    counts = {
        "red": (
            (hues <= 5) | (hues >= 170)
        ).sum(),

        "orange": (
            (hues > 5) & (hues <= 22)
        ).sum(),

        "pink": (
            (hues > 145) & (hues < 170)
        ).sum(),

        "blue": (
            (hues >= 90) & (hues <= 135)
        ).sum(),
    }

    color = max(counts, key=counts.get)

    if counts[color] / len(hues) < 0.20:
        return "other"

    return color


# --------------------------------------------------
# EXTRACT UPPER BODY
# --------------------------------------------------

def extract_upper_body(frame, x1, y1, x2, y2):
    height = y2 - y1
    width = x2 - x1

    # Exclude some head and lower-body area
    top = int(y1 + height * 0.18)
    bottom = int(y1 + height * 0.58)

    left = int(x1 + width * 0.10)
    right = int(x2 - width * 0.10)

    h, w = frame.shape[:2]

    left = max(0, min(left, w))
    right = max(0, min(right, w))
    top = max(0, min(top, h))
    bottom = max(0, min(bottom, h))

    if right <= left or bottom <= top:
        return None

    crop = frame[top:bottom, left:right]

    if crop.size == 0:
        return None

    return crop


# --------------------------------------------------
# MAIN VIDEO PROCESSING
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for color in COLORS:
        (OUTPUT_DIR / color).mkdir(
            parents=True, exist_ok=True
        )

    videos = [
        p for p in VIDEO_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print(f"No videos found in: {VIDEO_DIR}")
        return

    print(f"Found {len(videos)} videos.")

    manifest_path = OUTPUT_DIR / "manifest.csv"

    fields = [
        "video",
        "video_id",
        "frame",
        "timestamp_seconds",
        "track_id",
        "predicted_color",
        "crop_path"
    ]

    total = 0

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fields
        )
        writer.writeheader()

        for video_number, video_path in enumerate(videos):
            print(f"\nProcessing: {video_path.name}")

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print("Could not open video. Skipping.")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0

            video_id = f"video_{video_number:03d}"
            last_saved = {}

            frame_number = 0

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                if frame_number % FRAME_SKIP != 0:
                    continue

                timestamp = frame_number / fps

                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    tracker="bytetrack.yaml",
                    device="cpu",
                    conf=MIN_CONFIDENCE,
                    verbose=False
                )

                if not results or results[0].boxes is None:
                    continue

                boxes = results[0].boxes

                for i, box in enumerate(boxes):
                    confidence = float(box.conf[0])

                    if confidence < MIN_CONFIDENCE:
                        continue

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    if y2 - y1 < MIN_PERSON_HEIGHT:
                        continue

                    if boxes.id is not None:
                        track_id = int(boxes.id[i])
                    else:
                        track_id = frame_number * 1000 + i

                    # Avoid saving nearly identical crops
                    if (
                        track_id in last_saved
                        and timestamp - last_saved[track_id]
                        < MIN_SECONDS_BETWEEN_CROPS
                    ):
                        continue

                    crop = extract_upper_body(
                        frame, x1, y1, x2, y2
                    )

                    if crop is None:
                        continue

                    color = estimate_color(crop)

                    filename = (
                        f"{video_id}_track{track_id}_"
                        f"frame{frame_number}.jpg"
                    )

                    output_path = OUTPUT_DIR / color / filename

                    saved = cv2.imwrite(
                        str(output_path),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )

                    if not saved:
                        continue

                    last_saved[track_id] = timestamp
                    total += 1

                    writer.writerow({
                        "video": video_path.name,
                        "video_id": video_id,
                        "frame": frame_number,
                        "timestamp_seconds": round(timestamp, 2),
                        "track_id": track_id,
                        "predicted_color": color,
                        "crop_path": str(output_path)
                    })

                if frame_number % 500 == 0:
                    print(
                        f"Processed {frame_number} frames | "
                        f"Crops saved: {total}"
                    )

            cap.release()
            print(f"Finished: {video_path.name}")

    print("\n================================")
    print("DATASET EXTRACTION COMPLETED")
    print(f"Total crops: {total}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print("================================")


if __name__ == "__main__":
    main()
