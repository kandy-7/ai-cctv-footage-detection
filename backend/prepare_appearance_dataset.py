"""
prepare_appearance_dataset.py
─────────────────────────────
Prepares a MobileNetV3-ready image dataset from verified appearance labels.

Usage:
    python prepare_appearance_dataset.py [--labels backend/appearance_labels.csv]
                                         [--crops  backend/person_crops]
                                         [--output backend/dataset]
                                         [--val-fraction 0.2]
                                         [--min-samples 10]

Label CSV format (must have a header row):
    crop_path,upper_color,verified
    backend/person_crops/abc/t1_f30.jpg,orange,True
    ...

IMPORTANT:
  - Only rows where verified == True are used.
  - Splits are done PER TRACK (extracted from filename prefix),
    so every image of the same person is entirely in one split.
  - If any class has fewer than --min-samples verified examples the
    script aborts without touching the output directory.
  - Does NOT fabricate labels or use unverified predictions as ground truth.
"""

import os
import sys
import csv
import shutil
import random
import argparse
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
logger = logging.getLogger("prepare_dataset")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare appearance classifier dataset"
    )
    p.add_argument(
        "--labels",
        default=os.path.join(BACKEND_DIR, "appearance_labels.csv"),
        help="Path to the labels CSV file",
    )
    p.add_argument(
        "--crops",
        default=os.path.join(BACKEND_DIR, "person_crops"),
        help="Root directory containing person crops",
    )
    p.add_argument(
        "--output",
        default=os.path.join(BACKEND_DIR, "dataset"),
        help="Output directory for train/val splits",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of tracks reserved for validation (0.0–0.5)",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum verified samples required per class",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def _track_key(crop_path: str) -> str:
    """
    Derive a stable track identifier from the crop filename.

    Expected format:  vid<video_id>_t<track_id>_f<frame>.jpg
    Falls back to the crop's parent directory name if the format differs.
    """
    fname = os.path.basename(crop_path)
    # Try to extract vid + track prefix, e.g. "vidabc123_t5"
    parts = fname.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return os.path.basename(os.path.dirname(crop_path)) or fname


def load_verified_labels(csv_path: str) -> list[dict]:
    """
    Read CSV and return only rows where verified == True.
    Validates that required columns are present.
    """
    if not os.path.isfile(csv_path):
        logger.error(
            "Labels CSV not found: %s\n"
            "Create it with columns: crop_path,upper_color,verified",
            csv_path,
        )
        sys.exit(1)

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"crop_path", "upper_color", "verified"}

        if not required.issubset(set(reader.fieldnames or [])):
            logger.error(
                "CSV is missing required columns. "
                "Expected: crop_path, upper_color, verified. "
                "Got: %s",
                reader.fieldnames,
            )
            sys.exit(1)

        for row in reader:
            if row.get("verified", "").strip().lower() not in (
                "true", "1", "yes"
            ):
                continue

            crop_path = row["crop_path"].strip()
            color = row["upper_color"].strip().lower()

            if not crop_path or not color:
                continue

            rows.append({"crop_path": crop_path, "upper_color": color})

    logger.info("Loaded %d verified label rows from %s", len(rows), csv_path)
    return rows


def validate_images(rows: list[dict]) -> list[dict]:
    """Remove rows whose crop image file does not exist or is unreadable."""
    valid: list[dict] = []
    missing = 0

    for row in rows:
        path = row["crop_path"]
        if not os.path.isfile(path):
            logger.warning("Missing crop: %s", path)
            missing += 1
            continue
        if os.path.getsize(path) == 0:
            logger.warning("Empty crop: %s", path)
            missing += 1
            continue
        valid.append(row)

    if missing:
        logger.warning("Skipped %d missing/empty crop files.", missing)
    logger.info("Valid crops: %d", len(valid))
    return valid


def check_class_balance(rows: list[dict], min_samples: int) -> dict[str, list]:
    """Group rows by class. Abort if any class is below min_samples."""
    by_class: dict[str, list] = defaultdict(list)
    for row in rows:
        by_class[row["upper_color"]].append(row)

    logger.info("Class distribution (verified):")
    abort = False
    for cls, items in sorted(by_class.items()):
        flag = "" if len(items) >= min_samples else "  ← BELOW MINIMUM"
        logger.info("  %-15s %4d samples%s", cls, len(items), flag)
        if len(items) < min_samples:
            abort = True

    if abort:
        logger.error(
            "One or more classes have fewer than %d verified samples. "
            "Collect more labelled crops before preparing the dataset.",
            min_samples,
        )
        sys.exit(1)

    return dict(by_class)


def track_based_split(
    rows: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split rows into train / val ensuring every image of the same track
    goes to the same split.

    Returns (train_rows, val_rows).
    """
    # Group by track key
    track_to_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = _track_key(row["crop_path"])
        track_to_rows[key].append(row)

    all_tracks = sorted(track_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(all_tracks)

    n_val = max(1, int(len(all_tracks) * val_fraction))
    val_tracks = set(all_tracks[:n_val])

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for track, track_rows in track_to_rows.items():
        if track in val_tracks:
            val_rows.extend(track_rows)
        else:
            train_rows.extend(track_rows)

    logger.info(
        "Track-based split: %d tracks → train=%d images, val=%d images",
        len(all_tracks),
        len(train_rows),
        len(val_rows),
    )
    return train_rows, val_rows


def copy_to_split(
    rows: list[dict],
    split_dir: str,
    split_name: str,
) -> None:
    """Copy crop images into split_dir/<class>/<filename> structure."""
    for row in rows:
        cls_dir = os.path.join(split_dir, row["upper_color"])
        os.makedirs(cls_dir, exist_ok=True)
        dst = os.path.join(cls_dir, os.path.basename(row["crop_path"]))
        if not os.path.exists(dst):
            shutil.copy2(row["crop_path"], dst)

    class_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        class_counts[row["upper_color"]] += 1

    logger.info("  %s split class distribution:", split_name)
    for cls, cnt in sorted(class_counts.items()):
        logger.info("    %-15s %4d", cls, cnt)


def main():
    args = parse_args()
    random.seed(args.seed)

    logger.info("Loading verified labels from %s", args.labels)
    rows = load_verified_labels(args.labels)

    if not rows:
        logger.error(
            "No verified rows found in %s. "
            "Set verified=True for samples you have personally confirmed.",
            args.labels,
        )
        sys.exit(1)

    rows = validate_images(rows)

    if not rows:
        logger.error("No valid images remain after validation.")
        sys.exit(1)

    by_class = check_class_balance(rows, args.min_samples)

    train_rows, val_rows = track_based_split(rows, args.val_fraction, args.seed)

    train_dir = os.path.join(args.output, "train")
    val_dir = os.path.join(args.output, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    logger.info("Copying training images to %s", train_dir)
    copy_to_split(train_rows, train_dir, "train")

    logger.info("Copying validation images to %s", val_dir)
    copy_to_split(val_rows, val_dir, "val")

    logger.info(
        "Dataset prepared successfully in %s\n"
        "  Train: %d images | Val: %d images",
        args.output,
        len(train_rows),
        len(val_rows),
    )
    logger.info(
        "Next step: python train_appearance.py --dataset %s",
        args.output,
    )


if __name__ == "__main__":
    main()
