"""
train.py – Standalone YOLOv8 training pipeline.

NOT imported by main.py.  Run as a script:
    python train.py --dataset_yaml data.yaml --base_model yolov8s.pt --epochs 50

Workflow:
  1. (Optionally) validate dataset structure
  2. Train with model.train()
  3. Evaluate with model.val() – prints mAP50, mAP50-95, precision, recall
  4. Print path to best saved weights
"""

import argparse
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def train_custom_model(
    dataset_yaml: str,
    base_model: str = "yolov8s.pt",
    epochs: int = 50,
    imgsz: int = 640,
    device: str = "cpu",
    project: str = "runs/train",
    name: str = "cctv_model",
) -> str:
    """
    Fine-tune a YOLOv8 model on a custom dataset.

    Args:
        dataset_yaml: Path to the YOLO-format dataset YAML (with train/val splits).
        base_model:   Starting checkpoint – 'yolov8s.pt', 'yolov8m.pt', etc.
        epochs:       Number of training epochs.
        imgsz:        Input image size (square).
        device:       Torch device string – 'cpu', '0' (first GPU), etc.
        project:      Root directory for training outputs.
        name:         Sub-directory name for this run.

    Returns:
        Path to best.pt weights file.
    """
    from ultralytics import YOLO  # Import here so the module loads fast when unused

    dataset_yaml = str(Path(dataset_yaml).resolve())
    if not os.path.isfile(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    logger.info(f"Starting training – base={base_model}, epochs={epochs}, "
                f"imgsz={imgsz}, device={device}")
    logger.info(f"Dataset: {dataset_yaml}")

    model = YOLO(base_model)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Evaluate on validation set
    # ------------------------------------------------------------------
    logger.info("Evaluating on validation set …")
    val_results = model.val()

    try:
        metrics = val_results.results_dict
        map50     = metrics.get("metrics/mAP50(B)",    metrics.get("map50",    "N/A"))
        map50_95  = metrics.get("metrics/mAP50-95(B)", metrics.get("map50_95", "N/A"))
        precision = metrics.get("metrics/precision(B)", metrics.get("precision", "N/A"))
        recall    = metrics.get("metrics/recall(B)",    metrics.get("recall",    "N/A"))
    except Exception:
        # Fallback for different ultralytics versions
        try:
            map50     = val_results.box.map50
            map50_95  = val_results.box.map50_95
            precision = val_results.box.p.mean() if hasattr(val_results.box, "p") else "N/A"
            recall    = val_results.box.r.mean() if hasattr(val_results.box, "r") else "N/A"
        except Exception as exc:
            logger.warning(f"Could not extract detailed metrics: {exc}")
            map50 = map50_95 = precision = recall = "N/A"

    logger.info("=== Validation Results ===")
    logger.info(f"  mAP@0.5        : {map50}")
    logger.info(f"  mAP@0.5:0.95   : {map50_95}")
    logger.info(f"  Precision      : {precision}")
    logger.info(f"  Recall         : {recall}")

    # ------------------------------------------------------------------
    # Locate best weights
    # ------------------------------------------------------------------
    run_dir = Path(project) / name
    best_pt = run_dir / "weights" / "best.pt"

    if best_pt.exists():
        logger.info(f"Best model saved at: {best_pt.resolve()}")
        return str(best_pt.resolve())
    else:
        logger.warning("best.pt not found in expected location; check runs/ directory.")
        return str(run_dir / "weights")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a custom YOLOv8 model for CCTV detection."
    )
    p.add_argument(
        "dataset_yaml",
        help="Path to dataset YAML file (YOLO format, with train/val keys).",
    )
    p.add_argument(
        "--base_model", default="yolov8s.pt",
        help="YOLOv8 base checkpoint to fine-tune (default: yolov8s.pt).",
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50).",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640).",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Torch device: 'cpu', '0' for first GPU, etc. (default: cpu).",
    )
    p.add_argument(
        "--project", default="runs/train",
        help="Output project directory (default: runs/train).",
    )
    p.add_argument(
        "--name", default="cctv_model",
        help="Run name / sub-directory (default: cctv_model).",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    best = train_custom_model(
        dataset_yaml=args.dataset_yaml,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print(f"\nTraining complete. Best weights: {best}")
