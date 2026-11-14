"""
train_appearance.py
───────────────────
Fine-tunes MobileNetV3-Small on a verified appearance dataset to classify
upper-body clothing colours.

Usage:
    python train_appearance.py [--dataset backend/dataset]
                               [--output  backend/models/appearance_classifier]
                               [--epochs  30]
                               [--batch   8]
                               [--imgsz   160]
                               [--patience 5]
                               [--lr      1e-3]

Requirements:
    pip install torch torchvision scikit-learn Pillow

IMPORTANT:
  - Runs entirely on CPU (no CUDA required).
  - Will NOT start if backend/dataset/train/ is missing or has fewer than
    2 classes with sufficient data.
  - Does NOT overwrite YOLO weights.
  - Reports accuracy, precision, recall, F1, per-class breakdown,
    and confusion matrix on the validation split.
  - Early stopping stops training when val loss stops improving.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
logger = logging.getLogger("train_appearance")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train MobileNetV3-Small appearance classifier"
    )
    p.add_argument(
        "--dataset",
        default=os.path.join(BACKEND_DIR, "dataset"),
        help="Path to dataset directory containing train/ and val/ splits",
    )
    p.add_argument(
        "--output",
        default=os.path.join(
            BACKEND_DIR, "models", "appearance_classifier"
        ),
        help="Directory to save best.pth and class_names.json",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Maximum training epochs",
    )
    p.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (keep small for CPU training)",
    )
    p.add_argument(
        "--imgsz", type=int, default=160,
        help="Input image size (square)",
    )
    p.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (epochs without val loss improvement)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate",
    )
    p.add_argument(
        "--seed", type=int, default=42,
    )
    return p.parse_args()


def check_dataset(dataset_dir: str):
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    if not os.path.isdir(train_dir):
        logger.error(
            "Training split not found: %s\n"
            "Run prepare_appearance_dataset.py first.",
            train_dir,
        )
        sys.exit(1)

    if not os.path.isdir(val_dir):
        logger.error(
            "Validation split not found: %s\n"
            "Run prepare_appearance_dataset.py first.",
            val_dir,
        )
        sys.exit(1)

    classes = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )

    if len(classes) < 2:
        logger.error(
            "Need at least 2 classes in %s. Found: %s",
            train_dir,
            classes,
        )
        sys.exit(1)

    logger.info("Classes found: %s", classes)

    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        logger.info("  train/%s: %d images", cls, len(images))

    return classes


def build_model(num_classes: int):
    """Load MobileNetV3-Small with pretrained ImageNet weights, replace head."""
    try:
        import torchvision
        from torchvision.models import (
            mobilenet_v3_small,
            MobileNet_V3_Small_Weights,
        )
    except ImportError:
        logger.error(
            "torchvision is not installed. "
            "Run: pip install torchvision"
        )
        sys.exit(1)

    logger.info("Loading MobileNetV3-Small with pretrained ImageNet weights...")
    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    # Replace classifier head for our number of classes
    in_features = model.classifier[3].in_features
    import torch.nn as nn
    model.classifier[3] = nn.Linear(in_features, num_classes)

    logger.info(
        "Model: MobileNetV3-Small | Head: %d → %d classes",
        in_features,
        num_classes,
    )
    return model


def get_transforms(imgsz: int):
    from torchvision import transforms

    train_tf = transforms.Compose([
        transforms.Resize((imgsz + 20, imgsz + 20)),
        transforms.RandomCrop((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tf, val_tf


def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    torch.manual_seed(args.seed)

    # Verify dataset
    classes = check_dataset(args.dataset)
    num_classes = len(classes)

    train_tf, val_tf = get_transforms(args.imgsz)

    train_ds = ImageFolder(
        os.path.join(args.dataset, "train"),
        transform=train_tf,
    )
    val_ds = ImageFolder(
        os.path.join(args.dataset, "val"),
        transform=val_tf,
    )

    logger.info(
        "Dataset sizes: train=%d  val=%d",
        len(train_ds),
        len(val_ds),
    )

    if len(val_ds) == 0:
        logger.error("Validation set is empty. Check val/ directory.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,          # Windows compatibility
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device("cpu")
    logger.info("Training on: CPU")

    model = build_model(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    os.makedirs(args.output, exist_ok=True)
    best_model_path = os.path.join(args.output, "best.pth")
    class_names_path = os.path.join(args.output, "class_names.json")

    # Save class names (order must match ImageFolder's internal order)
    class_names = train_ds.classes
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info("Class names saved to %s: %s", class_names_path, class_names)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Training ──
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()

        train_loss /= len(train_ds)
        train_acc = train_correct / len(train_ds)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | "
            "train_loss=%.4f acc=%.3f | "
            "val_loss=%.4f acc=%.3f | "
            "%.1fs",
            epoch, args.epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            elapsed,
        )

        # Early stopping / best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "imgsz": args.imgsz,
                },
                best_model_path,
            )
            logger.info("  ✓ Best model saved (val_loss=%.4f)", val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(
                    "Early stopping at epoch %d "
                    "(no improvement for %d epochs).",
                    epoch, args.patience,
                )
                break

    # ── Final evaluation ──
    logger.info("\n── Final Evaluation on Validation Set ──")

    try:
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            accuracy_score,
        )

        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            zero_division=0,
        )
        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)

        logger.info("Accuracy: %.4f", acc)
        logger.info("\n%s", report)
        logger.info("Confusion matrix:\n%s", cm)

    except ImportError:
        logger.warning(
            "scikit-learn not installed; skipping detailed metrics. "
            "pip install scikit-learn"
        )

    logger.info(
        "\nTraining complete.\n"
        "Best model: %s\n"
        "Class names: %s",
        best_model_path,
        class_names_path,
    )
    logger.info(
        "\nNOTE: Accuracy figures above are on the held-out validation set.\n"
        "Do not treat these numbers as production accuracy until the model\n"
        "has been tested on fresh footage it has never seen."
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
