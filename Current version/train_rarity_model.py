#!/usr/bin/env python3
"""
Train a simple CNN to classify MTG rarity from symbol ROI crops.
Dataset: dataset/rarity/{Common,Uncommon,Rare,Mythic}
Outputs: models/rarity_cnn.pt
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset" / "rarity"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "rarity_cnn.pt"

BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-3
IMAGE_SIZE = 64

CLASS_NAMES = ["Common", "Uncommon", "Rare", "Mythic"]

CHUNK_SIZE_MB = 24


def _split_model(model_path, chunk_mb=CHUNK_SIZE_MB):
    """Split a saved model file into <=chunk_mb binary chunks named <stem>.part00.pt, etc.
    The original monolithic file is removed after a successful split.
    If the file is already at or below the chunk size, no split is performed."""
    from pathlib import Path as _Path
    mp = _Path(model_path)
    if not mp.exists():
        return
    size_mb = mp.stat().st_size / (1024 * 1024)
    if size_mb <= chunk_mb:
        print(f"  ℹ {mp.name} is {size_mb:.2f} MB — no split needed")
        return
    chunk_bytes = int(chunk_mb * 1024 * 1024)
    stem = mp.stem  # e.g. "rarity_cnn"
    parent = mp.parent
    with open(mp, 'rb') as f:
        idx = 0
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            part_path = parent / f"{stem}.part{idx:02d}.pt"
            with open(part_path, 'wb') as out:
                out.write(data)
            print(f"  ✓ Wrote {part_path.name}: {len(data)/(1024*1024):.2f} MB")
            idx += 1
    mp.unlink()
    print(f"  ✗ Removed monolithic {mp.name} ({size_mb:.2f} MB) — {idx} chunks written")


class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    if not DATA_DIR.exists():
        print(f"Dataset not found: {DATA_DIR}")
        return 1

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # Ensure class order
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    print(f"Class map: {idx_to_class}")

    # Split train/val
    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SmallCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total * 100
        train_loss = running_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total * 100

        print(f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": dataset.class_to_idx,
                "image_size": IMAGE_SIZE
            }, MODEL_PATH)
            print(f"  ✓ Saved model to {MODEL_PATH}")
            # Split into ≤24 MB chunks so every part fits under the 24 MB limit
            _split_model(MODEL_PATH)

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
