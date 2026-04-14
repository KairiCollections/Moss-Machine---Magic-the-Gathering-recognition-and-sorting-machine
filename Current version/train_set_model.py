#!/usr/bin/env python3
"""
Train a CNN to classify MTG set from symbol ROI crops.
Dataset: dataset/set/{SET_CODE}
Outputs: models/set_cnn.pt
"""

from pathlib import Path
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset" / "set"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "set_cnn.pt"

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
IMAGE_SIZE = 64


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
    stem = mp.stem  # e.g. "set_cnn"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN to classify MTG set from symbol ROI crops.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate (default: 1e-3)")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Input image size (default: 64)")
    parser.add_argument("--max-classes", type=int, default=0, help="Limit number of classes (0 = all)")
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit samples per class (0 = all)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, class_map):
        self.base = base
        self.indices = indices
        self.class_map = class_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.base[self.indices[idx]]
        return image, self.class_map[label]


def main():
    args = parse_args()
    if not DATA_DIR.exists():
        print(f"Dataset not found: {DATA_DIR}")
        return 1

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    if len(dataset.classes) < 2:
        print("Need at least 2 sets to train.")
        return 1

    torch.manual_seed(args.seed)

    # Optional class and sample limits for faster training
    selected_class_indices = list(range(len(dataset.classes)))
    if args.max_classes and args.max_classes > 0:
        selected_class_indices = selected_class_indices[: args.max_classes]

    selected_class_names = [dataset.classes[i] for i in selected_class_indices]
    selected_class_to_idx = {name: i for i, name in enumerate(selected_class_names)}
    selected_class_map = {orig_idx: selected_class_to_idx[dataset.classes[orig_idx]] for orig_idx in selected_class_indices}

    max_per_class = args.max_per_class if args.max_per_class and args.max_per_class > 0 else None
    counts = defaultdict(int)
    indices = []
    for idx, target in enumerate(dataset.targets):
        if target not in selected_class_map:
            continue
        if max_per_class is not None and counts[target] >= max_per_class:
            continue
        counts[target] += 1
        indices.append(idx)

    if len(selected_class_names) < 2:
        print("Need at least 2 sets to train after filtering.")
        return 1

    dataset = RemappedSubset(dataset, indices, selected_class_map)

    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes: {len(selected_class_names)}")

    # Small ResNet18
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(selected_class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
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

        print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": selected_class_to_idx,
                "image_size": args.image_size
            }, MODEL_PATH)
            print(f"  ✓ Saved model to {MODEL_PATH}")
            # Split into ≤24 MB chunks so every part fits under the 24 MB limit
            _split_model(MODEL_PATH)

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
