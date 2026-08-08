import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfms, eval_tfms

def build_datasets():
    train_tfms, eval_tfms = get_transforms()

    train_ds = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=train_tfms,
        download=True,
    )

    # Use the official test split for final evaluation.
    test_ds = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        target_types="category",
        transform=eval_tfms,
        download=True,
    )

    return train_ds, test_ds

def build_model(num_classes):
    # ImageNet-pretrained ResNet18 transfer learning.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("Downloading/loading Oxford-IIIT Pet Dataset if necessary...")

    train_ds, test_ds = build_datasets()

    print(f"Training images: {len(train_ds)}")
    print(f"Test images: {len(test_ds)}")
    print(f"Classes: {len(train_ds.classes)}")

    # Save the exact label order used during training.
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(train_ds.classes, f, indent=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # safer for Windows
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(len(train_ds.classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, labels in progress:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            progress.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{100 * correct / total:.2f}%"
            )

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, test_loader, criterion)
        scheduler.step(val_acc)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": val_loss,
            "test_accuracy": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_acc={train_acc*100:.2f}% | "
            f"test_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": train_ds.classes,
                    "best_accuracy": best_acc,
                },
                MODEL_DIR / "best_model.pth",
            )
            print(f"Saved new best model: {best_acc*100:.2f}%")

    with open(MODEL_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print(f"Best test accuracy: {best_acc*100:.2f}%")
    print(f"Checkpoint: {MODEL_DIR / 'best_model.pth'}")

if __name__ == "__main__":
    main()
