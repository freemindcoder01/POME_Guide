"""
Oil Palm FFB Grading System - Training Script
Based on MPOB Oil Palm Fruit Grading Manual (2nd Edition)
Author: AI-generated prototype
Framework: PyTorch + torchvision
"""

import os
import json
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # Paths
    "data_dir": "./dataset",          # Structure: dataset/train/ripe/*.jpg etc.
    "output_dir": "./output",
    "model_save_path": "./output/ffb_model_best.pth",

    # Classes (must match subfolder names in dataset/)
    "class_names": ["ripe", "underripe", "unripe", "rotten", "empty"],

    # Training
    "num_epochs": 40,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "img_size": 224,
    "num_workers": 4,

    # Model
    "backbone": "efficientnet_b0",    # options: efficientnet_b0, mobilenet_v3_large, resnet50
    "pretrained": True,
    "dropout_rate": 0.3,

    # Training strategy
    "freeze_epochs": 5,               # Epochs to train only the head before unfreezing backbone
    "early_stopping_patience": 10,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
NUM_CLASSES = len(CONFIG["class_names"])

print(f"Device: {CONFIG['device']}")
print(f"Classes: {CONFIG['class_names']}")


# ─────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────

def get_transforms(phase: str):
    """
    Training: aggressive augmentation to compensate for small datasets.
    Validation/Test: only resize + normalize.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    size = CONFIG["img_size"]

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────

def load_datasets():
    """
    Expected directory structure:
    dataset/
        train/
            ripe/        *.jpg / *.png
            underripe/
            unripe/
            rotten/
            empty/
        val/
            ripe/
            ...
        test/       (optional)
            ...
    """
    data_dir = Path(CONFIG["data_dir"])
    phases = ["train", "val"]
    if (data_dir / "test").exists():
        phases.append("test")

    datasets_dict = {
        phase: datasets.ImageFolder(
            root=str(data_dir / phase),
            transform=get_transforms(phase)
        )
        for phase in phases
    }

    # Class-balanced sampling for training (handles imbalanced datasets)
    train_ds = datasets_dict["train"]
    class_counts = np.bincount([label for _, label in train_ds.imgs])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, label in train_ds.imgs]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            sampler=sampler,
            num_workers=CONFIG["num_workers"],
            pin_memory=True
        ),
        "val": DataLoader(
            datasets_dict["val"],
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=True
        ),
    }
    if "test" in phases:
        loaders["test"] = DataLoader(
            datasets_dict["test"],
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
        )

    print("\nDataset sizes:")
    for phase, ds in datasets_dict.items():
        counts = np.bincount([label for _, label in ds.imgs])
        for cls, cnt in zip(CONFIG["class_names"], counts):
            print(f"  {phase}/{cls}: {cnt} images")

    return loaders, datasets_dict


# ─────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────

def build_model() -> nn.Module:
    backbone = CONFIG["backbone"]
    pretrained = CONFIG["pretrained"]

    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=CONFIG["dropout_rate"], inplace=True),
            nn.Linear(in_features, NUM_CLASSES)
        )

    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        )
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    elif backbone == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=CONFIG["dropout_rate"]),
            nn.Linear(in_features, NUM_CLASSES)
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the final classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        total += inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc  = running_correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def train(model, loaders, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(CONFIG["num_epochs"]):

        # Phase 1: train head only; Phase 2: unfreeze full model
        if epoch == CONFIG["freeze_epochs"]:
            print("\n>>> Unfreezing backbone for full fine-tuning <<<\n")
            unfreeze_backbone(model)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=CONFIG["learning_rate"] * 0.1,
                weight_decay=CONFIG["weight_decay"]
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=CONFIG["num_epochs"] - epoch,
                eta_min=1e-7
            )

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, loaders["val"], criterion, device
        )
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}] "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, CONFIG["model_save_path"])
            print(f"  ✓ New best model saved (val_acc={best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

    # Save training history
    with open(os.path.join(CONFIG["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final classification report on validation set
    model.load_state_dict(best_model_weights)
    _, _, val_preds, val_labels = evaluate(model, loaders["val"], criterion, device)
    report = classification_report(
        val_labels, val_preds,
        target_names=CONFIG["class_names"],
        digits=4
    )
    print("\n=== Final Validation Classification Report ===")
    print(report)

    with open(os.path.join(CONFIG["output_dir"], "classification_report.txt"), "w") as f:
        f.write(report)

    return model, best_val_acc


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    device = torch.device(CONFIG["device"])

    loaders, _ = load_datasets()
    model = build_model()

    # Start with frozen backbone
    if CONFIG["freeze_epochs"] > 0:
        freeze_backbone(model)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {CONFIG['backbone']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable (initial): {trainable_params:,}\n")

    model, best_acc = train(model, loaders, device)
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {CONFIG['model_save_path']}")

    # Save config alongside model
    with open(os.path.join(CONFIG["output_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)


if __name__ == "__main__":
    main()
