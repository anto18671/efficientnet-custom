from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from efficientnet import EfficientNet
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torchsummary
import torch
import math


# ---------------------------
# Compose torchvision transforms
# ---------------------------
def build_transforms():
    # Training transforms
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
    ])

    # Validation transforms
    valid_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Return all transforms
    return train_tfms, valid_tfms


# ---------------------------
# Create DataLoaders from imagenet1k/
# ---------------------------
def create_dataloaders():
    # Constants
    data_dir = "imagenet1k"
    batch_size = 32
    val_split = 0.1
    seed = 42

    # Build transforms
    train_tfms, valid_tfms = build_transforms()
    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tfms)

    # Split dataset
    class_to_idx = full_ds.class_to_idx
    num_items = len(full_ds)
    val_len = int(math.floor(num_items * val_split))
    train_len = num_items - val_len

    # Create data splits
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=generator)

    # Set validation transforms
    val_ds.dataset.transform = valid_tfms

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, persistent_workers=True)

    # Return all loaders
    return train_loader, val_loader, class_to_idx


# ---------------------------
# Build model, loss, optimizer, scheduler, device
# ---------------------------
def build_training_objects(num_classes):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model configuration
    base_cfg = [
        (1, 32, 2, 2),
        (2, 64, 2, 2),
        (4, 128, 3, 2),
        (6, 192, 4, 2),
        (6, 256, 1, 1),
        (6, 384, 1, 1),
    ]

    # Create model
    model = EfficientNet(base_cfg=base_cfg, num_classes=num_classes, width_mult=1.0, depth_mult=1.0, dropout=0.25)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    # Return all objects
    return model, criterion, optimizer, scheduler, device


# ---------------------------
# Train for one epoch with tqdm
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch):
    # Set model to training mode
    model.train()

    # Initialize metrics
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    # Progress bar
    loop = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    # Iterate over data
    for images, targets in loop:
        # Move data to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute predictions
        preds = outputs.argmax(dim=1)

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == targets).sum().item()
        running_count += images.size(0)

        # Compute averages
        avg_loss = running_loss / max(1, running_count)
        avg_acc = running_correct / max(1, running_count)
        loop.set_postfix(loss=avg_loss, acc=avg_acc)

    # Compute epoch metrics
    epoch_loss = running_loss / max(1, running_count)
    epoch_acc = running_correct / max(1, running_count)

    # Return all metrics
    return epoch_loss, epoch_acc


# ---------------------------
# Evaluate on validation set with tqdm
# ---------------------------
def evaluate(model, loader, criterion, device, epoch):
    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    # Progress bar
    loop = tqdm(loader, desc=f"Valid Epoch {epoch}", leave=False)

    # Disable gradient calculation
    with torch.no_grad():
        for images, targets in loop:
            # Move data to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Compute predictions
            preds = outputs.argmax(dim=1)

            # Accumulate metrics
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == targets).sum().item()
            running_count += images.size(0)

            # Compute averages
            avg_loss = running_loss / max(1, running_count)
            avg_acc = running_correct / max(1, running_count)
            loop.set_postfix(loss=avg_loss, acc=avg_acc)

    # Compute epoch metrics
    epoch_loss = running_loss / max(1, running_count)
    epoch_acc = running_correct / max(1, running_count)

    # Return all metrics
    return epoch_loss, epoch_acc


# ---------------------------
# Training orchestration
# ---------------------------
def main():
    # Create data loaders
    train_loader, val_loader, class_to_idx = create_dataloaders()
    num_classes = len(class_to_idx)

    # Build training objects
    model, criterion, optimizer, scheduler, device = build_training_objects(num_classes)
    scaler = torch.amp.GradScaler('cuda')

    # Print model summary
    torchsummary.summary(model, (3, 224, 224))

    # Do training and validation
    for epoch in range(1, 21):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        scheduler.step()

        # Print metrics
        print(
            f"Epoch {epoch:03d}/20 | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )


# ---------------------------
# Invoke main
# ---------------------------
if __name__ == "__main__":
    main()
