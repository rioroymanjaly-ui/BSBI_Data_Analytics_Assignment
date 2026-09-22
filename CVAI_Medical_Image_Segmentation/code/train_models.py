from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SEED = 42
IMAGE_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 10
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Kvasir-SEG" / "Kvasir-SEG"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class KvasirDataset(Dataset):
    def __init__(self, names: list[str], augment: bool = False):
        self.names = names
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = Image.open(DATA_DIR / "images" / name).convert("RGB")
        mask = Image.open(DATA_DIR / "masks" / name).convert("L")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        if self.augment and random.random() < 0.5:
            image_array = np.ascontiguousarray(image_array[:, ::-1])
            mask_array = np.ascontiguousarray(mask_array[:, ::-1])
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array)[None]
        return image_tensor, mask_tensor, name


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.gate = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.skip = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.score = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_channels, 1, 1), nn.Sigmoid())

    def forward(self, gate, skip):
        attention = self.score(self.gate(gate) + self.skip(skip))
        return skip * attention


class UNet(nn.Module):
    def __init__(self, base: int = 8, attention: bool = False, dropout: float = 0.0):
        super().__init__()
        self.attention = attention
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(base * 4, base * 8, dropout=dropout)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        if attention:
            self.att3 = AttentionGate(base * 4, base * 4, base * 2)
            self.att2 = AttentionGate(base * 2, base * 2, base)
            self.att1 = AttentionGate(base, base, max(base // 2, 1))
        self.dec3 = ConvBlock(base * 8, base * 4, dropout=dropout)
        self.dec2 = ConvBlock(base * 4, base * 2, dropout=dropout)
        self.dec1 = ConvBlock(base * 2, base, dropout=dropout)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bridge = self.bridge(self.pool(e3))
        d3 = self.up3(bridge)
        s3 = self.att3(d3, e3) if self.attention else e3
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3)
        s2 = self.att2(d2, e2) if self.attention else e2
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2)
        s1 = self.att1(d1, e1) if self.attention else e1
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return self.out(d1)


def dice_loss(logits, targets, smooth: float = 1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    total = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (total + smooth)).mean()


def combined_loss(logits, targets):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


def confusion_counts(logits, targets):
    predictions = torch.sigmoid(logits) >= 0.5
    actual = targets >= 0.5
    tp = (predictions & actual).sum().item()
    fp = (predictions & ~actual).sum().item()
    fn = (~predictions & actual).sum().item()
    tn = (~predictions & ~actual).sum().item()
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-8
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_total = 0.0
    counts = np.zeros(4, dtype=np.float64)
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss_total += combined_loss(logits, masks).item() * images.size(0)
        counts += np.array(confusion_counts(logits, masks))
    values = metrics_from_counts(*counts)
    values["loss"] = loss_total / len(loader.dataset)
    return values


def train_model(label, model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    best_dice = -1.0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset), **validation}
        history.append(row)
        print(label, row, flush=True)
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return history, elapsed


@torch.no_grad()
def benchmark_inference(model, loader, device):
    model.eval()
    images, _, _ = next(iter(loader))
    images = images.to(device)
    for _ in range(3):
        model(images)
    timings = []
    for _ in range(10):
        start = time.perf_counter()
        model(images)
        timings.append((time.perf_counter() - start) * 1000 / images.size(0))
    return float(np.mean(timings))


@torch.no_grad()
def save_predictions(models, loader, device):
    batch = next(iter(loader))
    images, masks, names = batch
    images_device = images.to(device)
    predictions = {}
    for label, model in models.items():
        model.eval()
        predictions[label] = torch.sigmoid(model(images_device)).cpu()
    n = 4
    fig, axes = plt.subplots(n, 4, figsize=(10, 10))
    for row in range(n):
        axes[row, 0].imshow(images[row].permute(1, 2, 0))
        axes[row, 1].imshow(masks[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(predictions["U-Net"][row, 0] >= 0.5, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(predictions["Attention U-Net"][row, 0] >= 0.5, cmap="gray", vmin=0, vmax=1)
        for col in range(4):
            axes[row, col].axis("off")
    for col, title in enumerate(["Image", "Ground truth", "U-Net", "Attention U-Net"]):
        axes[0, col].set_title(title)
    fig.suptitle("Test-set qualitative comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_uncertainty(model, loader, device, passes: int = 12):
    images, masks, names = next(iter(loader))
    image = images[:1].to(device)
    model.train()
    samples = [torch.sigmoid(model(image)).cpu().numpy()[0, 0] for _ in range(passes)]
    samples = np.stack(samples)
    mean_prediction = samples.mean(axis=0)
    variance = samples.var(axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    panels = [
        (images[0].permute(1, 2, 0), "Image", None),
        (masks[0, 0], "Ground truth", "gray"),
        (mean_prediction >= 0.5, "Mean prediction", "gray"),
        (variance, "MC-dropout uncertainty", "magma"),
    ]
    for ax, (panel, title, cmap) in zip(axes, panels):
        ax.imshow(panel, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "uncertainty_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"sample": names[0], "passes": passes, "mean_variance": float(variance.mean()), "max_variance": float(variance.max())}


def plot_history(histories):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, history in histories.items():
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o", label=label)
        axes[1].plot(epochs, [r["dice"] for r in history], marker="o", label=label)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="BCE + Dice loss")
    axes[1].set(title="Validation Dice", xlabel="Epoch", ylabel="Dice coefficient")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in (DATA_DIR / "images").glob("*.jpg"))
    assert len(names) == 1000, f"Expected 1000 images, found {len(names)}"
    assert all((DATA_DIR / "masks" / name).exists() for name in names)
    rng = random.Random(SEED)
    rng.shuffle(names)
    train_names, val_names, test_names = names[:800], names[800:900], names[900:]
    (OUTPUT_DIR / "split.json").write_text(json.dumps({"seed": SEED, "train": train_names, "validation": val_names, "test": test_names}, indent=2))
    train_loader = DataLoader(KvasirDataset(train_names, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(KvasirDataset(val_names), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(KvasirDataset(test_names), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configurations = {
        "U-Net": UNet(base=8, attention=False, dropout=0.0),
        "Attention U-Net": UNet(base=8, attention=True, dropout=0.20),
    }
    histories, summary = {}, {}
    for label, model in configurations.items():
        model = model.to(device)
        history, training_seconds = train_model(label, model, train_loader, val_loader, device)
        histories[label] = history
        test_metrics = evaluate(model, test_loader, device)
        test_metrics.update({
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_seconds": training_seconds,
            "inference_ms_per_image": benchmark_inference(model, test_loader, device),
        })
        summary[label] = test_metrics
        torch.save(model.state_dict(), OUTPUT_DIR / f"{label.lower().replace(' ', '_').replace('-', '')}.pt")
        configurations[label] = model
    plot_history(histories)
    save_predictions(configurations, test_loader, device)
    summary["uncertainty"] = save_uncertainty(configurations["Attention U-Net"], test_loader, device)
    summary["experiment"] = {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "split": {"train": 800, "validation": 100, "test": 100},
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "history.json").write_text(json.dumps(histories, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
