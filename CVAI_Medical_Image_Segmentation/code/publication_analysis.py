from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from torch.utils.data import DataLoader

import train_models as tm


PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "publication_outputs"
SEEDS = [7, 42, 123]


def make_loaders(seed):
    names = sorted(p.name for p in (tm.DATA_DIR / "images").glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(names)
    train, val, test = names[:800], names[800:900], names[900:]
    return (
        DataLoader(tm.KvasirDataset(train, augment=True), batch_size=tm.BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(tm.KvasirDataset(val), batch_size=tm.BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(tm.KvasirDataset(test), batch_size=tm.BATCH_SIZE, shuffle=False, num_workers=0),
        test,
    )


@torch.no_grad()
def per_image_metrics(model, loader, device):
    model.eval()
    rows = []
    for images, masks, names in loader:
        probabilities = torch.sigmoid(model(images.to(device))).cpu()
        predictions = probabilities >= 0.5
        actual = masks >= 0.5
        for index, name in enumerate(names):
            pred, truth = predictions[index], actual[index]
            tp = (pred & truth).sum().item()
            fp = (pred & ~truth).sum().item()
            fn = (~pred & truth).sum().item()
            tn = (~pred & ~truth).sum().item()
            rows.append({"name": name, **tm.metrics_from_counts(tp, fp, fn, tn)})
    return rows


@torch.no_grad()
def uncertainty_analysis(model, loader, device, passes=12):
    model.train()
    rows = []
    for images, masks, names in loader:
        images_device = images.to(device)
        samples = torch.stack([torch.sigmoid(model(images_device)).cpu() for _ in range(passes)])
        mean_probability = samples.mean(dim=0)
        variance = samples.var(dim=0)
        predictions = mean_probability >= 0.5
        actual = masks >= 0.5
        for index, name in enumerate(names):
            pred, truth = predictions[index], actual[index]
            tp = (pred & truth).sum().item()
            fp = (pred & ~truth).sum().item()
            fn = (~pred & truth).sum().item()
            tn = (~pred & ~truth).sum().item()
            metrics = tm.metrics_from_counts(tp, fp, fn, tn)
            rows.append({
                "name": name,
                "dice": metrics["dice"],
                "error": 1.0 - metrics["dice"],
                "mean_variance": variance[index].mean().item(),
                "max_variance": variance[index].max().item(),
            })
    return rows


def bootstrap_ci(values, seed=2026, iterations=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    seed_summaries = []
    models_42 = {}
    for seed in SEEDS:
        tm.set_seed(seed)
        train_loader, val_loader, test_loader, test_names = make_loaders(seed)
        for label, attention, dropout in [("U-Net", False, 0.0), ("Attention U-Net", True, 0.20)]:
            model = tm.UNet(base=8, attention=attention, dropout=dropout).to(device)
            if seed == 42:
                filename = "unet.pt" if label == "U-Net" else "attention_unet.pt"
                model.load_state_dict(torch.load(PROJECT / "outputs" / filename, map_location=device, weights_only=True))
                training_seconds = json.loads((PROJECT / "outputs" / "results.json").read_text())[label]["training_seconds"]
            else:
                _, training_seconds = tm.train_model(f"seed={seed} {label}", model, train_loader, val_loader, device)
            rows = per_image_metrics(model, test_loader, device)
            for row in rows:
                row.update({"seed": seed, "model": label})
            all_rows.extend(rows)
            seed_summaries.append({
                "seed": seed,
                "model": label,
                "mean_dice": float(np.mean([r["dice"] for r in rows])),
                "mean_iou": float(np.mean([r["iou"] for r in rows])),
                "mean_precision": float(np.mean([r["precision"] for r in rows])),
                "mean_recall": float(np.mean([r["recall"] for r in rows])),
                "training_seconds": training_seconds,
            })
            if seed == 42:
                models_42[label] = model

    paired_differences = []
    for seed in SEEDS:
        u = sorted([r for r in all_rows if r["seed"] == seed and r["model"] == "U-Net"], key=lambda x: x["name"])
        a = sorted([r for r in all_rows if r["seed"] == seed and r["model"] == "Attention U-Net"], key=lambda x: x["name"])
        paired_differences.extend([ar["dice"] - ur["dice"] for ur, ar in zip(u, a)])
    statistic, p_value = wilcoxon(paired_differences, zero_method="wilcox", alternative="two-sided")

    attention_loader = make_loaders(42)[2]
    uncertainty_rows = uncertainty_analysis(models_42["Attention U-Net"], attention_loader, device)
    rho, rho_p = spearmanr([r["mean_variance"] for r in uncertainty_rows], [r["error"] for r in uncertainty_rows])
    ordered = sorted(uncertainty_rows, key=lambda row: row["mean_variance"])
    risk_coverage = []
    for coverage in [1.0, 0.8, 0.6, 0.4, 0.2]:
        retained = ordered[:max(1, int(len(ordered) * coverage))]
        risk_coverage.append({
            "coverage": coverage,
            "retained_images": len(retained),
            "mean_dice": float(np.mean([r["dice"] for r in retained])),
            "mean_error": float(np.mean([r["error"] for r in retained])),
        })

    publication = {
        "protocol": {"seeds": SEEDS, "epochs": tm.EPOCHS, "image_size": tm.IMAGE_SIZE, "test_images_per_seed": 100},
        "seed_summaries": seed_summaries,
        "model_ci": {
            label: {
                "dice_mean_ci95": bootstrap_ci([r["dice"] for r in all_rows if r["model"] == label]),
                "iou_mean_ci95": bootstrap_ci([r["iou"] for r in all_rows if r["model"] == label]),
            }
            for label in ["U-Net", "Attention U-Net"]
        },
        "paired_wilcoxon_attention_minus_unet": {
            "median_difference": float(np.median(paired_differences)),
            "statistic": float(statistic),
            "p_value": float(p_value),
            "paired_observations": len(paired_differences),
        },
        "uncertainty_error_spearman_seed42": {"rho": float(rho), "p_value": float(rho_p), "n": len(uncertainty_rows)},
        "risk_coverage_seed42": risk_coverage,
        "uncertainty_rows_seed42": uncertainty_rows,
    }
    (OUT / "publication_results.json").write_text(json.dumps(publication, indent=2))
    (OUT / "per_image_metrics.json").write_text(json.dumps(all_rows, indent=2))
    print(json.dumps({k: v for k, v in publication.items() if not k.endswith("rows_seed42")}, indent=2))


if __name__ == "__main__":
    main()
