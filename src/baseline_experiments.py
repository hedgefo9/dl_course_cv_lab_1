from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ViT_B_16_Weights,
    resnet18,
    vit_b_16,
)
from tqdm.auto import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline training for garbage classification."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/standardized_256"),
        help="Path to folder with class subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/baseline"),
        help="Path for all numeric artifacts.",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=Path("artifacts/plots"),
        help="Path for generated plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs-resnet", type=int, default=6)
    parser.add_argument("--epochs-vit", type=int, default=4)
    parser.add_argument("--batch-size-resnet", type=int, default=64)
    parser.add_argument("--batch-size-vit", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--trainable-backbone",
        action="store_true",
        help="If set, fine-tune all layers. Otherwise, only classifier head is trainable.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Record:
    path: Path
    label: int


class RecordsDataset(Dataset):
    def __init__(self, records: list[Record], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")
        return self.transform(image), record.label


def collect_metadata(data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    class_names = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class directories found in {data_root}")

    rows: list[dict] = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label_idx, class_name in enumerate(class_names):
        for file_path in sorted((data_root / class_name).iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in valid_ext:
                rows.append(
                    {
                        "path": str(file_path.resolve()),
                        "label": label_idx,
                        "class_name": class_name,
                    }
                )

    if not rows:
        raise ValueError(f"No supported image files found in {data_root}")

    df = pd.DataFrame(rows)
    return df, class_names


def make_splits(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
) -> pd.DataFrame:
    if not (0.0 < test_size < 1.0 and 0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must be in (0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    val_ratio_in_train_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_in_train_val,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    return split_df


def make_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def freeze_backbone(model: nn.Module, model_name: str, trainable_backbone: bool) -> None:
    if trainable_backbone:
        return

    for param in model.parameters():
        param.requires_grad = False

    if model_name == "resnet18":
        for param in model.fc.parameters():
            param.requires_grad = True
        return

    if model_name == "vit_b_16":
        for param in model.heads.parameters():
            param.requires_grad = True
        return

    raise ValueError(f"Unsupported model for freezing: {model_name}")


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        try:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as exc:
            print(f"[warn] Failed to load ResNet18 pretrained weights, using random init: {exc}")
            model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "vit_b_16":
        try:
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        except Exception as exc:
            print(f"[warn] Failed to load ViT-B/16 pretrained weights, using random init: {exc}")
            model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model name: {model_name}")


def make_optimizer(model: nn.Module, model_name: str) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if model_name == "resnet18":
        return torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    if model_name == "vit_b_16":
        return torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    raise ValueError(f"Unsupported model name: {model_name}")


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float], list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = epoch_loss
    return metrics, all_targets, all_preds


def plot_class_distribution(split_df: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
    counts = (
        split_df.groupby(["split", "class_name"], as_index=False)
        .size()
        .pivot(index="class_name", columns="split", values="size")
        .fillna(0)
        .reindex(class_names)
    )
    counts = counts[[col for col in ["train", "val", "test"] if col in counts.columns]]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25

    for i, split_name in enumerate(counts.columns):
        plt.bar(x + (i - 1) * width, counts[split_name].values, width=width, label=split_name)

    plt.xticks(x, class_names, rotation=35, ha="right")
    plt.ylabel("Number of images")
    plt.title("Class distribution by split")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_learning_curves(history_df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_macro_f1"], label="train_macro_f1")
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(f"{model_name}: Macro F1")
    plt.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def df_to_records(df: pd.DataFrame) -> list[Record]:
    return [Record(path=Path(row.path), label=int(row.label)) for row in df.itertuples()]


def make_loaders(
    split_df: pd.DataFrame,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_ds = RecordsDataset(df_to_records(train_df), make_transforms(img_size, train=True))
    val_ds = RecordsDataset(df_to_records(val_df), make_transforms(img_size, train=False))
    test_ds = RecordsDataset(df_to_records(test_df), make_transforms(img_size, train=False))

    common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def train_model(
    model_name: str,
    split_df: pd.DataFrame,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    epochs = args.epochs_resnet if model_name == "resnet18" else args.epochs_vit
    batch_size = (
        args.batch_size_resnet if model_name == "resnet18" else args.batch_size_vit
    )

    model_dir = args.output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        split_df=split_df,
        img_size=args.img_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = build_model(model_name, num_classes=len(class_names))
    freeze_backbone(model, model_name, trainable_backbone=args.trainable_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, model_name)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history_rows: list[dict] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_path = model_dir / "best.pt"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics, _, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics, _, _ = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)

        print(
            f"[{model_name}] epoch {epoch:02d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f}, val_loss={row['val_loss']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}, val_macro_f1={row['val_macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    train_seconds = time.time() - start_time

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(model_dir / "history.csv", index=False)
    plot_learning_curves(
        history_df,
        model_name=model_name,
        out_path=args.plots_root / f"{model_name}_learning_curves.png",
    )

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, y_true, y_pred = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    plot_confusion(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        title=f"{model_name} confusion matrix (normalized)",
        out_path=args.plots_root / f"{model_name}_confusion_matrix.png",
    )

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    pred_df.to_csv(model_dir / "test_predictions.csv", index=False)

    metrics_payload = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "test_metrics": test_metrics,
    }
    with (model_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    return metrics_payload


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = get_device()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.plots_root.mkdir(parents=True, exist_ok=True)

    metadata_df, class_names = collect_metadata(args.data_root)
    split_df = make_splits(
        metadata_df,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    split_df = split_df.sort_values(by=["split", "class_name", "path"]).reset_index(drop=True)
    split_df.to_csv(args.output_root / "splits.csv", index=False)

    dataset_summary = {
        "data_root": str(args.data_root.resolve()),
        "num_images": int(len(metadata_df)),
        "num_classes": int(len(class_names)),
        "classes": class_names,
        "split_counts": split_df["split"].value_counts().to_dict(),
        "per_class_counts": metadata_df["class_name"].value_counts().to_dict(),
        "seed": args.seed,
        "device": str(device),
        "trainable_backbone": bool(args.trainable_backbone),
    }
    with (args.output_root / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)

    plot_class_distribution(
        split_df=split_df,
        class_names=class_names,
        out_path=args.plots_root / "class_distribution_by_split.png",
    )

    model_names = ["resnet18", "vit_b_16"]
    all_results: list[dict] = []
    print(f"Using device: {device}")
    print(f"Trainable backbone: {args.trainable_backbone}")
    for model_name in model_names:
        print(f"\n=== Training {model_name} ===")
        result = train_model(
            model_name=model_name,
            split_df=split_df,
            class_names=class_names,
            args=args,
            device=device,
        )
        all_results.append(result)

    summary_rows = []
    for result in all_results:
        m = result["test_metrics"]
        summary_rows.append(
            {
                "model_name": result["model_name"],
                "best_epoch": result["best_epoch"],
                "train_seconds": result["train_seconds"],
                "test_accuracy": m["accuracy"],
                "test_macro_f1": m["macro_f1"],
                "test_weighted_f1": m["weighted_f1"],
                "test_macro_precision": m["macro_precision"],
                "test_macro_recall": m["macro_recall"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(by="test_macro_f1", ascending=False)
    summary_df.to_csv(args.output_root / "summary_metrics.csv", index=False)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
