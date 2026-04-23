from __future__ import annotations

import argparse
import json
import math
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
from tqdm.auto import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiments with custom-implemented models for garbage classification."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/standardized_256"),
        help="Path to folder with class subfolders.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("artifacts/baseline/splits.csv"),
        help="Optional split file for strict comparability with previous points.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/custom_models"),
        help="Path for numeric artifacts.",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=Path("artifacts/plots/custom_models"),
        help="Path for generated plots.",
    )
    parser.add_argument("--run-name", type=str, default="baseline_short")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["baseline", "improved"],
        default="baseline",
        help="baseline: simple training; improved: techniques from point 3c.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["custom_cnn", "tiny_vit"],
        choices=["custom_cnn", "tiny_vit"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs-cnn", type=int, default=6)
    parser.add_argument("--epochs-vit", type=int, default=6)
    parser.add_argument("--batch-size-cnn", type=int, default=64)
    parser.add_argument("--batch-size-vit", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-cnn-baseline", type=float, default=1e-3)
    parser.add_argument("--lr-vit-baseline", type=float, default=5e-4)
    parser.add_argument("--lr-cnn-improved", type=float, default=6e-4)
    parser.add_argument("--lr-vit-improved", type=float, default=4e-4)
    parser.add_argument("--vit-patch-size", type=int, default=16)
    parser.add_argument("--vit-dim", type=int, default=192)
    parser.add_argument("--vit-depth", type=int, default=8)
    parser.add_argument("--vit-heads", type=int, default=6)
    parser.add_argument("--vit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--vit-dropout", type=float, default=0.1)
    parser.add_argument("--vit-drop-path-baseline", type=float, default=0.10)
    parser.add_argument("--vit-drop-path-improved", type=float, default=0.15)
    parser.add_argument("--vit-label-smoothing-baseline", type=float, default=0.05)
    parser.add_argument("--vit-label-smoothing-improved", type=float, default=0.08)
    parser.add_argument("--vit-mixup-alpha-baseline", type=float, default=0.0)
    parser.add_argument("--vit-mixup-alpha-improved", type=float, default=0.0)
    parser.add_argument("--vit-warmup-epochs-baseline", type=int, default=3)
    parser.add_argument("--vit-warmup-epochs-improved", type=int, default=3)
    parser.add_argument("--vit-min-lr-ratio-baseline", type=float, default=0.15)
    parser.add_argument("--vit-min-lr-ratio-improved", type=float, default=0.05)
    parser.add_argument("--improved-random-erasing-prob", type=float, default=0.15)
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


def load_or_create_splits(
    args: argparse.Namespace, metadata_df: pd.DataFrame
) -> pd.DataFrame:
    if args.split_csv.exists():
        split_df = pd.read_csv(args.split_csv)
        required_cols = {"path", "label", "class_name", "split"}
        if not required_cols.issubset(set(split_df.columns)):
            raise ValueError(
                f"Split file {args.split_csv} misses required columns {required_cols}"
            )
        split_df["path"] = split_df["path"].astype(str)
        # Some split files can contain absolute paths from another workspace.
        # If files are missing, remap them to current data_root using class and filename.
        exists_mask = split_df["path"].map(lambda p: Path(p).exists())
        if not bool(exists_mask.all()):
            split_df = split_df.copy()
            split_df["path"] = split_df.apply(
                lambda row: str((args.data_root / row["class_name"] / Path(row["path"]).name).resolve()),
                axis=1,
            )
            exists_mask = split_df["path"].map(lambda p: Path(p).exists())
            if not bool(exists_mask.all()):
                missing_count = int((~exists_mask).sum())
                raise FileNotFoundError(
                    f"Could not remap {missing_count} paths from split file {args.split_csv} "
                    f"to current data_root {args.data_root}"
                )
        return split_df

    split_df = make_splits(
        metadata_df,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    return split_df


def make_transforms(
    img_size: int,
    phase: str,
    train: bool,
    model_name: str,
    improved_random_erasing_prob: float,
) -> transforms.Compose:
    if not train:
        return transforms.Compose(
            [
                transforms.Resize(img_size + 32),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if phase == "baseline":
        return transforms.Compose(
            [
                transforms.Resize(img_size + 32),
                transforms.CenterCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    # phase == "improved": stronger but still lightweight augmentation recipe.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.78, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)],
                p=0.5,
            ),
            transforms.RandomRotation(degrees=7, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=improved_random_erasing_prob if model_name == "tiny_vit" else 0.08,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.features = nn.Sequential(
            ConvBlock(3, c1),
            ConvBlock(c1, c1),
            nn.MaxPool2d(2),
            ConvBlock(c1, c2),
            ConvBlock(c2, c2),
            nn.MaxPool2d(2),
            ConvBlock(c2, c3),
            ConvBlock(c3, c3),
            nn.MaxPool2d(2),
            ConvBlock(c3, c4),
            ConvBlock(c4, c4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma_2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path(self.gamma_1 * attn_out)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 16,
        dim: int = 192,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if patch_size % 4 != 0:
            raise ValueError("patch_size must be divisible by 4 for conv stem")

        stem_dim = max(64, dim // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )
        self.patch_embed = nn.Conv2d(
            in_channels=stem_dim,
            out_channels=dim,
            kernel_size=patch_size // 4,
            stride=patch_size // 4,
        )

        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr_values = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=float(dpr_values[idx]),
                    layer_scale_init=layer_scale_init,
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        bsz = x.shape[0]
        cls_token = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        pooled = 0.5 * (x[:, 0] + x[:, 1:].mean(dim=1))
        return self.head(pooled)


def build_model(
    model_name: str,
    num_classes: int,
    img_size: int,
    args: argparse.Namespace,
) -> nn.Module:
    if model_name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if model_name == "tiny_vit":
        drop_path = (
            args.vit_drop_path_improved
            if args.phase == "improved"
            else args.vit_drop_path_baseline
        )
        return TinyViT(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=args.vit_patch_size,
            dim=args.vit_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            mlp_ratio=args.vit_mlp_ratio,
            dropout=args.vit_dropout,
            drop_path_rate=drop_path,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_epochs(args: argparse.Namespace, model_name: str) -> int:
    return args.epochs_cnn if model_name == "custom_cnn" else args.epochs_vit


def resolve_batch_size(args: argparse.Namespace, model_name: str) -> int:
    return args.batch_size_cnn if model_name == "custom_cnn" else args.batch_size_vit


def resolve_lr(args: argparse.Namespace, model_name: str) -> float:
    if args.phase == "baseline":
        return args.lr_cnn_baseline if model_name == "custom_cnn" else args.lr_vit_baseline
    return args.lr_cnn_improved if model_name == "custom_cnn" else args.lr_vit_improved


def resolve_vit_label_smoothing(args: argparse.Namespace) -> float:
    if args.phase == "baseline":
        return args.vit_label_smoothing_baseline
    return args.vit_label_smoothing_improved


def resolve_vit_mixup_alpha(args: argparse.Namespace) -> float:
    if args.phase == "baseline":
        return args.vit_mixup_alpha_baseline
    return args.vit_mixup_alpha_improved


def resolve_vit_warmup_epochs(args: argparse.Namespace) -> int:
    if args.phase == "baseline":
        return args.vit_warmup_epochs_baseline
    return args.vit_warmup_epochs_improved


def resolve_vit_min_lr_ratio(args: argparse.Namespace) -> float:
    if args.phase == "baseline":
        return args.vit_min_lr_ratio_baseline
    return args.vit_min_lr_ratio_improved


def build_weight_decay_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "pos_embed" in name
            or "cls_token" in name
            or "gamma_1" in name
            or "gamma_2" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
    model_name: str,
) -> torch.optim.Optimizer:
    lr = resolve_lr(args, model_name)
    if model_name == "tiny_vit":
        param_groups = build_weight_decay_param_groups(model, weight_decay=args.weight_decay)
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    if args.phase == "baseline":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    model_name: str,
    epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if model_name == "tiny_vit":
        return None
    if args.phase == "improved":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
            eta_min=resolve_lr(args, model_name) * 0.1,
        )
    return None


def update_vit_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        scale = epoch / warmup_epochs
    else:
        if total_epochs <= warmup_epochs:
            progress = 1.0
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    lr = base_lr * scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


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
    grad_clip_norm: float,
    mixup_alpha: float = 0.0,
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

        mixed_targets: torch.Tensor | None = None
        mixup_lambda = 1.0
        if is_train and mixup_alpha > 0.0 and images.size(0) > 1:
            mixup_lambda = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(images.size(0), device=device)
            images = mixup_lambda * images + (1.0 - mixup_lambda) * images[permutation]
            mixed_targets = targets[permutation]

        logits = model(images)
        if mixed_targets is None:
            loss = criterion(logits, targets)
        else:
            loss = mixup_lambda * criterion(logits, targets) + (1.0 - mixup_lambda) * criterion(
                logits, mixed_targets
            )

        if is_train:
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = epoch_loss
    return metrics, all_targets, all_preds


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
    args: argparse.Namespace,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_ds = RecordsDataset(
        records=df_to_records(train_df),
        transform=make_transforms(
            args.img_size,
            phase=args.phase,
            train=True,
            model_name=model_name,
            improved_random_erasing_prob=args.improved_random_erasing_prob,
        ),
    )
    val_ds = RecordsDataset(
        records=df_to_records(val_df),
        transform=make_transforms(
            args.img_size,
            phase=args.phase,
            train=False,
            model_name=model_name,
            improved_random_erasing_prob=args.improved_random_erasing_prob,
        ),
    )
    test_ds = RecordsDataset(
        records=df_to_records(test_df),
        transform=make_transforms(
            args.img_size,
            phase=args.phase,
            train=False,
            model_name=model_name,
            improved_random_erasing_prob=args.improved_random_erasing_prob,
        ),
    )

    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
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
    run_output_root: Path,
    run_plots_root: Path,
) -> dict:
    epochs = resolve_epochs(args, model_name)
    batch_size = resolve_batch_size(args, model_name)
    lr = resolve_lr(args, model_name)

    model_dir = run_output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        split_df=split_df,
        args=args,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        img_size=args.img_size,
        args=args,
    )
    model.to(device)
    if model_name == "tiny_vit":
        criterion = nn.CrossEntropyLoss(label_smoothing=resolve_vit_label_smoothing(args))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model=model, args=args, model_name=model_name)
    scheduler = make_scheduler(
        optimizer=optimizer,
        args=args,
        model_name=model_name,
        epochs=epochs,
    )
    vit_mixup_alpha = resolve_vit_mixup_alpha(args) if model_name == "tiny_vit" else 0.0
    vit_warmup_epochs = resolve_vit_warmup_epochs(args)
    vit_min_lr_ratio = resolve_vit_min_lr_ratio(args)

    history_rows: list[dict] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_path = model_dir / "best.pt"

    params_count = count_trainable_params(model)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if model_name == "tiny_vit":
            update_vit_lr(
                optimizer=optimizer,
                base_lr=lr,
                epoch=epoch,
                total_epochs=epochs,
                warmup_epochs=vit_warmup_epochs,
                min_lr_ratio=vit_min_lr_ratio,
            )

        train_metrics, _, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            grad_clip_norm=args.grad_clip_norm if (args.phase == "improved" or model_name == "tiny_vit") else 0.0,
            mixup_alpha=vit_mixup_alpha,
            optimizer=optimizer,
        )
        val_metrics, _, _ = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            grad_clip_norm=0.0,
            mixup_alpha=0.0,
            optimizer=None,
        )
        if scheduler is not None:
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
        history_df=history_df,
        model_name=model_name,
        out_path=run_plots_root / f"{model_name}_learning_curves.png",
    )

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, y_true, y_pred = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        grad_clip_norm=0.0,
        mixup_alpha=0.0,
        optimizer=None,
    )
    plot_confusion(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        title=f"{model_name} confusion matrix (normalized)",
        out_path=run_plots_root / f"{model_name}_confusion_matrix.png",
    )
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        model_dir / "test_predictions.csv", index=False
    )

    metrics_payload = {
        "model_name": model_name,
        "phase": args.phase,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "train_seconds": train_seconds,
        "params_trainable": params_count,
        "lr": lr,
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

    run_output_root = args.output_root / args.run_name
    run_plots_root = args.plots_root / args.run_name
    run_output_root.mkdir(parents=True, exist_ok=True)
    run_plots_root.mkdir(parents=True, exist_ok=True)

    metadata_df, class_names = collect_metadata(args.data_root)
    split_df = load_or_create_splits(args=args, metadata_df=metadata_df)
    split_df = split_df.sort_values(by=["split", "class_name", "path"]).reset_index(drop=True)
    split_df.to_csv(run_output_root / "splits.csv", index=False)

    run_config = {
        "run_name": args.run_name,
        "phase": args.phase,
        "seed": args.seed,
        "device": str(device),
        "models": args.models,
        "img_size": args.img_size,
        "epochs_cnn": args.epochs_cnn,
        "epochs_vit": args.epochs_vit,
        "batch_size_cnn": args.batch_size_cnn,
        "batch_size_vit": args.batch_size_vit,
        "num_workers": args.num_workers,
        "lr_cnn_baseline": args.lr_cnn_baseline,
        "lr_vit_baseline": args.lr_vit_baseline,
        "lr_cnn_improved": args.lr_cnn_improved,
        "lr_vit_improved": args.lr_vit_improved,
        "vit_patch_size": args.vit_patch_size,
        "vit_dim": args.vit_dim,
        "vit_depth": args.vit_depth,
        "vit_heads": args.vit_heads,
        "vit_mlp_ratio": args.vit_mlp_ratio,
        "vit_dropout": args.vit_dropout,
        "vit_drop_path_baseline": args.vit_drop_path_baseline,
        "vit_drop_path_improved": args.vit_drop_path_improved,
        "vit_label_smoothing_baseline": args.vit_label_smoothing_baseline,
        "vit_label_smoothing_improved": args.vit_label_smoothing_improved,
        "vit_mixup_alpha_baseline": args.vit_mixup_alpha_baseline,
        "vit_mixup_alpha_improved": args.vit_mixup_alpha_improved,
        "vit_warmup_epochs_baseline": args.vit_warmup_epochs_baseline,
        "vit_warmup_epochs_improved": args.vit_warmup_epochs_improved,
        "vit_min_lr_ratio_baseline": args.vit_min_lr_ratio_baseline,
        "vit_min_lr_ratio_improved": args.vit_min_lr_ratio_improved,
        "improved_random_erasing_prob": args.improved_random_erasing_prob,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "split_csv": str(args.split_csv),
        "data_root": str(args.data_root.resolve()),
    }
    with (run_output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print(f"Using device: {device}")
    print(f"Run name: {args.run_name}")
    print(f"Phase: {args.phase}")

    all_results: list[dict] = []
    for model_name in args.models:
        print(f"\n=== Training {model_name} ===")
        result = train_model(
            model_name=model_name,
            split_df=split_df,
            class_names=class_names,
            args=args,
            device=device,
            run_output_root=run_output_root,
            run_plots_root=run_plots_root,
        )
        all_results.append(result)

    summary_rows = []
    for result in all_results:
        m = result["test_metrics"]
        summary_rows.append(
            {
                "model_name": result["model_name"],
                "phase": result["phase"],
                "params_trainable": result["params_trainable"],
                "best_epoch": result["best_epoch"],
                "best_val_macro_f1": result["best_val_macro_f1"],
                "train_seconds": result["train_seconds"],
                "lr": result["lr"],
                "test_accuracy": m["accuracy"],
                "test_macro_f1": m["macro_f1"],
                "test_weighted_f1": m["weighted_f1"],
                "test_macro_precision": m["macro_precision"],
                "test_macro_recall": m["macro_recall"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(by="test_macro_f1", ascending=False)
    summary_df.to_csv(run_output_root / "summary_metrics.csv", index=False)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
