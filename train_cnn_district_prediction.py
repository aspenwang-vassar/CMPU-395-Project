#!/usr/bin/env python3
"""
CNN-based district-level prediction from Street View images.

The educational outcome is defined at the school-district level, not at the
individual image level. All splits, aggregation, metrics, and predictions in
this script therefore operate by district_id. Images from the same district are
never split across train/validation/test.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


LOGGER = logging.getLogger("train_cnn_district_prediction")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EMBEDDING_DIM = 2048
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainResult:
    model_name: str
    best_epoch: int
    history: pd.DataFrame
    checkpoint_path: Path


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "training.log")
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_district_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text.replace(".0", "")


def resolve_id_column(df: pd.DataFrame, requested: str, frame_name: str) -> str:
    candidates = [requested, "district_id", "sedalea", "SEDALEA", "lea_id", "LEAID"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No district id column found in {frame_name}. Tried {candidates}; "
        f"available columns are {df.columns.tolist()}"
    )


def resolve_image_path(raw_path: object, image_root: Path, metadata_path: Path) -> Optional[Path]:
    if pd.isna(raw_path):
        return None
    path_text = str(raw_path).strip()
    if not path_text:
        return None

    path = Path(path_text)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([image_root / path, metadata_path.parent / path, Path.cwd() / path])
        candidates.append(image_root / path.name)

    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate.resolve()
    return None


def load_and_clean_data(
    image_metadata: Path,
    district_outcomes: Path,
    image_root: Path,
    target_col: str,
    district_id_col: str,
    min_images_per_district: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge image rows to district labels and keep only valid district-level examples."""
    LOGGER.info("Loading image metadata from %s", image_metadata)
    metadata_df = pd.read_csv(image_metadata)
    LOGGER.info("Loading district outcomes from %s", district_outcomes)
    outcomes_df = pd.read_csv(district_outcomes)

    metadata_id_col = resolve_id_column(metadata_df, district_id_col, "image metadata")
    outcome_id_col = resolve_id_column(outcomes_df, district_id_col, "district outcomes")

    if "image_path" not in metadata_df.columns:
        raise ValueError("Image metadata must contain an image_path column.")
    if target_col not in outcomes_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in outcomes. "
            f"Available columns: {outcomes_df.columns.tolist()}"
        )

    metadata_df = metadata_df.copy()
    outcomes_df = outcomes_df.copy()
    metadata_df["district_id"] = metadata_df[metadata_id_col].map(normalize_district_id)
    outcomes_df["district_id"] = outcomes_df[outcome_id_col].map(normalize_district_id)

    outcomes_keep = ["district_id", target_col]
    for optional_col in ["math_score", "rla_score", "reddit_sentiment", "stateabb", "sedaleaname"]:
        if optional_col in outcomes_df.columns and optional_col not in outcomes_keep:
            outcomes_keep.append(optional_col)
    outcomes_df = outcomes_df[outcomes_keep].dropna(subset=[target_col]).drop_duplicates("district_id")

    metadata_df["image_path"] = metadata_df["image_path"].map(
        lambda p: resolve_image_path(p, image_root=image_root, metadata_path=image_metadata)
    )
    metadata_df = metadata_df.dropna(subset=["district_id", "image_path"]).copy()
    metadata_df["image_path"] = metadata_df["image_path"].astype(str)

    keep_meta_cols = [
        col
        for col in [
            "district_id",
            "image_path",
            "pano_id",
            "heading",
            "lat",
            "lon",
            "pano_lat",
            "pano_lon",
            "date",
            "stateabb",
        ]
        if col in metadata_df.columns
    ]
    if "district_id" not in keep_meta_cols:
        keep_meta_cols.insert(0, "district_id")
    if "image_path" not in keep_meta_cols:
        keep_meta_cols.insert(1, "image_path")
    metadata_df = metadata_df[keep_meta_cols].drop_duplicates(["district_id", "image_path"])

    merged_df = metadata_df.merge(outcomes_df, on="district_id", how="inner", suffixes=("", "_outcome"))
    merged_df = merged_df.dropna(subset=["image_path", target_col]).copy()
    counts = merged_df.groupby("district_id")["image_path"].nunique().rename("image_count")
    merged_df = merged_df.merge(counts, on="district_id", how="left")
    merged_df = merged_df[merged_df["image_count"] >= min_images_per_district].copy()
    merged_df = merged_df.sort_values(["district_id", "image_path"]).reset_index(drop=True)

    if merged_df.empty:
        raise ValueError(
            "No modeling rows remain after cleaning. Check image paths, target column, "
            "district IDs, and min_images_per_district."
        )

    district_df = (
        merged_df[["district_id", target_col, "image_count"] + [c for c in ["reddit_sentiment", "stateabb", "sedaleaname"] if c in merged_df.columns]]
        .drop_duplicates("district_id")
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_dir / "cleaned_modeling_metadata.csv", index=False)
    LOGGER.info(
        "Cleaned data: %s images across %s districts after min_images_per_district=%s",
        len(merged_df),
        merged_df["district_id"].nunique(),
        min_images_per_district,
    )
    return merged_df, district_df


def create_district_splits(
    district_df: pd.DataFrame,
    output_dir: Path,
    random_seed: int,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    """Create reproducible train/validation/test assignments by district_id."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    district_ids = district_df["district_id"].drop_duplicates().to_numpy()
    if len(district_ids) < 10:
        raise ValueError("Need at least 10 districts for a stable train/val/test split.")

    train_ids, temp_ids = train_test_split(
        district_ids,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True,
    )
    relative_val = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=relative_val,
        random_state=random_seed,
        shuffle=True,
    )

    split_df = pd.DataFrame(
        {
            "district_id": list(train_ids) + list(val_ids) + list(test_ids),
            "split": ["train"] * len(train_ids) + ["val"] * len(val_ids) + ["test"] * len(test_ids),
        }
    )
    split_df = split_df.merge(district_df[["district_id", "image_count"]], on="district_id", how="left")
    split_df.to_csv(output_dir / "split_assignments.csv", index=False)
    LOGGER.info(
        "District split: train=%s val=%s test=%s",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )
    return split_df


def get_image_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class StreetViewImageDataset(Dataset):
    """Image-level dataset where each image inherits its district-level target."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        transform: transforms.Compose,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to load image {row['image_path']}: {exc}") from exc
        return {
            "image": self.transform(image),
            "target": torch.tensor(float(row[self.target_col]), dtype=torch.float32),
            "district_id": row["district_id"],
            "image_path": row["image_path"],
        }


class DistrictBagDataset(Dataset):
    """Multiple-instance dataset where each item is a district bag of up to K images."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        transform: transforms.Compose,
        bag_size: int,
        training: bool,
        random_seed: int,
    ) -> None:
        self.df = df.copy()
        self.target_col = target_col
        self.transform = transform
        self.bag_size = bag_size
        self.training = training
        self.random_seed = random_seed
        self.district_ids = sorted(self.df["district_id"].unique())
        self.groups = {district_id: group.reset_index(drop=True) for district_id, group in self.df.groupby("district_id")}

    def __len__(self) -> int:
        return len(self.district_ids)

    def _select_rows(self, district_id: str, idx: int) -> pd.DataFrame:
        group = self.groups[district_id]
        replace = len(group) < self.bag_size
        if self.training:
            sampled_idx = np.random.choice(len(group), size=self.bag_size, replace=replace)
        else:
            rng = np.random.default_rng(self.random_seed + idx)
            sampled_idx = np.arange(len(group))[: self.bag_size]
            if len(sampled_idx) < self.bag_size:
                pad = rng.choice(len(group), size=self.bag_size - len(sampled_idx), replace=True)
                sampled_idx = np.concatenate([sampled_idx, pad])
        return group.iloc[sampled_idx]

    def __getitem__(self, idx: int) -> dict[str, object]:
        district_id = self.district_ids[idx]
        rows = self._select_rows(district_id, idx)
        images = []
        for image_path in rows["image_path"]:
            try:
                image = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                raise RuntimeError(f"Failed to load image {image_path}: {exc}") from exc
            images.append(self.transform(image))

        target = float(rows[self.target_col].iloc[0])
        return {
            "images": torch.stack(images, dim=0),
            "target": torch.tensor(target, dtype=torch.float32),
            "district_id": district_id,
            "image_count": int(rows["image_count"].iloc[0]),
        }


def build_resnet_encoder(freeze_backbone: bool = True, unfreeze_layer4: bool = False) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True
    return model


def build_resnet_regressor(freeze_backbone: bool = True, unfreeze_layer4: bool = False) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True
    return model


class MILResNetRegressor(nn.Module):
    def __init__(self, freeze_backbone: bool = True, unfreeze_layer4: bool = False) -> None:
        super().__init__()
        self.encoder = build_resnet_encoder(freeze_backbone=freeze_backbone, unfreeze_layer4=unfreeze_layer4)
        self.regressor = nn.Sequential(
            nn.LayerNorm(EMBEDDING_DIM),
            nn.Dropout(0.2),
            nn.Linear(EMBEDDING_DIM, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, bag_size, channels, height, width = images.shape
        flat_images = images.view(batch_size * bag_size, channels, height, width)
        features = self.encoder(flat_images).view(batch_size, bag_size, -1)
        pooled = features.mean(dim=1)
        return self.regressor(pooled).squeeze(1)


def evaluate_regression(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    metrics = {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) >= 2 else np.nan,
        "pearson": np.nan,
    }
    if len(y_true_arr) >= 2 and np.std(y_true_arr) > 0 and np.std(y_pred_arr) > 0:
        metrics["pearson"] = float(pearsonr(y_true_arr, y_pred_arr).statistic)
    return metrics


def metrics_record(model: str, split: str, y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, object]:
    metrics = evaluate_regression(y_true, y_pred)
    return {"model": model, "split": split, **metrics, "n_districts": len(y_true)}


def aggregate_image_predictions_by_district(
    prediction_df: pd.DataFrame,
    split: str,
    model_name: str,
) -> pd.DataFrame:
    grouped = (
        prediction_df.groupby("district_id")
        .agg(y_true=("y_true", "first"), y_pred=("image_pred", "mean"), image_count=("image_path", "nunique"))
        .reset_index()
    )
    grouped["split"] = split
    grouped["model"] = model_name
    return grouped[["model", "district_id", "y_true", "y_pred", "image_count", "split"]]


@torch.no_grad()
def predict_image_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_col: str,
    split: str,
    model_name: str,
) -> pd.DataFrame:
    del target_col
    model.eval()
    rows: list[dict[str, object]] = []
    for batch in tqdm(loader, desc=f"predict {model_name} {split}", leave=False):
        images = batch["image"].to(device)
        preds = model(images).squeeze(1).detach().cpu().numpy()
        targets = batch["target"].detach().cpu().numpy()
        for district_id, image_path, y_true, y_pred in zip(batch["district_id"], batch["image_path"], targets, preds):
            rows.append(
                {
                    "district_id": district_id,
                    "image_path": image_path,
                    "y_true": float(y_true),
                    "image_pred": float(y_pred),
                    "split": split,
                    "model": model_name,
                }
            )
    return aggregate_image_predictions_by_district(pd.DataFrame(rows), split=split, model_name=model_name)


@torch.no_grad()
def predict_mil_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    model_name: str,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    for batch in tqdm(loader, desc=f"predict {model_name} {split}", leave=False):
        images = batch["images"].to(device)
        preds = model(images).detach().cpu().numpy()
        targets = batch["target"].detach().cpu().numpy()
        for district_id, image_count, y_true, y_pred in zip(batch["district_id"], batch["image_count"], targets, preds):
            rows.append(
                {
                    "model": model_name,
                    "district_id": district_id,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "image_count": int(image_count),
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def train_image_cnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    freeze_backbone: bool,
    unfreeze_layer4: bool,
    num_workers: int,
    patience: int = 5,
) -> tuple[nn.Module, TrainResult]:
    train_transform, eval_transform = get_image_transforms()
    train_loader = DataLoader(
        StreetViewImageDataset(train_df, target_col, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        StreetViewImageDataset(val_df, target_col, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet_regressor(freeze_backbone=freeze_backbone, unfreeze_layer4=unfreeze_layer4).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    loss_fn = nn.MSELoss()
    checkpoint_path = output_dir / "checkpoints" / "image_cnn_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"image_cnn epoch {epoch}", leave=False):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(images).squeeze(1)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_preds = predict_image_model(model, val_loader, device, target_col, "val", "image_cnn")
        val_metrics = evaluate_regression(val_preds["y_true"], val_preds["y_pred"])
        train_loss = float(np.mean(losses)) if losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        LOGGER.info("image_cnn epoch=%s train_loss=%.4f val_rmse=%.4f", epoch, train_loss, val_metrics["rmse"])

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_rmse": best_rmse}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("Early stopping image_cnn at epoch %s", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "image_cnn_training_curve.csv", index=False)
    return model, TrainResult("image_cnn", best_epoch, history_df, checkpoint_path)


def train_mil_cnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    bag_size: int,
    freeze_backbone: bool,
    unfreeze_layer4: bool,
    random_seed: int,
    num_workers: int,
    patience: int = 5,
) -> tuple[nn.Module, TrainResult]:
    train_transform, eval_transform = get_image_transforms()
    train_loader = DataLoader(
        DistrictBagDataset(train_df, target_col, train_transform, bag_size, training=True, random_seed=random_seed),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        DistrictBagDataset(val_df, target_col, eval_transform, bag_size, training=False, random_seed=random_seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MILResNetRegressor(freeze_backbone=freeze_backbone, unfreeze_layer4=unfreeze_layer4).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    loss_fn = nn.MSELoss()
    checkpoint_path = output_dir / "checkpoints" / "mil_cnn_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"mil_cnn epoch {epoch}", leave=False):
            images = batch["images"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_preds = predict_mil_model(model, val_loader, device, "val", "mil_cnn")
        val_metrics = evaluate_regression(val_preds["y_true"], val_preds["y_pred"])
        train_loss = float(np.mean(losses)) if losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        LOGGER.info("mil_cnn epoch=%s train_loss=%.4f val_rmse=%.4f", epoch, train_loss, val_metrics["rmse"])

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_rmse": best_rmse}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("Early stopping mil_cnn at epoch %s", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "mil_cnn_training_curve.csv", index=False)
    return model, TrainResult("mil_cnn", best_epoch, history_df, checkpoint_path)


@torch.no_grad()
def extract_embeddings(
    metadata_df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    embeddings_path = output_dir / "image_embeddings.parquet"
    if embeddings_path.exists():
        LOGGER.info("Loading cached image embeddings from %s", embeddings_path)
        return pd.read_parquet(embeddings_path)

    _, eval_transform = get_image_transforms()
    dataset = StreetViewImageDataset(metadata_df, target_col, eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    encoder = build_resnet_encoder(freeze_backbone=True).to(device)
    encoder.eval()

    records: list[dict[str, object]] = []
    for batch in tqdm(loader, desc="extract embeddings"):
        images = batch["image"].to(device)
        features = encoder(images).detach().cpu().numpy()
        targets = batch["target"].detach().cpu().numpy()
        for district_id, image_path, y_true, feature in zip(batch["district_id"], batch["image_path"], targets, features):
            row = {"district_id": district_id, "image_path": image_path, "y_true": float(y_true)}
            row.update({f"emb_{i}": float(value) for i, value in enumerate(feature)})
            records.append(row)

    embeddings_df = pd.DataFrame(records)
    embeddings_df.to_parquet(embeddings_path, index=False)
    LOGGER.info("Saved image embeddings to %s", embeddings_path)
    return embeddings_df


def embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("emb_")]


def aggregate_embeddings_by_district(embeddings_df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    feature_cols = embedding_columns(embeddings_df)
    district_features = embeddings_df.groupby("district_id")[feature_cols].mean().reset_index()
    district_targets = (
        metadata_df.groupby("district_id")
        .agg(y_true=("target_for_merge", "first"), image_count=("image_path", "nunique"))
        .reset_index()
    )
    if "reddit_sentiment" in metadata_df.columns:
        sentiment = metadata_df.groupby("district_id")["reddit_sentiment"].first().reset_index()
        district_targets = district_targets.merge(sentiment, on="district_id", how="left")
    district_features = district_features.merge(district_targets, on="district_id", how="left")
    district_features.to_parquet(output_dir / "district_features.parquet", index=False)
    LOGGER.info("Saved district features to %s", output_dir / "district_features.parquet")
    return district_features


def fit_ridge_predictions(
    model_name: str,
    district_features: pd.DataFrame,
    split_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    df = district_features.merge(split_df[["district_id", "split"]], on="district_id", how="inner")
    train_df = df[df["split"] == "train"].copy()
    predictions: list[pd.DataFrame] = []
    metrics_records: list[dict[str, object]] = []

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["y_true"].to_numpy(dtype=np.float32)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    for split in ["train", "val", "test"]:
        split_part = df[df["split"] == split].copy()
        X = scaler.transform(split_part[feature_cols].to_numpy(dtype=np.float32))
        split_part["y_pred"] = model.predict(X)
        split_part["model"] = model_name
        predictions.append(split_part[["model", "district_id", "y_true", "y_pred", "image_count", "split"]])
        metrics_records.append(metrics_record(model_name, split, split_part["y_true"], split_part["y_pred"]))

    return metrics_records, pd.concat(predictions, ignore_index=True)


def run_frozen_baseline(
    metadata_df: pd.DataFrame,
    split_df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modeling_df = metadata_df.copy()
    modeling_df["target_for_merge"] = modeling_df[target_col]
    embeddings_df = extract_embeddings(modeling_df, target_col, output_dir, device, batch_size, num_workers)
    district_features = aggregate_embeddings_by_district(embeddings_df, modeling_df, output_dir)

    feature_cols = embedding_columns(district_features)
    metrics_records, predictions = fit_ridge_predictions("frozen_baseline", district_features, split_df, feature_cols)

    if "reddit_sentiment" in district_features.columns and district_features["reddit_sentiment"].notna().any():
        sentiment_df = district_features.dropna(subset=["reddit_sentiment"]).copy()
        sentiment_split = split_df[split_df["district_id"].isin(sentiment_df["district_id"])]
        for name, cols in {
            "sentiment_only": ["reddit_sentiment"],
            "frozen_baseline_plus_sentiment": feature_cols + ["reddit_sentiment"],
        }.items():
            extra_metrics, extra_predictions = fit_ridge_predictions(name, sentiment_df, sentiment_split, cols)
            metrics_records.extend(extra_metrics)
            predictions = pd.concat([predictions, extra_predictions], ignore_index=True)

    return pd.DataFrame(metrics_records), predictions


def save_prediction_plots(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_df in predictions_df.groupby("model"):
        test_df = model_df[model_df["split"] == "test"]
        if test_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(test_df["y_true"], test_df["y_pred"], alpha=0.8)
        min_val = min(test_df["y_true"].min(), test_df["y_pred"].min())
        max_val = max(test_df["y_true"].max(), test_df["y_pred"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
        ax.set_title(f"{model_name}: predicted vs true")
        ax.set_xlabel("True district outcome")
        ax.set_ylabel("Predicted district outcome")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{model_name}_predicted_vs_true.png", dpi=200)
        plt.close(fig)

        residuals = test_df["y_true"] - test_df["y_pred"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(test_df["y_pred"], residuals, alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{model_name}: residuals")
        ax.set_xlabel("Predicted district outcome")
        ax.set_ylabel("Residual")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{model_name}_residuals.png", dpi=200)
        plt.close(fig)


def save_training_curve(history: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    if history.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train MSE")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_rmse"], color="tab:orange", label="val RMSE")
    ax2.set_ylabel("Validation RMSE")
    ax1.set_title(f"{model_name} training curve")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{model_name}_training_curve.png", dpi=200)
    plt.close(fig)


def save_pca_plot(district_features: pd.DataFrame, output_dir: Path) -> None:
    feature_cols = embedding_columns(district_features)
    if len(district_features) < 3 or not feature_cols:
        return
    features = district_features[feature_cols].to_numpy(dtype=np.float32)
    components = PCA(n_components=2, random_state=42).fit_transform(features)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(components[:, 0], components[:, 1], c=district_features["y_true"], cmap="viridis", alpha=0.8)
    ax.set_title("Frozen ResNet district embeddings PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="district outcome")
    fig.tight_layout()
    fig.savefig(plots_dir / "frozen_embeddings_pca.png", dpi=200)
    plt.close(fig)


def split_metadata(metadata_df: pd.DataFrame, split_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    merged = metadata_df.merge(split_df[["district_id", "split"]], on="district_id", how="inner")
    return {split: merged[merged["split"] == split].copy() for split in ["train", "val", "test"]}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train district-level CNN models from Street View images.")
    parser.add_argument("--image-metadata", type=Path, default=Path("outputs/data_collection/downloaded_images.csv"))
    parser.add_argument("--district-outcomes", type=Path, default=Path("data/sampled_districts.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/streetview_images"))
    parser.add_argument("--target-col", default="cs_mn_avg_eb")
    parser.add_argument("--district-id-col", default="district_id")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cnn_district_prediction"))
    parser.add_argument("--min-images-per-district", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-type", choices=["frozen_baseline", "image_cnn", "mil_cnn", "all"], default="all")
    parser.add_argument("--bag-size", type=int, default=16)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--unfreeze-layer4", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def write_experiment_summary(
    args: argparse.Namespace,
    device: torch.device,
    cleaned_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary = [
        "CNN district prediction experiment",
        "",
        "Methodological notes:",
        "- The target is district-level, so this is weakly supervised district-level regression.",
        "- Images are never randomly split across train/test; all splits are by district_id.",
        "- Image-level CNN predictions are averaged by district before evaluation.",
        "- MIL trains directly on district bags and district-level MSE.",
        "",
        f"Device: {device}",
        f"Target column: {args.target_col}",
        f"Images after cleaning: {len(cleaned_df)}",
        f"Districts after cleaning: {cleaned_df['district_id'].nunique()}",
        f"Minimum images per district: {args.min_images_per_district}",
        "",
        "Metrics:",
        metrics_df.to_string(index=False) if not metrics_df.empty else "No metrics were produced.",
        "",
        "Arguments:",
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2),
    ]
    (output_dir / "experiment_summary.txt").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.random_seed)
    setup_logging(args.output_dir)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    device = infer_device()
    LOGGER.info("Using device: %s", device)

    cleaned_df, district_df = load_and_clean_data(
        image_metadata=args.image_metadata,
        district_outcomes=args.district_outcomes,
        image_root=args.image_root,
        target_col=args.target_col,
        district_id_col=args.district_id_col,
        min_images_per_district=args.min_images_per_district,
        output_dir=args.output_dir,
    )
    split_df = create_district_splits(district_df, args.output_dir, args.random_seed)
    split_dfs = split_metadata(cleaned_df, split_df)

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    district_features_for_pca: Optional[pd.DataFrame] = None

    if args.model_type in {"frozen_baseline", "all"}:
        metrics_df, predictions_df = run_frozen_baseline(
            cleaned_df,
            split_df,
            args.target_col,
            args.output_dir,
            device,
            args.batch_size,
            args.num_workers,
        )
        all_metrics.append(metrics_df)
        all_predictions.append(predictions_df)
        district_features_path = args.output_dir / "district_features.parquet"
        if district_features_path.exists():
            district_features_for_pca = pd.read_parquet(district_features_path)

    if args.model_type in {"image_cnn", "all"}:
        model, result = train_image_cnn(
            split_dfs["train"],
            split_dfs["val"],
            args.target_col,
            args.output_dir,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.freeze_backbone,
            args.unfreeze_layer4,
            args.num_workers,
        )
        save_training_curve(result.history, "image_cnn", args.output_dir)
        _, eval_transform = get_image_transforms()
        image_metrics = []
        image_predictions = []
        for split in ["train", "val", "test"]:
            loader = DataLoader(
                StreetViewImageDataset(split_dfs[split], args.target_col, eval_transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            preds = predict_image_model(model, loader, device, args.target_col, split, "image_cnn")
            image_predictions.append(preds)
            image_metrics.append(metrics_record("image_cnn", split, preds["y_true"], preds["y_pred"]))
        all_metrics.append(pd.DataFrame(image_metrics))
        all_predictions.append(pd.concat(image_predictions, ignore_index=True))

    if args.model_type in {"mil_cnn", "all"}:
        model, result = train_mil_cnn(
            split_dfs["train"],
            split_dfs["val"],
            args.target_col,
            args.output_dir,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.bag_size,
            args.freeze_backbone,
            args.unfreeze_layer4,
            args.random_seed,
            args.num_workers,
        )
        save_training_curve(result.history, "mil_cnn", args.output_dir)
        _, eval_transform = get_image_transforms()
        mil_metrics = []
        mil_predictions = []
        for split in ["train", "val", "test"]:
            loader = DataLoader(
                DistrictBagDataset(split_dfs[split], args.target_col, eval_transform, args.bag_size, training=False, random_seed=args.random_seed),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            preds = predict_mil_model(model, loader, device, split, "mil_cnn")
            mil_predictions.append(preds)
            mil_metrics.append(metrics_record("mil_cnn", split, preds["y_true"], preds["y_pred"]))
        all_metrics.append(pd.DataFrame(mil_metrics))
        all_predictions.append(pd.concat(mil_predictions, ignore_index=True))

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df.to_csv(args.output_dir / "model_metrics.csv", index=False)
    predictions_df.to_csv(args.output_dir / "district_predictions.csv", index=False)
    save_prediction_plots(predictions_df, args.output_dir)
    if district_features_for_pca is not None:
        save_pca_plot(district_features_for_pca, args.output_dir)
    write_experiment_summary(args, device, cleaned_df, metrics_df, args.output_dir)

    LOGGER.info("Saved metrics to %s", args.output_dir / "model_metrics.csv")
    LOGGER.info("Saved district predictions to %s", args.output_dir / "district_predictions.csv")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
