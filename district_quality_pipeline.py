#!/usr/bin/env python3
"""
District-level educational quality prediction from Street View images.

This pipeline:
1. Loads image metadata and district outcomes.
2. Extracts pretrained ResNet-50 embeddings for each image.
3. Mean-pools image embeddings to district-level features.
4. Trains district-level regression baselines.
5. Writes metrics, predictions, cached features, and an optional PCA plot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from tqdm import tqdm


LOGGER = logging.getLogger("district_quality_pipeline")
IMAGE_SIZE = 224
EMBEDDING_DIM = 2048
DEFAULT_ID_COLUMN = "district_id"
DEFAULT_TARGET_COLUMN = "target_score"


class Config:
    """Editable script configuration for direct execution."""

    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "district_quality"

    IMAGE_METADATA_PATH = DATA_DIR / "your_image_metadata.csv"
    AUTO_GENERATE_METADATA = True
    GENERATED_METADATA_PATH = DATA_DIR / "generated_image_metadata.csv"
    DISTRICT_OUTCOME_PATH = DATA_DIR / "seda_geodist_annualsub_cs_6.0.csv"
    IMAGE_ROOT = DATA_DIR / "streetview_images"

    METADATA_ID_COLUMN = "sedalea"
    OUTCOME_ID_COLUMN = "sedalea"
    TARGET_COLUMN = "cs_mn_avg_mth_eb"

    MIN_IMAGES_PER_DISTRICT = 20
    BATCH_SIZE = 64
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    DEVICE = None
    EMBEDDINGS_FORMAT = "parquet"
    FORCE_RECOMPUTE_EMBEDDINGS = False
    SAVE_PCA_PLOT = True


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def infer_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_column(df: pd.DataFrame, column: str, frame_name: str) -> None:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' missing from {frame_name}. "
            f"Available columns: {df.columns.tolist()}"
        )


def normalize_district_ids(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def resolve_image_path(raw_path: str, image_root: Optional[Path], metadata_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if image_root is not None:
        candidate = image_root / path
        if candidate.exists():
            return candidate.resolve()
    candidate = metadata_path.parent / path
    if candidate.exists():
        return candidate.resolve()
    return path.resolve()


def get_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def generate_image_metadata_from_directory(
    image_root: Path,
    output_path: Path,
    metadata_id_column: str,
) -> Path:
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    image_paths = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No image files found under {image_root}")

    records: list[dict[str, str]] = []
    for image_path in image_paths:
        district_token = image_path.stem.split("_")[0]
        district_id = district_token.replace(".0", "").strip()
        if not district_id:
            LOGGER.warning("Skipping image with unparseable district id: %s", image_path)
            continue
        records.append(
            {
                metadata_id_column: district_id,
                "image_path": str(image_path.resolve()),
            }
        )

    if not records:
        raise ValueError(
            "No usable metadata rows could be generated from image filenames in "
            f"{image_root}"
        )

    metadata_df = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)
    LOGGER.info("Generated %s image metadata rows at %s", len(metadata_df), output_path)
    return output_path


def load_data(
    image_metadata_path: Path,
    district_outcome_path: Path,
    metadata_id_column: str = DEFAULT_ID_COLUMN,
    outcome_id_column: str = DEFAULT_ID_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    min_images_per_district: Optional[int] = None,
    image_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Loading image metadata from %s", image_metadata_path)
    metadata_df = pd.read_csv(image_metadata_path)
    LOGGER.info("Loading district outcomes from %s", district_outcome_path)
    outcomes_df = pd.read_csv(district_outcome_path)

    ensure_column(metadata_df, metadata_id_column, "image metadata")
    ensure_column(metadata_df, "image_path", "image metadata")
    ensure_column(outcomes_df, outcome_id_column, "district outcomes")
    ensure_column(outcomes_df, target_column, "district outcomes")

    metadata_df = metadata_df.copy()
    outcomes_df = outcomes_df.copy()

    metadata_df["district_id"] = normalize_district_ids(metadata_df[metadata_id_column])
    outcomes_df["district_id"] = normalize_district_ids(outcomes_df[outcome_id_column])

    metadata_df = metadata_df.dropna(subset=["image_path"])
    outcomes_df = outcomes_df.dropna(subset=[target_column])

    shared_ids = set(metadata_df["district_id"]).intersection(outcomes_df["district_id"])
    metadata_df = metadata_df[metadata_df["district_id"].isin(shared_ids)].copy()
    outcomes_df = outcomes_df[outcomes_df["district_id"].isin(shared_ids)].copy()

    metadata_df["image_path"] = metadata_df["image_path"].astype(str).str.strip()
    metadata_df = metadata_df[metadata_df["image_path"] != ""].copy()
    metadata_df["image_path"] = metadata_df["image_path"].map(
        lambda p: str(resolve_image_path(p, image_root=image_root, metadata_path=image_metadata_path))
    )

    image_counts = metadata_df.groupby("district_id").size().rename("image_count_raw")
    metadata_df = metadata_df.merge(image_counts, on="district_id", how="left")

    if min_images_per_district is not None and min_images_per_district > 0:
        metadata_df = metadata_df[
            metadata_df["image_count_raw"] >= min_images_per_district
        ].copy()
        kept_ids = set(metadata_df["district_id"])
        outcomes_df = outcomes_df[outcomes_df["district_id"].isin(kept_ids)].copy()

    metadata_df = metadata_df.sort_values(["district_id", "image_path"]).reset_index(drop=True)
    outcomes_df = outcomes_df.drop_duplicates(subset=["district_id"]).reset_index(drop=True)

    LOGGER.info(
        "Prepared %s image rows across %s districts after filtering",
        len(metadata_df),
        metadata_df["district_id"].nunique(),
    )
    LOGGER.info("Prepared %s district outcome rows", len(outcomes_df))
    return metadata_df, outcomes_df


def build_resnet50_encoder(device: torch.device) -> torch.nn.Module:
    LOGGER.info("Loading pretrained ResNet-50 backbone on %s", device)
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    encoder = torch.nn.Sequential(*list(model.children())[:-1])
    encoder.eval()
    encoder.to(device)
    return encoder


def extract_embedding(
    image_path: str,
    encoder: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> Optional[np.ndarray]:
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        LOGGER.warning("Skipping unreadable image %s: %s", image_path, exc)
        return None

    with torch.no_grad():
        embedding = encoder(tensor)
    return embedding.squeeze().detach().cpu().numpy().astype(np.float32)


def embedding_columns() -> list[str]:
    return [f"emb_{i}" for i in range(EMBEDDING_DIM)]


def preferred_extension(preferred_format: str) -> str:
    return ".parquet" if preferred_format == "parquet" else ".csv"


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception as exc:
            fallback = path.with_suffix(".csv")
            LOGGER.warning(
                "Failed to write parquet to %s (%s). Falling back to %s",
                path,
                exc,
                fallback,
            )
            df.to_csv(fallback, index=False)
            return fallback
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    else:
        raise ValueError(f"Unsupported output format for {path}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format for {path}")


def choose_existing_cache(base_path: Path) -> Optional[Path]:
    candidates = [base_path, base_path.with_suffix(".parquet"), base_path.with_suffix(".csv")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def generate_image_embeddings(
    metadata_df: pd.DataFrame,
    encoder: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    force_recompute: bool = False,
    preferred_format: str = "parquet",
) -> pd.DataFrame:
    transform = get_image_transform()
    batch_size = max(1, batch_size)
    preferred_path = output_dir / f"image_embeddings{preferred_extension(preferred_format)}"
    cache_path = choose_existing_cache(preferred_path)
    log_path = output_dir / "corrupted_images.csv"
    emb_cols = embedding_columns()

    existing_df: Optional[pd.DataFrame] = None
    processed_paths: set[str] = set()
    if cache_path is not None and not force_recompute:
        LOGGER.info("Loading cached image embeddings from %s", cache_path)
        existing_df = read_table(cache_path)
        if "image_path" in existing_df.columns:
            processed_paths = set(existing_df["image_path"].astype(str))
        LOGGER.info("Reusing %s cached image embeddings", len(processed_paths))

    pending_df = metadata_df[~metadata_df["image_path"].isin(processed_paths)].copy()
    LOGGER.info("Need to compute embeddings for %s images", len(pending_df))

    records: list[dict[str, object]] = []
    bad_records: list[dict[str, str]] = []
    batched_tensors: list[torch.Tensor] = []
    batched_meta: list[tuple[str, str]] = []
    flush_every = max(250, batch_size * 4)

    def flush_records() -> None:
        nonlocal cache_path, existing_df, records
        if not records:
            return
        new_df = pd.DataFrame.from_records(records)
        existing_df = new_df if existing_df is None else pd.concat([existing_df, new_df], ignore_index=True)
        cache_path = write_table(existing_df, preferred_path)
        records = []

    def flush_batch() -> None:
        nonlocal batched_meta, batched_tensors, records
        if not batched_tensors:
            return

        with torch.no_grad():
            batch_tensor = torch.stack(batched_tensors, dim=0).to(device)
            batch_embeddings = (
                encoder(batch_tensor)
                .squeeze(-1)
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        for (district_id, image_path), embedding in zip(batched_meta, batch_embeddings):
            record = {"district_id": district_id, "image_path": image_path}
            record.update({col: float(value) for col, value in zip(emb_cols, embedding)})
            records.append(record)

        batched_tensors = []
        batched_meta = []

        if len(records) >= flush_every:
            flush_records()

    for row in tqdm(
        pending_df.itertuples(index=False),
        total=len(pending_df),
        desc="Extracting image embeddings",
    ):
        if batch_size == 1:
            embedding = extract_embedding(row.image_path, encoder, transform, device)
            if embedding is None:
                bad_records.append({"district_id": row.district_id, "image_path": row.image_path})
            else:
                record = {
                    "district_id": row.district_id,
                    "image_path": row.image_path,
                }
                record.update({col: float(value) for col, value in zip(emb_cols, embedding)})
                records.append(record)
                if len(records) >= flush_every:
                    flush_records()
            continue

        try:
            with Image.open(row.image_path) as image:
                image = image.convert("RGB")
                tensor = transform(image)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            LOGGER.warning("Skipping unreadable image %s: %s", row.image_path, exc)
            bad_records.append({"district_id": row.district_id, "image_path": row.image_path})
            continue

        batched_tensors.append(tensor)
        batched_meta.append((row.district_id, row.image_path))

        if len(batched_tensors) >= batch_size:
            flush_batch()

    flush_batch()
    flush_records()

    if existing_df is None:
        existing_df = pd.DataFrame(columns=["district_id", "image_path", *emb_cols])
        cache_path = write_table(existing_df, preferred_path)

    if bad_records:
        bad_df = pd.DataFrame.from_records(bad_records).drop_duplicates()
        bad_df.to_csv(log_path, index=False)
        LOGGER.warning("Logged %s unreadable images to %s", len(bad_df), log_path)

    existing_df = existing_df.drop_duplicates(subset=["district_id", "image_path"]).reset_index(drop=True)
    cache_path = write_table(existing_df, preferred_path)
    LOGGER.info("Saved %s image embeddings to %s", len(existing_df), cache_path)
    return existing_df


def aggregate_district_features(
    embeddings_df: pd.DataFrame,
    output_dir: Path,
    preferred_format: str = "parquet",
) -> pd.DataFrame:
    emb_cols = embedding_columns()
    keep_cols = [col for col in emb_cols if col in embeddings_df.columns]
    if len(keep_cols) != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM} embedding columns but found {len(keep_cols)}."
        )

    district_df = (
        embeddings_df.groupby("district_id")[keep_cols]
        .mean()
        .reset_index()
    )
    image_count_df = (
        embeddings_df.groupby("district_id")
        .size()
        .rename("image_count")
        .reset_index()
    )
    district_df = district_df.merge(image_count_df, on="district_id", how="left")

    path = output_dir / f"district_features{preferred_extension(preferred_format)}"
    saved_path = write_table(district_df, path)
    LOGGER.info("Saved %s district-level feature rows to %s", len(district_df), saved_path)
    return district_df


def save_pca_plot(
    modeling_df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    output_dir: Path,
) -> None:
    feature_matrix = modeling_df[list(feature_columns)].to_numpy(dtype=np.float32)
    if len(modeling_df) < 2:
        LOGGER.warning("Skipping PCA plot because fewer than 2 districts are available.")
        return

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(feature_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        components[:, 0],
        components[:, 1],
        c=modeling_df[target_column],
        cmap="viridis",
        alpha=0.85,
    )
    ax.set_title(f"District Embeddings PCA Colored by {target_column}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(target_column)
    fig.tight_layout()

    plot_path = output_dir / "district_embeddings_pca.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved PCA plot to %s", plot_path)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    corr = np.nan
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(pearsonr(y_true, y_pred).statistic)
    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "pearson_correlation": corr,
    }


def train_and_evaluate_models(
    district_features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    output_dir: Path,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    save_pca: bool = True,
) -> pd.DataFrame:
    modeling_df = district_features_df.merge(
        outcomes_df[["district_id", target_column]],
        on="district_id",
        how="inner",
    ).dropna(subset=[target_column])

    feature_cols = embedding_columns()
    if modeling_df["district_id"].nunique() < 5:
        raise ValueError("Need at least 5 districts after aggregation to train/evaluate models.")

    X = modeling_df[feature_cols].to_numpy(dtype=np.float32)
    y = modeling_df[target_column].to_numpy(dtype=np.float32)
    districts = modeling_df["district_id"].to_numpy()
    image_counts = modeling_df["image_count"].to_numpy()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        district_train,
        district_test,
        image_count_train,
        image_count_test,
    ) = train_test_split(
        X,
        y,
        districts,
        image_counts,
        test_size=test_size,
        random_state=random_state,
    )

    del district_train, image_count_train

    models_to_run = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    metrics_records: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, model in models_to_run.items():
        LOGGER.info("Training %s model", model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        metrics["model"] = model_name
        metrics_records.append(metrics)

        LOGGER.info(
            "%s | MAE=%.4f RMSE=%.4f R2=%.4f Pearson=%.4f",
            model_name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["pearson_correlation"],
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "district_id": district_test,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "image_count": image_count_test,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics_records)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_path = output_dir / "model_metrics.csv"
    predictions_path = output_dir / "model_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved predictions to %s", predictions_path)

    print("\nEvaluation metrics")
    print(metrics_df.to_string(index=False))

    if save_pca:
        save_pca_plot(modeling_df, feature_cols, target_column, output_dir)

    return predictions_df


def run_pipeline(
    image_metadata_path: Path | str,
    district_outcome_path: Path | str,
    output_dir: Path | str,
    image_root: Optional[Path | str] = None,
    metadata_id_column: str = DEFAULT_ID_COLUMN,
    outcome_id_column: str = DEFAULT_ID_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    min_images_per_district: int = 20,
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
    device: Optional[str] = None,
    embeddings_format: str = "parquet",
    force_recompute_embeddings: bool = False,
    save_pca_plot: bool = True,
    auto_generate_metadata: bool = False,
    generated_metadata_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    image_metadata_path = Path(image_metadata_path)
    district_outcome_path = Path(district_outcome_path)
    output_dir = Path(output_dir)
    image_root = Path(image_root) if image_root is not None else None
    generated_metadata_path = (
        Path(generated_metadata_path) if generated_metadata_path is not None else output_dir / "generated_image_metadata.csv"
    )

    setup_logging(output_dir)
    resolved_device = infer_device(device)
    if embeddings_format not in {"parquet", "csv"}:
        raise ValueError("embeddings_format must be either 'parquet' or 'csv'.")

    LOGGER.info("Using device: %s", resolved_device)

    if auto_generate_metadata and not image_metadata_path.exists():
        if image_root is None:
            raise ValueError("image_root is required when auto_generate_metadata is enabled.")
        image_metadata_path = generate_image_metadata_from_directory(
            image_root=image_root,
            output_path=generated_metadata_path,
            metadata_id_column=metadata_id_column,
        )
    elif not image_metadata_path.exists():
        raise FileNotFoundError(
            f"Image metadata file not found: {image_metadata_path}. "
            "Either create it first or enable metadata auto-generation."
        )

    metadata_df, outcomes_df = load_data(
        image_metadata_path=image_metadata_path,
        district_outcome_path=district_outcome_path,
        metadata_id_column=metadata_id_column,
        outcome_id_column=outcome_id_column,
        target_column=target_column,
        min_images_per_district=min_images_per_district,
        image_root=image_root,
    )

    encoder = build_resnet50_encoder(device=resolved_device)
    embeddings_df = generate_image_embeddings(
        metadata_df=metadata_df,
        encoder=encoder,
        output_dir=output_dir,
        device=resolved_device,
        batch_size=batch_size,
        force_recompute=force_recompute_embeddings,
        preferred_format=embeddings_format,
    )
    district_features_df = aggregate_district_features(
        embeddings_df=embeddings_df,
        output_dir=output_dir,
        preferred_format=embeddings_format,
    )
    predictions_df = train_and_evaluate_models(
        district_features_df=district_features_df,
        outcomes_df=outcomes_df,
        output_dir=output_dir,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        save_pca=save_pca_plot,
    )
    return predictions_df


def main() -> pd.DataFrame:
    return run_pipeline(
        image_metadata_path=Config.IMAGE_METADATA_PATH,
        district_outcome_path=Config.DISTRICT_OUTCOME_PATH,
        output_dir=Config.OUTPUT_DIR,
        image_root=Config.IMAGE_ROOT,
        metadata_id_column=Config.METADATA_ID_COLUMN,
        outcome_id_column=Config.OUTCOME_ID_COLUMN,
        target_column=Config.TARGET_COLUMN,
        min_images_per_district=Config.MIN_IMAGES_PER_DISTRICT,
        batch_size=Config.BATCH_SIZE,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        device=Config.DEVICE,
        embeddings_format=Config.EMBEDDINGS_FORMAT,
        force_recompute_embeddings=Config.FORCE_RECOMPUTE_EMBEDDINGS,
        save_pca_plot=Config.SAVE_PCA_PLOT,
        auto_generate_metadata=Config.AUTO_GENERATE_METADATA,
        generated_metadata_path=Config.GENERATED_METADATA_PATH,
    )


if __name__ == "__main__":
    main()
