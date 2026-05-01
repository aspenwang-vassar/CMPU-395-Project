#!/usr/bin/env python3
"""Generate report-ready visualizations for CNN district prediction outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


LOGGER = logging.getLogger("cnn_visualizations")
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {
    "train": "#4C78A8",
    "val": "#F58518",
    "test": "#54A24B",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known column variants used across experiment outputs."""
    df = df.copy()
    rename_map = {}
    variants = {
        "model_name": "model",
        "pearson_correlation": "pearson",
        "sedalea": "district_id",
    }
    for old, new in variants.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    if rename_map:
        df = df.rename(columns=rename_map)
    if "district_id" in df.columns:
        df["district_id"] = (
            df["district_id"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
    return df


def ordered_splits(values: pd.Series) -> list[str]:
    present = [split for split in SPLIT_ORDER if split in set(values.dropna().astype(str))]
    extras = sorted(set(values.dropna().astype(str)) - set(present))
    return present + extras


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def load_metrics(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "model_metrics.csv"
    if not path.exists():
        LOGGER.warning("Metrics file not found: %s", path)
        return pd.DataFrame()
    metrics = normalize_common_columns(pd.read_csv(path))
    required = {"split", "mae", "rmse", "r2", "pearson"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing required metrics columns: {sorted(missing)}")
    if "model" not in metrics.columns:
        metrics["model"] = "model"
    LOGGER.info("Loaded %s metric rows", len(metrics))
    return metrics


def load_predictions(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "district_predictions.csv"
    if not path.exists():
        LOGGER.warning("Predictions file not found: %s", path)
        return pd.DataFrame()
    predictions = normalize_common_columns(pd.read_csv(path))
    required = {"district_id", "y_true", "y_pred", "image_count", "split"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")
    if "model" not in predictions.columns:
        predictions["model"] = "model"
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    predictions["absolute_error"] = predictions["residual"].abs()
    LOGGER.info("Loaded %s prediction rows", len(predictions))
    return predictions


def representative_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Use one row per district for dataset-level summaries."""
    if predictions.empty:
        return predictions
    return predictions.drop_duplicates(subset=["district_id", "split"]).copy()


def plot_split_summary(predictions: pd.DataFrame, output_dir: Path) -> None:
    if predictions.empty:
        return
    district_df = representative_predictions(predictions)
    splits = ordered_splits(district_df["split"])

    district_counts = district_df.groupby("split")["district_id"].nunique().reindex(splits)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(district_counts.index, district_counts.values, color=[SPLIT_COLORS.get(s, "#777777") for s in district_counts.index])
    ax.set_title("Districts by Data Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of districts")
    for i, value in enumerate(district_counts.values):
        ax.text(i, value, f"{int(value)}", ha="center", va="bottom")
    save_figure(fig, output_dir / "split_district_counts.png")

    if "image_count" in district_df.columns:
        image_counts = district_df.groupby("split")["image_count"].sum().reindex(splits)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(image_counts.index, image_counts.values, color=[SPLIT_COLORS.get(s, "#777777") for s in image_counts.index])
        ax.set_title("Images by Data Split")
        ax.set_xlabel("Split")
        ax.set_ylabel("Number of images")
        for i, value in enumerate(image_counts.values):
            ax.text(i, value, f"{int(value):,}", ha="center", va="bottom")
        save_figure(fig, output_dir / "split_image_counts.png")


def plot_image_count_distribution(predictions: pd.DataFrame, output_dir: Path) -> None:
    if predictions.empty or "image_count" not in predictions.columns:
        return
    district_df = representative_predictions(predictions)
    splits = ordered_splits(district_df["split"])
    min_count = int(district_df["image_count"].min())
    max_count = int(district_df["image_count"].max())
    mean_count = float(district_df["image_count"].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(min_count, max_count + 2) - 0.5
    for split in splits:
        subset = district_df[district_df["split"] == split]
        ax.hist(
            subset["image_count"],
            bins=bins,
            alpha=0.65,
            label=split,
            color=SPLIT_COLORS.get(split),
            edgecolor="white",
        )
    ax.axvline(min_count, color="#555555", linestyle="--", linewidth=1, label=f"Min = {min_count}")
    ax.axvline(mean_count, color="#222222", linestyle="-", linewidth=1.5, label=f"Mean = {mean_count:.1f}")
    ax.axvline(max_count, color="#555555", linestyle=":", linewidth=1.5, label=f"Max = {max_count}")
    title = f"Image Count per District ({min_count}-{max_count} images, mean {mean_count:.1f})"
    ax.set_title(title)
    ax.set_xlabel("Images per district")
    ax.set_ylabel("Number of districts")
    ax.legend()
    save_figure(fig, output_dir / "image_count_distribution.png")


def plot_metrics_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    metric_specs = [
        ("mae", "MAE", "Lower is better"),
        ("rmse", "RMSE", "Lower is better"),
        ("r2", "R2", "Higher is better"),
        ("pearson", "Pearson Correlation", "Higher is better"),
    ]
    splits = ordered_splits(metrics["split"])
    models = sorted(metrics["model"].dropna().astype(str).unique())

    for metric_col, label, note in metric_specs:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(splits))
        width = min(0.8 / max(len(models), 1), 0.35)
        for model_idx, model in enumerate(models):
            model_df = metrics[metrics["model"].astype(str) == model].set_index("split")
            values = model_df.reindex(splits)[metric_col].to_numpy()
            offset = (model_idx - (len(models) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=model)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"{label} by Split\n{note}")
        ax.set_xlabel("Split")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        if len(models) > 1 or models[0] != "model":
            ax.legend(title="Model")
        save_figure(fig, output_dir / f"metric_{metric_col}_by_split.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric_col, label, note) in zip(axes.ravel(), metric_specs):
        x = np.arange(len(splits))
        width = min(0.8 / max(len(models), 1), 0.35)
        for model_idx, model in enumerate(models):
            model_df = metrics[metrics["model"].astype(str) == model].set_index("split")
            values = model_df.reindex(splits)[metric_col].to_numpy()
            offset = (model_idx - (len(models) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=model)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel("Split")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles and (len(models) > 1 or models[0] != "model"):
        fig.legend(handles, labels, title="Model", loc="upper center", ncol=min(len(models), 4))
    fig.suptitle("Model Metrics by Split", y=1.02)
    save_figure(fig, output_dir / "model_metrics_comparison.png")


def plot_predicted_vs_true(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    target_name: str,
) -> None:
    if predictions.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    splits = ordered_splits(predictions["split"])
    for split in splits:
        subset = predictions[predictions["split"] == split]
        ax.scatter(
            subset["y_true"],
            subset["y_pred"],
            s=42,
            alpha=0.75,
            label=split,
            color=SPLIT_COLORS.get(split),
            edgecolors="white",
            linewidths=0.5,
        )
    min_value = float(min(predictions["y_true"].min(), predictions["y_pred"].min()))
    max_value = float(max(predictions["y_true"].max(), predictions["y_pred"].max()))
    pad = (max_value - min_value) * 0.05 if max_value > min_value else 0.1
    line_min, line_max = min_value - pad, max_value + pad
    ax.plot([line_min, line_max], [line_min, line_max], color="#333333", linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)

    subtitle = ""
    test_metrics = metrics[metrics["split"].astype(str) == "test"] if not metrics.empty else pd.DataFrame()
    if not test_metrics.empty:
        row = test_metrics.iloc[0]
        subtitle = f"\nTest R2 = {row['r2']:.3f}, Pearson = {row['pearson']:.3f}"
    ax.set_title(f"Predicted vs. True {target_name}{subtitle}")
    ax.set_xlabel(f"True {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.legend()
    save_figure(fig, output_dir / "predicted_vs_true.png")


def plot_residuals(predictions: pd.DataFrame, output_dir: Path, target_name: str) -> None:
    if predictions.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ordered_splits(predictions["split"]):
        subset = predictions[predictions["split"] == split]
        ax.scatter(
            subset["y_pred"],
            subset["residual"],
            s=40,
            alpha=0.75,
            label=split,
            color=SPLIT_COLORS.get(split),
            edgecolors="white",
            linewidths=0.5,
        )
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_title("Residuals by Predicted Value")
    ax.set_xlabel(f"Predicted {target_name}")
    ax.set_ylabel("Residual (true - predicted)")
    ax.legend()
    save_figure(fig, output_dir / "residual_plot.png")


def plot_prediction_bias(predictions: pd.DataFrame, output_dir: Path, target_name: str) -> None:
    if predictions.empty:
        return
    splits = ordered_splits(predictions["split"])
    bias_df = predictions.groupby("split").agg(mean_true=("y_true", "mean"), mean_pred=("y_pred", "mean")).reindex(splits)
    x = np.arange(len(splits))
    width = 0.36

    title_note = ""
    if "test" in bias_df.index:
        test_true = bias_df.loc["test", "mean_true"]
        test_pred = bias_df.loc["test", "mean_pred"]
        direction = "higher" if test_pred > test_true else "lower"
        title_note = f"\nTest mean prediction is {direction} than test mean true value"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, bias_df["mean_true"], width=width, label="Mean true", color="#4C78A8")
    ax.bar(x + width / 2, bias_df["mean_pred"], width=width, label="Mean predicted", color="#F58518")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Prediction Bias by Split{title_note}")
    ax.set_xlabel("Split")
    ax.set_ylabel(f"Mean {target_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    save_figure(fig, output_dir / "prediction_bias_by_split.png")


def plot_training_curves(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    path = input_dir / "mil_cnn_training_curve.csv"
    if not path.exists():
        LOGGER.warning("Training curve file not found: %s", path)
        return pd.DataFrame()
    curve = pd.read_csv(path)
    if "epoch" not in curve.columns:
        LOGGER.warning("Training curve has no epoch column; skipping.")
        return pd.DataFrame()

    metric_specs = [
        ("train_loss", "Train Loss", "min"),
        ("val_mae", "Validation MAE", "min"),
        ("val_rmse", "Validation RMSE", "min"),
        ("val_r2", "Validation R2", "max"),
        ("val_pearson", "Validation Pearson", "max"),
    ]
    available_specs = [(col, label, mode) for col, label, mode in metric_specs if col in curve.columns]

    for col, label, mode in available_specs:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(curve["epoch"], curve[col], marker="o", linewidth=1.8, color="#4C78A8")
        best_idx = curve[col].idxmin() if mode == "min" else curve[col].idxmax()
        best = curve.loc[best_idx]
        ax.scatter([best["epoch"]], [best[col]], s=90, color="#E45756", zorder=3, label=f"Best epoch {int(best['epoch'])}")
        ax.set_title(f"{label} over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        save_figure(fig, output_dir / f"training_{col}.png")

    if available_specs:
        rows = int(np.ceil(len(available_specs) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, (col, label, mode) in zip(axes_flat, available_specs):
            ax.plot(curve["epoch"], curve[col], marker="o", linewidth=1.6, color="#4C78A8")
            best_idx = curve[col].idxmin() if mode == "min" else curve[col].idxmax()
            best = curve.loc[best_idx]
            ax.scatter([best["epoch"]], [best[col]], s=70, color="#E45756", zorder=3)
            ax.set_title(f"{label} (best epoch {int(best['epoch'])})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
        for ax in axes_flat[len(available_specs):]:
            ax.axis("off")
        fig.suptitle("MIL CNN Training Curves", y=1.01)
        save_figure(fig, output_dir / "training_curves_combined.png")
    return curve


def plot_pca_if_available(input_dir: Path, predictions: pd.DataFrame, output_dir: Path) -> None:
    path = input_dir / "district_features.parquet"
    if not path.exists():
        LOGGER.warning("District feature file not found; skipping PCA: %s", path)
        return
    features = normalize_common_columns(pd.read_parquet(path))
    if "district_id" not in features.columns:
        LOGGER.warning("District features have no district_id column; skipping PCA.")
        return

    meta_cols = {"district_id", "split", "model", "y_true", "y_pred", "image_count", "residual", "absolute_error"}
    feature_cols = [
        col for col in features.columns
        if col not in meta_cols and pd.api.types.is_numeric_dtype(features[col])
    ]
    if len(feature_cols) < 2:
        LOGGER.warning("Need at least two numeric feature columns for PCA; found %s.", len(feature_cols))
        return

    merged = features[["district_id", *feature_cols]].copy()
    if not predictions.empty:
        pred_meta = predictions.drop_duplicates("district_id")[["district_id", "y_true", "split"]]
        merged = merged.merge(pred_meta, on="district_id", how="left")

    matrix = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    components = PCA(n_components=2, random_state=42).fit_transform(matrix)
    merged["pc1"] = components[:, 0]
    merged["pc2"] = components[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    splits = ordered_splits(merged["split"]) if "split" in merged.columns and merged["split"].notna().any() else ["all"]
    markers = {"train": "o", "val": "s", "test": "^", "all": "o"}
    if "y_true" in merged.columns and merged["y_true"].notna().any():
        for split in splits:
            subset = merged if split == "all" else merged[merged["split"] == split]
            scatter = ax.scatter(
                subset["pc1"],
                subset["pc2"],
                c=subset["y_true"],
                cmap="viridis",
                marker=markers.get(split, "o"),
                s=48,
                alpha=0.82,
                label=split,
                edgecolors="white",
                linewidths=0.4,
            )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("True outcome")
    else:
        ax.scatter(merged["pc1"], merged["pc2"], s=45, alpha=0.8, color="#4C78A8")
    ax.set_title("PCA of District-Level CNN Features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if len(splits) > 1:
        ax.legend(title="Split")
    save_figure(fig, output_dir / "district_features_pca.png")


def plot_error_vs_image_count(predictions: pd.DataFrame, output_dir: Path) -> None:
    if predictions.empty or "image_count" not in predictions.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ordered_splits(predictions["split"]):
        subset = predictions[predictions["split"] == split]
        ax.scatter(
            subset["image_count"],
            subset["absolute_error"],
            s=42,
            alpha=0.75,
            label=split,
            color=SPLIT_COLORS.get(split),
            edgecolors="white",
            linewidths=0.5,
        )

    valid = predictions[["image_count", "absolute_error"]].dropna()
    if len(valid) >= 2 and valid["image_count"].nunique() > 1:
        slope, intercept = np.polyfit(valid["image_count"], valid["absolute_error"], 1)
        xs = np.linspace(valid["image_count"].min(), valid["image_count"].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#333333", linestyle="--", linewidth=1.4, label="Linear trend")
    ax.set_title("Prediction Error vs. Number of Images per District")
    ax.set_xlabel("Images per district")
    ax.set_ylabel("Absolute error")
    ax.legend()
    save_figure(fig, output_dir / "error_vs_image_count.png")


def save_top_error_table(predictions: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    columns = ["district_id", "split", "y_true", "y_pred", "residual", "absolute_error", "image_count"]
    if "model" in predictions.columns:
        columns.insert(1, "model")
    top_errors = predictions[columns].sort_values("absolute_error", ascending=False).reset_index(drop=True)
    path = input_dir / "top_prediction_errors.csv"
    top_errors.to_csv(path, index=False)
    LOGGER.info("Saved %s", path)
    return top_errors


def markdown_table(df: pd.DataFrame, max_rows: int = 10, include_index: bool = False) -> str:
    if df.empty:
        return "_No rows available._"
    table_df = df.head(max_rows).copy()
    if include_index:
        table_df = table_df.reset_index()
    formatted = table_df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: f"{value:.3f}")
        else:
            formatted[col] = formatted[col].astype(str)

    headers = [str(col) for col in formatted.columns]
    rows = formatted.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_visualization_summary(
    input_dir: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    curve: pd.DataFrame,
    top_errors: pd.DataFrame,
    target_name: str,
) -> None:
    district_df = representative_predictions(predictions)
    total_districts = int(district_df["district_id"].nunique()) if not district_df.empty else 0
    total_images = int(district_df["image_count"].sum()) if not district_df.empty and "image_count" in district_df else 0
    split_summary = (
        district_df.groupby("split")
        .agg(n_districts=("district_id", "nunique"), n_images=("image_count", "sum"), mean_images=("image_count", "mean"))
        .reindex(ordered_splits(district_df["split"]))
        if not district_df.empty else pd.DataFrame()
    )

    lines = [
        "# CNN District Prediction Visualization Summary",
        "",
        "## Dataset",
        "",
        f"- Total districts: **{total_districts:,}**",
        f"- Total images: **{total_images:,}**",
        "- Splits are district-level, so images from the same district are not mixed across train/validation/test splits.",
    ]
    if total_districts == 250 and total_images == 7033:
        lines.append("- This matches the expected rough-draft dataset size of **250 districts** and **7,033 images**.")
    if not split_summary.empty:
        lines.extend(["", "### Split Summary", "", markdown_table(split_summary, include_index=True)])

    if not metrics.empty:
        lines.extend(["", "## Model Metrics", "", markdown_table(metrics)])
        val_row = metrics[metrics["split"].astype(str) == "val"]
        test_row = metrics[metrics["split"].astype(str) == "test"]
        if not val_row.empty:
            row = val_row.iloc[0]
            if row["r2"] > 0 and row["pearson"] > 0:
                lines.append(
                    f"- Validation performance is moderately promising: val R2 = **{row['r2']:.3f}** and val Pearson = **{row['pearson']:.3f}**."
                )
        if not test_row.empty:
            row = test_row.iloc[0]
            if row["r2"] < 0 or row["pearson"] < 0.2:
                lines.append(
                    f"- Test performance is weak: test R2 = **{row['r2']:.3f}** and test Pearson = **{row['pearson']:.3f}**."
                )

    if not predictions.empty and "test" in set(predictions["split"].astype(str)):
        test_pred = predictions[predictions["split"].astype(str) == "test"]
        test_mean_true = float(test_pred["y_true"].mean())
        test_mean_pred = float(test_pred["y_pred"].mean())
        test_mae = float(test_pred["absolute_error"].mean())
        bias_direction = "higher than" if test_mean_pred > test_mean_true else "lower than"
        lines.extend(
            [
                "",
                "## Prediction Bias",
                "",
                f"- Test mean true {target_name}: **{test_mean_true:.3f}**",
                f"- Test mean predicted {target_name}: **{test_mean_pred:.3f}**",
                f"- Test MAE: **{test_mae:.3f}**",
                f"- The test mean prediction is **{bias_direction}** the test mean true value.",
            ]
        )

    if not curve.empty:
        lines.extend(["", "## Training Curve Highlights", ""])
        for col, label, mode in [
            ("val_mae", "Validation MAE", "min"),
            ("val_rmse", "Validation RMSE", "min"),
            ("val_r2", "Validation R2", "max"),
            ("val_pearson", "Validation Pearson", "max"),
        ]:
            if col in curve.columns:
                idx = curve[col].idxmin() if mode == "min" else curve[col].idxmax()
                row = curve.loc[idx]
                lines.append(f"- Best {label}: **{row[col]:.3f}** at epoch **{int(row['epoch'])}**")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The CNN currently does not generalize reliably to held-out districts. The validation split shows some signal, but the test split indicates weak out-of-sample performance. More tuning, stronger regularization, alternative aggregation, more districts, or multimodal features may be needed.",
            "",
            "## Largest Prediction Errors",
            "",
            markdown_table(top_errors[["district_id", "split", "y_true", "y_pred", "residual", "absolute_error", "image_count"]] if not top_errors.empty else top_errors),
            "",
            "## Generated Figures",
            "",
            "- `plots/split_district_counts.png`",
            "- `plots/split_image_counts.png`",
            "- `plots/image_count_distribution.png`",
            "- `plots/model_metrics_comparison.png`",
            "- `plots/predicted_vs_true.png`",
            "- `plots/residual_plot.png`",
            "- `plots/prediction_bias_by_split.png`",
            "- `plots/training_curves_combined.png`",
            "- `plots/district_features_pca.png` if district features are available",
            "- `plots/error_vs_image_count.png`",
        ]
    )

    path = input_dir / "visualization_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CNN district prediction visualizations.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/cnn_district_prediction"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cnn_district_prediction/plots"))
    parser.add_argument("--target-name", default="SEDA achievement score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Input directory: %s", args.input_dir)
    LOGGER.info("Output directory: %s", args.output_dir)

    metrics = load_metrics(args.input_dir)
    predictions = load_predictions(args.input_dir)
    plot_split_summary(predictions, args.output_dir)
    plot_image_count_distribution(predictions, args.output_dir)
    plot_metrics_comparison(metrics, args.output_dir)
    plot_predicted_vs_true(predictions, metrics, args.output_dir, args.target_name)
    plot_residuals(predictions, args.output_dir, args.target_name)
    plot_prediction_bias(predictions, args.output_dir, args.target_name)
    curve = plot_training_curves(args.input_dir, args.output_dir)
    plot_pca_if_available(args.input_dir, predictions, args.output_dir)
    plot_error_vs_image_count(predictions, args.output_dir)
    top_errors = save_top_error_table(predictions, args.input_dir)
    write_visualization_summary(args.input_dir, metrics, predictions, curve, top_errors, args.target_name)

    LOGGER.info("Visualization generation complete.")


if __name__ == "__main__":
    main()
