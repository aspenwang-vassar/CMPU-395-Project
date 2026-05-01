"""
multimodal_model.py

Multimodal regression combining ResNet-50 district-level CNN embeddings
with RoBERTa-based Reddit sentiment scores to predict SEDA achievement.

Outputs:
  - multimodal_results.csv   : per-district predictions and true values
  - multimodal_metrics.csv   : MAE, RMSE, R2, Pearson for each modality & split
  - modality_comparison.png  : bar chart comparing unimodal vs multimodal performance
  - multimodal_scatter.png   : predicted vs true scatter for test set (multimodal)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# ── 1. Load data ─────────────────────────────────────────────────────────────

embeddings = pd.read_parquet("district_features.parquet")
sentiment  = pd.read_csv("reddit_sentiment.csv")

# Normalise district_id types so we can merge
embeddings["district_id"] = embeddings["district_id"].astype(str).str.strip()
sentiment["sedalea"]      = sentiment["sedalea"].astype(str).str.strip()

# ── 2. Merge ──────────────────────────────────────────────────────────────────

merged = embeddings.merge(
    sentiment[["sedalea", "cs_mn_avg_eb", "mean_sentiment", "post_count", "std_sentiment"]],
    left_on="district_id",
    right_on="sedalea",
    how="inner"
).dropna(subset=["cs_mn_avg_eb", "mean_sentiment"])

print(f"Districts after merge: {len(merged)}")

# ── 3. Feature matrices ───────────────────────────────────────────────────────

emb_cols = [c for c in merged.columns if c.startswith("emb_")]
X_cnn  = merged[emb_cols].values
X_nlp  = merged[["mean_sentiment"]].values
X_both = np.hstack([X_cnn, X_nlp])
y      = merged["cs_mn_avg_eb"].values

# ── 4. Train / val / test split (district-level, 70/15/15) ────────────────────

np.random.seed(42)
n = len(merged)
idx = np.random.permutation(n)

n_train = int(0.70 * n)
n_val   = int(0.15 * n)

train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train + n_val]
test_idx  = idx[n_train + n_val:]

def split(X):
    return X[train_idx], X[val_idx], X[test_idx]

X_cnn_tr,  X_cnn_va,  X_cnn_te  = split(X_cnn)
X_nlp_tr,  X_nlp_va,  X_nlp_te  = split(X_nlp)
X_both_tr, X_both_va, X_both_te = split(X_both)
y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]

print(f"Split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# ── 5. Scale + fit ────────────────────────────────────────────────────────────

def fit_ridge(X_tr, X_va, X_te, y_tr):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    alphas  = np.logspace(-2, 4, 50)
    model   = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_tr_s, y_tr)
    return model, scaler, X_tr_s, X_va_s, X_te_s

def metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return {"model": label, "MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": r}

results = []

for name, (X_tr, X_va, X_te) in [
    ("CNN only",        (X_cnn_tr,  X_cnn_va,  X_cnn_te)),
    ("NLP only",        (X_nlp_tr,  X_nlp_va,  X_nlp_te)),
    ("Multimodal",      (X_both_tr, X_both_va, X_both_te)),
]:
    model, scaler, Xtr_s, Xva_s, Xte_s = fit_ridge(X_tr, X_va, X_te, y_tr)

    for split_name, Xs, ys in [("Train", Xtr_s, y_tr), ("Val", Xva_s, y_va), ("Test", Xte_s, y_te)]:
        preds = model.predict(Xs)
        row   = metrics(ys, preds, name)
        row["split"] = split_name
        results.append(row)

    # Save test predictions for multimodal scatter
    if name == "Multimodal":
        test_preds    = model.predict(Xte_s)
        test_district = merged.iloc[test_idx]["district_id"].values

metrics_df = pd.DataFrame(results)[["model", "split", "MAE", "RMSE", "R2", "Pearson"]]
metrics_df = metrics_df.round(4)
print("\n", metrics_df.to_string(index=False))
metrics_df.to_csv("multimodal_metrics.csv", index=False)

# ── 6. Save per-district test predictions ─────────────────────────────────────

pred_df = pd.DataFrame({
    "district_id": test_district,
    "y_true":      y_te,
    "y_pred_multimodal": test_preds
})
pred_df.to_csv("multimodal_results.csv", index=False)

# ── 7. Bar chart: Test-set Pearson by modality ────────────────────────────────

test_metrics = metrics_df[metrics_df["split"] == "Test"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
colors = ["#4C72B0", "#55A868", "#C44E52"]

for ax, metric in zip(axes, ["Pearson", "R2"]):
    vals   = test_metrics[metric].values
    models = test_metrics["model"].values
    bars   = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Test {metric} by Modality", fontsize=13)
    ax.set_ylabel(metric)
    ax.set_ylim(min(vals.min() - 0.1, -0.1), vals.max() + 0.15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

plt.suptitle("Unimodal vs. Multimodal — Test Set Performance", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("modality_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 8. Scatter: multimodal predicted vs true (test) ──────────────────────────

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_te, test_preds, alpha=0.7, edgecolors="white", linewidth=0.5, color="#C44E52")
lims = [min(y_te.min(), test_preds.min()) - 0.05,
        max(y_te.max(), test_preds.max()) + 0.05]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("True SEDA Score")
ax.set_ylabel("Predicted SEDA Score")
ax.set_title("Multimodal Model: Predicted vs. True (Test Set)")
ax.legend()
plt.tight_layout()
plt.savefig("multimodal_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nDone. Outputs: multimodal_metrics.csv, multimodal_results.csv,")
print("               modality_comparison.png, multimodal_scatter.png")
