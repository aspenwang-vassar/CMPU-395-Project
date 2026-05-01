# CNN District Prediction Visualization Summary

## Dataset

- Total districts: **250**
- Total images: **7,033**
- Splits are district-level, so images from the same district are not mixed across train/validation/test splits.
- This matches the expected rough-draft dataset size of **250 districts** and **7,033 images**.

### Split Summary

| split | n_districts | n_images | mean_images |
| --- | --- | --- | --- |
| train | 175 | 4940 | 28.229 |
| val | 37 | 1045 | 28.243 |
| test | 38 | 1048 | 27.579 |

## Model Metrics

| model | split | mae | rmse | r2 | pearson | n_districts |
| --- | --- | --- | --- | --- | --- | --- |
| mil_cnn | train | 0.355 | 0.434 | -0.256 | 0.218 | 175 |
| mil_cnn | val | 0.306 | 0.392 | 0.127 | 0.453 | 37 |
| mil_cnn | test | 0.435 | 0.533 | -0.452 | 0.125 | 38 |
- Validation performance is moderately promising: val R2 = **0.127** and val Pearson = **0.453**.
- Test performance is weak: test R2 = **-0.452** and test Pearson = **0.125**.

## Prediction Bias

- Test mean true SEDA achievement score: **0.073**
- Test mean predicted SEDA achievement score: **0.244**
- Test MAE: **0.435**
- The test mean prediction is **higher than** the test mean true value.

## Training Curve Highlights

- Best Validation MAE: **0.300** at epoch **7**
- Best Validation RMSE: **0.392** at epoch **5**
- Best Validation R2: **0.127** at epoch **5**
- Best Validation Pearson: **0.488** at epoch **10**

## Interpretation

The CNN currently does not generalize reliably to held-out districts. The validation split shows some signal, but the test split indicates weak out-of-sample performance. More tuning, stronger regularization, alternative aggregation, more districts, or multimodal features may be needed.

## Largest Prediction Errors

| district_id | split | y_true | y_pred | residual | absolute_error | image_count |
| --- | --- | --- | --- | --- | --- | --- |
| 2709510 | test | -0.580 | 0.610 | -1.190 | 1.190 | 40 |
| 2632280 | train | -0.526 | 0.549 | -1.075 | 1.075 | 30 |
| 2200240 | test | -0.939 | 0.119 | -1.057 | 1.057 | 19 |
| 4211700 | train | -0.569 | 0.483 | -1.053 | 1.053 | 23 |
| 3904936 | train | 0.871 | -0.151 | 1.022 | 1.022 | 19 |
| 3904505 | test | -0.294 | 0.688 | -0.982 | 0.982 | 40 |
| 601421 | train | -0.574 | 0.408 | -0.982 | 0.982 | 18 |
| 3415030 | test | -0.328 | 0.641 | -0.969 | 0.969 | 40 |
| 3404770 | val | 1.328 | 0.397 | 0.931 | 0.931 | 30 |
| 4110560 | train | 1.150 | 0.225 | 0.925 | 0.925 | 40 |

## Generated Figures

- `plots/split_district_counts.png`
- `plots/split_image_counts.png`
- `plots/image_count_distribution.png`
- `plots/model_metrics_comparison.png`
- `plots/predicted_vs_true.png`
- `plots/residual_plot.png`
- `plots/prediction_bias_by_split.png`
- `plots/training_curves_combined.png`
- `plots/district_features_pca.png` if district features are available
- `plots/error_vs_image_count.png`
