# CMPU-395-Project

Street View and district-level educational outcome analysis pipeline.

## District-Level Prediction Pipeline

`district_quality_pipeline.py` builds a resumable prediction workflow that:

- loads Street View image metadata and district outcomes
- extracts pretrained ResNet-50 embeddings per image
- mean-pools embeddings to the district level
- trains Ridge and Random Forest baselines on district-level targets
- saves cached embeddings, district features, predictions, metrics, and a PCA plot

Install dependencies from `requirements.txt`, then edit the `Config` values near the top of [district_quality_pipeline.py](/Users/aspen/PycharmProjects/CMPU-395-Project/district_quality_pipeline.py) and run the file directly:

```bash
python3 district_quality_pipeline.py
```

The default script-style settings include:

- `Config.IMAGE_METADATA_PATH`
- `Config.AUTO_GENERATE_METADATA`
- `Config.GENERATED_METADATA_PATH`
- `Config.DISTRICT_OUTCOME_PATH`
- `Config.IMAGE_ROOT`
- `Config.TARGET_COLUMN`
- `Config.MIN_IMAGES_PER_DISTRICT`
- `Config.DEVICE`

If you do not already have an image metadata CSV, set `Config.AUTO_GENERATE_METADATA = True` and the pipeline will scan `Config.IMAGE_ROOT`, parse district IDs from filenames like `3304040.0_2125.jpg`, and save a generated metadata file before training.

A Colab-ready notebook with zip upload, extraction, and automatic repo-root detection is available at [colab_district_quality_pipeline.ipynb](/Users/aspen/PycharmProjects/CMPU-395-Project/colab_district_quality_pipeline.ipynb).
