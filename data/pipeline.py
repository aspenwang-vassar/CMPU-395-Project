"""
Google Street View Pipeline for School District Analysis
=========================================================

A modular pipeline that:
1. Loads SEDA outcomes & school district shapefiles
2. Samples points within districts
3. Queries Street View metadata API
4. Downloads Street View images


Author: Data Science Pipeline
"""

import os
import sys
import json
import time
import requests
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime
from urllib.parse import urlencode
import traceback

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

from shapely.geometry import Point, MultiPolygon, Polygon

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

class Config:
    """Central configuration for the pipeline"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Preprocessed data (output from data_preprocessing.py)
    SAMPLED_DISTRICTS_CSV = DATA_DIR / "out/sampled_districts.csv"
    PREPROCESSED_GEOJSON = DATA_DIR / "out/preprocessed_data.geojson"
    
    # Original data sources (used by data_preprocessing.py)
    SEDA_CSV = DATA_DIR / "seda_geodist_annualsub_cs_6.0.csv"  # SEDA district-level data
    SHAPEFILE = DATA_DIR / "schooldistricttl25_tidy.shp"  # Tiger Line school districts
    
    # Output files
    IMAGES_DIR = DATA_DIR / "streetview_images"
    METADATA_CSV = DATA_DIR / "street_view_metadata.csv"
    VALID_POINTS_CSV = DATA_DIR / "valid_points.csv"
    IMAGE_DOWNLOAD_PROGRESS_CSV = DATA_DIR / "image_download_progress.csv"
    DISTRICT_COVERAGE_CSV = DATA_DIR / "district_image_coverage.csv"
    COVERAGE_REPORT_JSON = DATA_DIR / "district_image_coverage.json"
    DOWNLOADED_IMAGES_CSV = DATA_DIR / "downloaded_images.csv"
    EMBEDDINGS_CSV = DATA_DIR / "district_embeddings.csv"
    MERGED_DATASET_CSV = DATA_DIR / "final_dataset.csv"
    
    # API
    GOOGLE_API_KEY = os.getenv("GOOGLE_STREET_VIEW_API_KEY", "YOUR_API_KEY_HERE")
    
    # Parameters
    POINTS_PER_DISTRICT = 80
    MAX_PANORAMAS_PER_DISTRICT = 40
    MIN_IMAGES_PER_DISTRICT = 5
    STREETVIEW_IMAGE_SIZE = (640, 640)
    API_RETRY_ATTEMPTS = 3
    API_RETRY_DELAY = 1.0  # seconds
    REQUEST_DELAY = 0.5    # seconds between requests
    
    # Data
    TARGET_YEAR = 2019
    DISTRICT_ID_COL = "sedalea"  # SEDA LEA ID
    DISTRICT_NAME_COL = "sedaleaname"  # Geographic District Name
    LAT_COL = "latitude"
    LON_COL = "longitude"
    SUBGROUP_COL = "subgroup"  # Subgroup identifier
    AGGREGATE_SUBGROUP = "all"  # Filter to aggregate (all groups) or specific subgroup
    OUTCOME_COLS = [
        "cs_mn_avg_mth_eb",  # Math outcomes
        "cs_mn_avg_rla_eb"   # Reading/ELA outcomes
    ]
    OUTCOME_SE_COLS = [
        "cs_mn_avg_mth_eb_se_adj",  # Adjusted SE for math
        "cs_mn_avg_rla_eb_se_adj"   # Adjusted SE for ELA
    ]
    
    # Model
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Logging
    LOG_FILE = PROJECT_ROOT / "pipeline.log"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for the pipeline"""
    logger = logging.getLogger("SVPipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(Config.LOG_FILE)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()


def ensure_fiona_compatibility() -> None:
    """Patch Fiona module attributes expected by older GeoPandas releases."""
    if not hasattr(fiona, "path"):
        import fiona.path as fiona_path
        fiona.path = fiona_path


ensure_fiona_compatibility()

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_preprocessed_data() -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Load preprocessed district data from data_preprocessing.py output.
    
    Prerequisites:
    - Run data_preprocessing.py first to generate sampled districts
    
    Returns:
        - GeoDataFrame: District boundaries with SEDA outcomes
        - DataFrame: Sampled districts data (without geometry for reference)
    """
    logger.info("Loading preprocessed data...")
    
    # Check if preprocessed files exist
    if not Config.SAMPLED_DISTRICTS_CSV.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {Config.SAMPLED_DISTRICTS_CSV}\n"
            f"Please run: python data_preprocessing.py"
        )
    
    # Load GeoDataFrame with geometry
    logger.info(f"Loading GeoJSON from {Config.PREPROCESSED_GEOJSON}")
    gdf = gpd.read_file(Config.PREPROCESSED_GEOJSON)
    logger.info(f"Loaded {len(gdf):,} sampled districts")
    
    # Load CSV version for reference
    logger.info(f"Loading sampled districts CSV from {Config.SAMPLED_DISTRICTS_CSV}")
    df = pd.read_csv(Config.SAMPLED_DISTRICTS_CSV)
    logger.info(f"Verified {len(df):,} districts in CSV")
    
    # Ensure CRS is correct
    if gdf.crs != "EPSG:4326":
        logger.info(f"Converting CRS from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    
    logger.info(f"Preprocessed data loaded: {len(gdf):,} districts")
    return gdf, df


def resolve_outcome_columns(df: pd.DataFrame) -> None:
    """Select outcome columns that actually exist in the preprocessed data."""
    available_outcome_cols = [col for col in Config.OUTCOME_COLS if col in df.columns]
    available_se_cols = [col for col in Config.OUTCOME_SE_COLS if col in df.columns]

    if available_outcome_cols:
        Config.OUTCOME_COLS = available_outcome_cols
        Config.OUTCOME_SE_COLS = available_se_cols
        logger.info(f"Using outcome columns: {Config.OUTCOME_COLS}")
        return

    fallback_outcomes = ["cs_mn_avg_eb"]
    fallback_se = ["cs_mn_avg_eb_se_adj"]
    fallback_outcomes = [col for col in fallback_outcomes if col in df.columns]
    fallback_se = [col for col in fallback_se if col in df.columns]

    if not fallback_outcomes:
        raise ValueError(
            "No usable outcome columns found in preprocessed data. "
            f"Available columns: {df.columns.tolist()}"
        )

    Config.OUTCOME_COLS = fallback_outcomes
    Config.OUTCOME_SE_COLS = fallback_se
    logger.warning(
        "Configured outcome columns were unavailable; "
        f"falling back to {Config.OUTCOME_COLS}"
    )


# ============================================================================
# 2. POINT SAMPLING
# ============================================================================

def get_largest_polygon(geometry):
    """Extract largest polygon from MultiPolygon, or return Polygon as-is"""
    if isinstance(geometry, MultiPolygon):
        largest = max(geometry.geoms, key=lambda p: p.area)
        return largest
    return geometry


def sample_points_in_polygon(polygon: Polygon, n_points: int, seed: int = None) -> List[Point]:
    """
    Sample n random points inside a polygon using rejection sampling
    """
    np.random.seed(seed)
    
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    
    while len(points) < n_points:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = Point(x, y)
        
        if polygon.contains(point):
            points.append(point)
    
    return points


def sample_all_districts(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Sample points for all districts"""
    logger.info(f"Sampling {Config.POINTS_PER_DISTRICT} points per district")
    
    all_points = []
    
    for idx, row in gdf.iterrows():
        geometry = get_largest_polygon(row.geometry)
        
        try:
            points = sample_points_in_polygon(
                geometry,
                Config.POINTS_PER_DISTRICT,
                seed=idx
            )
            
            for i, point in enumerate(points):
                all_points.append({
                    Config.DISTRICT_ID_COL: row[Config.DISTRICT_ID_COL],
                    "point_id": i,
                    Config.LAT_COL: point.y,
                    Config.LON_COL: point.x
                })
        except Exception as e:
            logger.warning(f"Failed to sample points for district {row[Config.DISTRICT_ID_COL]}: {e}")
    
    points_df = pd.DataFrame(all_points)
    logger.info(f"Sampled {len(points_df)} total points")
    return points_df


# ============================================================================
# 3. STREET VIEW METADATA API
# ============================================================================

def call_streetview_metadata(lat: float, lon: float, api_key: str) -> Dict:
    """
    Query Google Street View metadata endpoint
    
    Returns:
        Dict with status, pano_id, pano_lat, pano_lon, date
    """
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "key": api_key
    }
    
    for attempt in range(Config.API_RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            result = {
                "status": data.get("status", "UNKNOWN"),
                "pano_id": data.get("pano_id"),
                "pano_lat": data.get("location", {}).get("lat"),
                "pano_lon": data.get("location", {}).get("lng"),
                "date": data.get("date"),
                "error_message": data.get("error_message")
            }
            
            time.sleep(Config.REQUEST_DELAY)
            return result
            
        except Exception as e:
            logger.warning(f"Metadata API error (attempt {attempt+1}): {e}")
            if attempt < Config.API_RETRY_ATTEMPTS - 1:
                time.sleep(Config.API_RETRY_DELAY)
            else:
                return {
                    "status": "ERROR",
                    "pano_id": None,
                    "pano_lat": None,
                    "pano_lon": None,
                    "date": None,
                    "error_message": str(e)
                }


def validate_streetview_api(points_df: pd.DataFrame, api_key: str) -> None:
    """Fail fast if the Street View metadata API is not usable."""
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Google Street View API key is not configured. Set "
            "GOOGLE_STREET_VIEW_API_KEY before running pipeline.py."
        )

    if points_df.empty:
        raise ValueError("No sampled points available for Street View API validation.")

    sample_point = points_df.iloc[0]
    logger.info("Running Street View API preflight check...")
    metadata = call_streetview_metadata(
        sample_point[Config.LAT_COL],
        sample_point[Config.LON_COL],
        api_key
    )

    if metadata["status"] == "REQUEST_DENIED":
        error_message = metadata.get("error_message") or "No error message returned."
        raise ValueError(
            "Street View API preflight failed with REQUEST_DENIED. "
            f"Google response: {error_message}"
        )

    if metadata["status"] == "ERROR":
        error_message = metadata.get("error_message") or "Unknown request error."
        raise ValueError(
            "Street View API preflight failed before metadata collection. "
            f"Error: {error_message}"
        )

    logger.info(
        f"Street View API preflight status: {metadata['status']}"
    )


def get_metadata_for_all_points(points_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Fetch Street View metadata for all sampled points"""
    logger.info(f"Fetching metadata for {len(points_df)} points")

    validate_streetview_api(points_df, api_key)
    
    metadata_list = []
    
    for idx, row in points_df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing point {idx}/{len(points_df)}")
        
        metadata = call_streetview_metadata(
            row[Config.LAT_COL],
            row[Config.LON_COL],
            api_key
        )
        
        metadata_dict = row.to_dict()
        metadata_dict.update(metadata)
        metadata_list.append(metadata_dict)
    
    metadata_df = pd.DataFrame(metadata_list)
    logger.info(f"Retrieved metadata for {len(metadata_df)} points")
    
    # Save intermediate CSV
    metadata_df.to_csv(Config.METADATA_CSV, index=False)
    logger.info(f"Saved metadata to {Config.METADATA_CSV}")
    
    return metadata_df


# ============================================================================
# 4. SPATIAL FILTERING
# ============================================================================

def filter_valid_panoramas(
    metadata_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Filter metadata to keep only valid, in-district panoramas"""
    logger.info(f"Filtering metadata from {len(metadata_df)} rows")
    
    # Filter by status
    valid = metadata_df[metadata_df["status"] == "OK"].copy()
    logger.info(f"After status filter: {len(valid)} rows")
    
    # Filter by location existence
    valid = valid.dropna(subset=["pano_lat", "pano_lon"])
    logger.info(f"After location filter: {len(valid)} rows")
    
    # Spatial filter: check if pano location is in district polygon
    valid_filtered = []
    
    for idx, row in valid.iterrows():
        pano_point = Point(row["pano_lon"], row["pano_lat"])
        district_id = row[Config.DISTRICT_ID_COL]
        
        # Get district polygon
        district = gdf[gdf[Config.DISTRICT_ID_COL] == district_id]
        if len(district) == 0:
            continue
        
        geometry = get_largest_polygon(district.iloc[0].geometry)
        
        # Check if point is in polygon
        if geometry.contains(pano_point):
            valid_filtered.append(row)
    
    valid_df = pd.DataFrame(valid_filtered, columns=valid.columns)
    logger.info(f"After spatial filter: {len(valid_df)} rows")
    
    # Save intermediate CSV
    valid_df.to_csv(Config.VALID_POINTS_CSV, index=False)
    logger.info(f"Saved valid points to {Config.VALID_POINTS_CSV}")
    
    return valid_df


# ============================================================================
# 5. DOWNSAMPLING
# ============================================================================

def downsample_by_district(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Keep up to MAX_PANORAMAS_PER_DISTRICT per district"""
    logger.info(f"Downsampling to {Config.MAX_PANORAMAS_PER_DISTRICT} per district")

    if valid_df.empty:
        logger.warning("No valid panoramas available after filtering; skipping downsampling.")
        return valid_df.copy()
    
    downsampled = valid_df.groupby(Config.DISTRICT_ID_COL).apply(
        lambda x: x.sample(
            n=min(Config.MAX_PANORAMAS_PER_DISTRICT, len(x)),
            random_state=Config.RANDOM_STATE
        )
    ).reset_index(drop=True)
    
    logger.info(f"After downsampling: {len(downsampled)} rows")
    return downsampled


# ============================================================================
# 6. COVERAGE TRACKING AND IMAGE DOWNLOADS
# ============================================================================

def summarize_district_coverage(
    candidate_df: pd.DataFrame,
    images_df: Optional[pd.DataFrame] = None,
    district_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Summarize image coverage by district."""
    if candidate_df.empty:
        return pd.DataFrame(
            columns=[
                Config.DISTRICT_ID_COL,
                Config.DISTRICT_NAME_COL,
                "stateabb",
                "candidate_panoramas",
                "download_attempts",
                "successful_images",
                "threshold_met",
            ]
        )

    candidate_counts = (
        candidate_df.groupby(Config.DISTRICT_ID_COL)
        .size()
        .rename("candidate_panoramas")
        .reset_index()
    )

    if images_df is None or images_df.empty:
        successful_counts = pd.DataFrame(
            {
                Config.DISTRICT_ID_COL: candidate_counts[Config.DISTRICT_ID_COL],
                "successful_images": 0,
            }
        )
        attempted_counts = pd.DataFrame(
            {
                Config.DISTRICT_ID_COL: candidate_counts[Config.DISTRICT_ID_COL],
                "download_attempts": 0,
            }
        )
    else:
        attempted_counts = (
            images_df.groupby(Config.DISTRICT_ID_COL)["download_attempted"]
            .sum()
            .rename("download_attempts")
            .reset_index()
        )
        successful_counts = (
            images_df.groupby(Config.DISTRICT_ID_COL)["download_success"]
            .sum()
            .rename("successful_images")
            .reset_index()
        )

    coverage_df = candidate_counts.merge(
        attempted_counts,
        on=Config.DISTRICT_ID_COL,
        how="left"
    ).merge(
        successful_counts,
        on=Config.DISTRICT_ID_COL,
        how="left"
    )

    coverage_df["download_attempts"] = coverage_df["download_attempts"].fillna(0).astype(int)
    coverage_df["successful_images"] = coverage_df["successful_images"].fillna(0).astype(int)
    coverage_df["threshold_met"] = (
        coverage_df["successful_images"] >= Config.MIN_IMAGES_PER_DISTRICT
    )

    if district_df is not None and not district_df.empty:
        district_cols = [Config.DISTRICT_ID_COL]
        if Config.DISTRICT_NAME_COL in district_df.columns:
            district_cols.append(Config.DISTRICT_NAME_COL)
        if "stateabb" in district_df.columns:
            district_cols.append("stateabb")

        district_metadata = district_df[district_cols].drop_duplicates(
            subset=[Config.DISTRICT_ID_COL],
            keep="first"
        )
        coverage_df = coverage_df.merge(
            district_metadata,
            on=Config.DISTRICT_ID_COL,
            how="left"
        )

        ordered_cols = [Config.DISTRICT_ID_COL]
        if Config.DISTRICT_NAME_COL in coverage_df.columns:
            ordered_cols.append(Config.DISTRICT_NAME_COL)
        if "stateabb" in coverage_df.columns:
            ordered_cols.append("stateabb")
        ordered_cols.extend([
            "candidate_panoramas",
            "download_attempts",
            "successful_images",
            "threshold_met",
        ])
        coverage_df = coverage_df[ordered_cols]

    return coverage_df.sort_values(Config.DISTRICT_ID_COL).reset_index(drop=True)


def write_coverage_outputs(
    coverage_df: pd.DataFrame,
    output_csv: Path,
    output_json: Optional[Path] = None
) -> None:
    """Persist per-district coverage outputs."""
    coverage_df.to_csv(output_csv, index=False)

    if output_json is not None:
        report = {
            "timestamp": datetime.now().isoformat(),
            "min_images_per_district": Config.MIN_IMAGES_PER_DISTRICT,
            "districts_total": int(len(coverage_df)),
            "districts_meeting_threshold": int(coverage_df["threshold_met"].sum()) if not coverage_df.empty else 0,
            "districts_below_threshold": int((~coverage_df["threshold_met"]).sum()) if not coverage_df.empty else 0,
            "successful_images_total": int(coverage_df["successful_images"].sum()) if not coverage_df.empty else 0,
            "candidate_panoramas_total": int(coverage_df["candidate_panoramas"].sum()) if not coverage_df.empty else 0,
        }
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)


def filter_districts_by_min_images(
    images_df: pd.DataFrame,
    coverage_df: pd.DataFrame
) -> pd.DataFrame:
    """Keep only districts that meet the minimum image threshold."""
    eligible_district_ids = coverage_df.loc[
        coverage_df["threshold_met"],
        Config.DISTRICT_ID_COL
    ]

    filtered_df = images_df[
        images_df[Config.DISTRICT_ID_COL].isin(eligible_district_ids)
    ].copy()
    filtered_df = filtered_df.dropna(subset=["image_path"])

    logger.info(
        f"Districts meeting minimum image threshold ({Config.MIN_IMAGES_PER_DISTRICT}): "
        f"{len(eligible_district_ids):,}/{len(coverage_df):,}"
    )
    logger.info(
        f"Images retained after threshold filter: {len(filtered_df):,}/{int(images_df['download_success'].sum()):,}"
    )

    if filtered_df.empty:
        raise ValueError(
            "No districts met the minimum image threshold of "
            f"{Config.MIN_IMAGES_PER_DISTRICT}."
        )

    return filtered_df


def log_coverage_summary(coverage_df: pd.DataFrame) -> None:
    """Log aggregate district image coverage stats."""
    if coverage_df.empty:
        logger.warning("No district coverage statistics available.")
        return

    logger.info("District image coverage summary:")
    logger.info(f"  Districts total: {len(coverage_df):,}")
    logger.info(f"  Districts meeting threshold: {int(coverage_df['threshold_met'].sum()):,}")
    logger.info(f"  Districts below threshold: {int((~coverage_df['threshold_met']).sum()):,}")
    logger.info(f"  Successful images total: {int(coverage_df['successful_images'].sum()):,}")
    logger.info(
        "  Successful images per district "
        f"(min/median/max): {coverage_df['successful_images'].min()}/"
        f"{int(coverage_df['successful_images'].median())}/"
        f"{coverage_df['successful_images'].max()}"
    )

def download_streetview_image(
    lat: float,
    lon: float,
    api_key: str,
    size: Tuple[int, int] = Config.STREETVIEW_IMAGE_SIZE
) -> Optional[Image.Image]:
    """Download Street View image"""
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": f"{lat},{lon}",
        "size": f"{size[0]}x{size[1]}",
        "key": api_key
    }
    
    for attempt in range(Config.API_RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            time.sleep(Config.REQUEST_DELAY)
            return img
            
        except Exception as e:
            logger.warning(f"Image download error (attempt {attempt+1}): {e}")
            if attempt < Config.API_RETRY_ATTEMPTS - 1:
                time.sleep(Config.API_RETRY_DELAY)
    
    return None


def download_all_images(valid_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Download images for all valid points"""
    logger.info(f"Downloading {len(valid_df)} images")

    if valid_df.empty:
        logger.warning("No valid panoramas to download.")
        valid_df = valid_df.copy()
        valid_df["image_path"] = pd.Series(dtype="object")
        valid_df["download_attempted"] = pd.Series(dtype="int")
        valid_df["download_success"] = pd.Series(dtype="int")
        return valid_df
    
    Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    valid_df = valid_df.copy()
    valid_df["image_path"] = None
    valid_df["download_attempted"] = 0
    valid_df["download_success"] = 0

    initial_coverage = summarize_district_coverage(valid_df, district_df=valid_df)
    write_coverage_outputs(initial_coverage, Config.IMAGE_DOWNLOAD_PROGRESS_CSV)
    
    for idx, row in valid_df.iterrows():
        if idx % 50 == 0:
            logger.info(f"Downloaded {idx}/{len(valid_df)} images")
        
        valid_df.at[idx, "download_attempted"] = 1
        img = download_streetview_image(
            row["pano_lat"],
            row["pano_lon"],
            api_key
        )
        
        if img is not None:
            district_id = row[Config.DISTRICT_ID_COL]
            filename = f"{district_id}_{idx}.jpg"
            filepath = Config.IMAGES_DIR / filename
            
            img.save(filepath)
            valid_df.at[idx, "image_path"] = str(filepath)
            valid_df.at[idx, "download_success"] = 1
        else:
            valid_df.at[idx, "image_path"] = None

        progress_df = summarize_district_coverage(valid_df, valid_df, district_df=valid_df)
        write_coverage_outputs(progress_df, Config.IMAGE_DOWNLOAD_PROGRESS_CSV)
    
    success_count = int(valid_df["download_success"].sum())
    logger.info(f"Successfully downloaded {success_count} images")
    
    return valid_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """Execute the Street View imagery dataset generation pipeline."""
    logger.info("="*80)
    logger.info("STARTING GOOGLE STREET VIEW IMAGERY DATASET PIPELINE")
    logger.info("="*80)
    
    try:
        # Step 0: Load preprocessed data (already sampled)
        logger.info("\n[STEP 0] Loading preprocessed data...")
        merged_gdf, sampled_df = load_preprocessed_data()
        resolve_outcome_columns(merged_gdf)
        
        logger.info(f"Using {len(merged_gdf):,} sampled districts")
        logger.info("Note: Districts were sampled in data_preprocessing.py")
        
        # Step 1: Sample points
        logger.info("\n[STEP 1] Sampling points in districts...")
        points_df = sample_all_districts(merged_gdf)
        
        # Step 2: Get Street View metadata
        logger.info("\n[STEP 2] Fetching Street View metadata...")
        metadata_df = get_metadata_for_all_points(points_df, Config.GOOGLE_API_KEY)
        
        # Step 3: Spatial filtering
        logger.info("\n[STEP 3] Filtering valid panoramas...")
        valid_df = filter_valid_panoramas(metadata_df, merged_gdf)
        if valid_df.empty:
            status_counts = metadata_df["status"].value_counts(dropna=False).to_dict()
            raise ValueError(
                "Street View metadata returned no valid in-district panoramas for "
                f"the sampled points. Status counts: {status_counts}"
            )
        
        # Step 4: Downsample
        logger.info("\n[STEP 4] Downsampling by district...")
        downsampled_df = downsample_by_district(valid_df)
        
        # Step 5: Download images
        logger.info("\n[STEP 5] Downloading images...")
        images_df = download_all_images(downsampled_df, Config.GOOGLE_API_KEY)
        coverage_df = summarize_district_coverage(
            downsampled_df,
            images_df,
            district_df=merged_gdf
        )
        write_coverage_outputs(
            coverage_df,
            Config.DISTRICT_COVERAGE_CSV,
            Config.COVERAGE_REPORT_JSON
        )
        log_coverage_summary(coverage_df)
        images_df = filter_districts_by_min_images(images_df, coverage_df)
        
        images_df.to_csv(Config.DOWNLOADED_IMAGES_CSV, index=False)
        logger.info(f"Saved downloaded image manifest to {Config.DOWNLOADED_IMAGES_CSV}")
        
        logger.info("\n" + "="*80)
        logger.info("IMAGE DOWNLOAD PIPELINE COMPLETED")
        logger.info("="*80)
        logger.info(f"Districts retained after thresholding: {images_df[Config.DISTRICT_ID_COL].nunique():,}")
        logger.info(f"Downloaded images retained: {len(images_df):,}")
        logger.info("Run feature_extraction_pipeline.py for embeddings and final dataset creation.")
        
        return images_df
    
    except Exception as e:
        logger.error(f"\nFATAL ERROR: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    downloaded_images = run_pipeline()
