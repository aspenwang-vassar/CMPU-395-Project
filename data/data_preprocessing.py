"""
Data Preprocessing Pipeline for School District Analysis
=======================================================

This pipeline preprocesses and samples the school district information for downstream analysis

Workflow:
1. Load SEDA academic outcome data
2. Load school district shapefiles
3. Merge data on district ID
4. Sample districts with fixed seed
5. Save preprocessed data for downstream analysis

Output:
- sampled_districts.csv: List of sampled districts with metadata
- preprocessed_data.geojson: GeoJSON with sampled district boundaries

Author: Aspen Wang
"""

import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
from pathlib import Path
import traceback

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for data preprocessing"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    SEDA_CSV = DATA_DIR / "seda_geodist_annual_cs_6.0.csv"
    SHAPEFILE = DATA_DIR / "EDGE_SCHOOLDISTRICT_TL25_SY2425.shp"
    
    # Output paths
    SAMPLED_DISTRICTS_CSV = DATA_DIR / "out/sampled_districts.csv"
    PREPROCESSED_GEOJSON = DATA_DIR / "out/preprocessed_data.geojson"
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Data configuration
    TARGET_YEAR = 2019
    DISTRICT_ID_COL = "sedalea"
    DISTRICT_ID_COL = "sedalea"
    DISTRICT_NAME_COL = "sedaleaname"
    SUBGROUP_COL = "subgroup"
    AGGREGATE_SUBGROUP = "all"
    
    # Outcome columns
    OUTCOME_COLS = [
        "cs_mn_avg_mth_eb",
        "cs_mn_avg_rla_eb"
    ]
    OUTCOME_SE_COLS = [
        "cs_mn_avg_mth_eb_se_adj",
        "cs_mn_avg_rla_eb_se_adj"
    ]
    
    # Sampling
    SAMPLE_SIZE = None  # None = use all districts, or specify a number for random sample
    
    # Logging
    LOG_FILE = PROJECT_ROOT / "data_preprocessing.log"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for preprocessing"""
    logger = logging.getLogger("DataPreprocessing")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
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
# DATA LOADING
# ============================================================================

def load_seda_data(filepath: Path) -> pd.DataFrame:
    """
    Load SEDA CSV and filter to target year and aggregate subgroup.
    
    SEDA data structure: seda_geodist_annualsub_cs_6.0
    - One row per geographic district-subgroup-year combination
    - Multiple rows per district if subgroup data exists
    """
    logger.info(f"Loading SEDA data from {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"SEDA CSV not found at {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} total SEDA rows")
    
    # Filter to target year
    if "year" in df.columns:
        df = df[df["year"] == Config.TARGET_YEAR]
        logger.info(f"Filtered to year {Config.TARGET_YEAR}: {len(df):,} rows")
    else:
        logger.warning("'year' column not found in SEDA data")
    
    # Filter to aggregate subgroup
    if Config.AGGREGATE_SUBGROUP and Config.SUBGROUP_COL in df.columns:
        initial_count = len(df)
        df = df[df[Config.SUBGROUP_COL] == Config.AGGREGATE_SUBGROUP]
        logger.info(
            f"Filtered to subgroup '{Config.AGGREGATE_SUBGROUP}': "
            f"{len(df):,}/{initial_count:,} rows"
        )
    
    # Validate outcome columns
    missing_cols = [col for col in Config.OUTCOME_COLS if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing outcome columns: {missing_cols}")
        logger.info(f"Available columns: {df.columns.tolist()}")
    
    logger.info(f"SEDA dataset: {len(df):,} districts for year {Config.TARGET_YEAR}")
    return df


def resolve_outcome_columns(df: pd.DataFrame) -> None:
    """Select outcome columns that actually exist in the loaded SEDA extract."""
    available_outcome_cols = [col for col in Config.OUTCOME_COLS if col in df.columns]
    available_se_cols = [col for col in Config.OUTCOME_SE_COLS if col in df.columns]

    if available_outcome_cols:
        Config.OUTCOME_COLS = available_outcome_cols
        Config.OUTCOME_SE_COLS = available_se_cols
        logger.info(f"Using configured outcome columns: {Config.OUTCOME_COLS}")
        return

    fallback_outcomes = ["cs_mn_avg_eb"]
    fallback_se = ["cs_mn_avg_eb_se_adj"]
    fallback_outcomes = [col for col in fallback_outcomes if col in df.columns]
    fallback_se = [col for col in fallback_se if col in df.columns]

    if not fallback_outcomes:
        raise ValueError(
            "No usable outcome columns found in SEDA data. "
            f"Available columns: {df.columns.tolist()}"
        )

    Config.OUTCOME_COLS = fallback_outcomes
    Config.OUTCOME_SE_COLS = fallback_se
    logger.warning(
        "Configured outcome columns were unavailable; "
        f"falling back to {Config.OUTCOME_COLS}"
    )
def prune_seda_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only essential columns for modeling and drop the rest.
    """
    logger.info("Pruning unnecessary SEDA columns...")

    # Always keep these if available
    keep_cols = [
        "sedalea",
        "sedaleaname",
        "year",
        "stateabb",

        # Outcome (will adjust below)
        "cs_mn_avg_eb",
        "cs_mn_avg_eb_se",

        # Reliability
        "tot_asmts",
        "cellcount",
        "mn_asmts",
        "flag_estasmt"
    ]

    # If using math/RLA separately, keep those instead
    for col in [
        "cs_mn_avg_mth_eb", "cs_mn_avg_rla_eb",
        "cs_mn_avg_mth_eb_se", "cs_mn_avg_rla_eb_se"
    ]:
        if col in df.columns:
            keep_cols.append(col)

    # Only keep columns that actually exist
    keep_cols = [col for col in keep_cols if col in df.columns]

    logger.info(f"Keeping {len(keep_cols)} columns: {keep_cols}")

    df = df[keep_cols].copy()

    return df
def load_shapefile(filepath: Path) -> gpd.GeoDataFrame:
    """
    Load school district shapefile (Tiger Line Files).
    
    Expected columns:
    - GEOID: Geographic identifier (7 digits = state FIPS + district code)
    - geometry: District boundaries
    """
    logger.info(f"Loading shapefile from {filepath}")
    
    if not filepath.exists():
        # Try with different extension
        base = str(filepath).replace('.shp', '')
        if Path(f"{base}.shp").exists():
            filepath = Path(f"{base}.shp")
        else:
            raise FileNotFoundError(f"Shapefile not found at {filepath}")
    
    sidecar_suffixes = [".shx", ".dbf", ".prj"]
    missing_sidecars = [
        f"{filepath.stem}{suffix}"
        for suffix in sidecar_suffixes
        if not filepath.with_suffix(suffix).exists()
    ]
    if missing_sidecars:
        logger.warning(f"Missing shapefile sidecar files: {missing_sidecars}")

    with fiona.Env(SHAPE_RESTORE_SHX="YES"):
        gdf = gpd.read_file(filepath)
    logger.info(f"Loaded {len(gdf):,} districts from shapefile")
    logger.info(f"Shapefile columns: {gdf.columns.tolist()}")

    if list(gdf.columns) == ["geometry"]:
        raise ValueError(
            "Shapefile attributes are missing. The boundary file loaded only "
            "geometry, which usually means the .dbf file is absent or unreadable. "
            f"Expected sidecar files alongside {filepath.name}, especially "
            f"{filepath.with_suffix('.dbf').name}, so district IDs like GEOID/LEAID "
            "are available for merging."
        )
    
    # Ensure CRS is WGS84
    if gdf.crs is None:
        logger.warning("CRS not set, defaulting to EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs != "EPSG:4326":
        logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    
    # Handle GEOID column naming
    if "GEOID" in gdf.columns and Config.DISTRICT_ID_COL not in gdf.columns:
        logger.info(f"Renaming GEOID to {Config.DISTRICT_ID_COL}")
        gdf = gdf.rename(columns={"GEOID": Config.DISTRICT_ID_COL})
        gdf[Config.DISTRICT_ID_COL] = pd.to_numeric(
            gdf[Config.DISTRICT_ID_COL], 
            errors="coerce"
        )
    
    logger.info(f"Processed {len(gdf):,} districts from shapefile")
    return gdf


def merge_data(seda_df: pd.DataFrame, shape_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge SEDA data with shapefile on district ID.
    
    Handles ID format matching and validation.
    """
    logger.info("Merging SEDA data with shapefile...")
    
    seda_id_col = Config.DISTRICT_ID_COL
    
    # Find district ID column in shapefile
    shape_id_col = None
    for col in shape_gdf.columns:
        if col.upper() in ["SEDALEA", "GEOID", "LEAID", "DISTRICT_ID"]:
            shape_id_col = col
            break
    
    if shape_id_col is None:
        raise ValueError(
            f"Could not find district ID column in shapefile. "
            f"Available columns: {shape_gdf.columns.tolist()}"
        )
    
    logger.info(f"Merging on: SEDA '{seda_id_col}' ← Shapefile '{shape_id_col}'")
    
    # Ensure both are int type
    seda_df[seda_id_col] = pd.to_numeric(seda_df[seda_id_col], errors="coerce")
    shape_gdf[shape_id_col] = pd.to_numeric(shape_gdf[shape_id_col], errors="coerce")
    
    # Perform merge
    merged = shape_gdf.merge(
        seda_df,
        left_on=shape_id_col,
        right_on=seda_id_col,
        how="inner"
    )
    
    logger.info(f"After merge: {len(merged):,} records")
    
    # Filter to rows with valid outcome data
    missing_outcomes = merged[Config.OUTCOME_COLS].isnull()
    n_missing = missing_outcomes.any(axis=1).sum()
    if n_missing > 0:
        logger.info(f"Dropping {n_missing:,} rows with missing outcomes")
        merged = merged.dropna(subset=Config.OUTCOME_COLS)
    
    if len(merged) == 0:
        raise ValueError(
            "Merge resulted in 0 rows. Check that district IDs match between "
            "SEDA and shapefile."
        )
    
    logger.info(f"Final merged dataset: {len(merged):,} valid districts")
    return merged


# ============================================================================
# SAMPLING
# ============================================================================

def sample_districts(
    merged_gdf: gpd.GeoDataFrame,
    random_seed: int = Config.RANDOM_SEED
) -> gpd.GeoDataFrame:
    """
    Sample districts with fixed random seed for reproducibility.
    
    Args:
        merged_gdf: GeoDataFrame with merged SEDA and shapefile data
        random_seed: Random seed for reproducibility
    
    Returns:
        GeoDataFrame with sampled districts
    """
    np.random.seed(random_seed)
    
    total_districts = len(merged_gdf)
    sample_size = Config.SAMPLE_SIZE or total_districts
    
    logger.info(f"Sampling configuration:")
    logger.info(f"  Total districts: {total_districts:,}")
    logger.info(f"  Sample size: {sample_size:,}")
    logger.info(f"  Random seed: {random_seed}")
    
    if sample_size >= total_districts:
        logger.info("Sample size >= total districts, using all districts")
        sampled = merged_gdf.copy()
    else:
        logger.info(f"Sampling {sample_size:,} districts with seed {random_seed}")
        sampled = merged_gdf.sample(n=sample_size, random_state=random_seed)
    
    logger.info(f"Sampled dataset: {len(sampled):,} districts")
    return sampled


# ============================================================================
# SAVE PREPROCESSED DATA
# ============================================================================

def save_preprocessed_data(
    sampled_gdf: gpd.GeoDataFrame,
    random_seed: int = Config.RANDOM_SEED
) -> None:
    """
    Save preprocessed data outputs.
    """
    logger.info("\nSaving preprocessed data...")
    
    # Save sampled districts as CSV (without geometry for portability)
    sampled_df = pd.DataFrame(sampled_gdf.drop(columns=["geometry"]))
    sampled_df.to_csv(Config.SAMPLED_DISTRICTS_CSV, index=False)
    logger.info(f"Saved sampled districts to {Config.SAMPLED_DISTRICTS_CSV}")
    
    # Save GeoJSON with geometry for mapping
    sampled_gdf.to_file(Config.PREPROCESSED_GEOJSON, driver="GeoJSON")
    logger.info(f"Saved GeoJSON to {Config.PREPROCESSED_GEOJSON}")


# ============================================================================
# MAIN PREPROCESSING
# ============================================================================

def run_preprocessing():
    """Execute complete data preprocessing pipeline"""
    logger.info("="*80)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("="*80)
    
    try:
        # Step 1: Load SEDA data
        logger.info("\n[STEP 1] Loading SEDA data...")
        seda_df = load_seda_data(Config.SEDA_CSV)
        resolve_outcome_columns(seda_df)
        seda_df = prune_seda_columns(seda_df)
        
        # Step 2: Load shapefile
        logger.info("\n[STEP 2] Loading shapefile...")
        shape_gdf = load_shapefile(Config.SHAPEFILE)
        
        # Step 3: Merge data
        logger.info("\n[STEP 3] Merging data...")
        merged_gdf = merge_data(seda_df, shape_gdf)
        
        # Step 4: Sample districts
        logger.info("\n[STEP 4] Sampling districts...")
        sampled_gdf = sample_districts(merged_gdf, Config.RANDOM_SEED)
        
        # Step 5: Save preprocessed data
        logger.info("\n[STEP 5] Saving preprocessed data...")
        save_preprocessed_data(sampled_gdf, Config.RANDOM_SEED)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*80)
        logger.info(f"Random Seed: {Config.RANDOM_SEED}")
        logger.info(f"Districts Sampled: {len(sampled_gdf):,}")
        logger.info(f"Output Files:")
        logger.info(f"  - {Config.SAMPLED_DISTRICTS_CSV}")
        logger.info(f"  - {Config.PREPROCESSED_GEOJSON}")
        
        logger.info("\n" + "="*80)
        logger.info("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return sampled_gdf
    
    except Exception as e:
        logger.error(f"\nFATAL ERROR: {e}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess school district data with fixed random seed"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {Config.RANDOM_SEED})"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of districts to sample (default: all)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=Config.TARGET_YEAR,
        help=f"Target year (default: {Config.TARGET_YEAR})"
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.seed is not None:
        Config.RANDOM_SEED = args.seed
    if args.sample_size is not None:
        Config.SAMPLE_SIZE = args.sample_size
    if args.year is not None:
        Config.TARGET_YEAR = args.year
    
    # Run preprocessing
    sampled_gdf = run_preprocessing()
