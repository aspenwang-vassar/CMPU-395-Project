"""
Expand the Google Street View school district image dataset.

This script is designed to be resumable and conservative:
- existing images and metadata are loaded, never deleted
- candidate points are sampled with random, grid, and optional road-aware methods
- Street View metadata is queried before image download
- returned panorama coordinates are spatially validated against district polygons
- outputs are flushed after every district so interrupted runs can resume
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled with a clearer runtime error
    pd = None

try:
    import requests
except ImportError:  # pragma: no cover - handled with a clearer runtime error
    requests = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - handled with a clearer runtime error
    gpd = None

try:
    import fiona
except ImportError:  # pragma: no cover
    fiona = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

try:
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
    from shapely.prepared import prep
except ImportError:  # pragma: no cover - handled with a clearer runtime error
    LineString = MultiLineString = MultiPolygon = Point = Polygon = None
    prep = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_collection"
IMAGE_DIR = DATA_DIR / "streetview_images"

DEFAULT_BOUNDARIES = DATA_DIR / "preprocessed_data.geojson"
DEFAULT_SEDA = DATA_DIR / "sampled_districts.csv"
LEGACY_DOWNLOADED = DATA_DIR / "downloaded_images.csv"
LEGACY_GENERATED_METADATA = DATA_DIR / "generated_image_metadata.csv"
LEGACY_STREETVIEW_METADATA = DATA_DIR / "street_view_metadata.csv"
LEGACY_COVERAGE = DATA_DIR / "district_image_coverage.csv"

DOWNLOADED_IMAGES = OUTPUT_DIR / "downloaded_images.csv"
METADATA_CACHE = OUTPUT_DIR / "metadata_cache.csv"
REJECTED_CANDIDATES = OUTPUT_DIR / "rejected_candidates.csv"
DISTRICT_COVERAGE = OUTPUT_DIR / "district_image_coverage.csv"
ANALYSIS_READY = OUTPUT_DIR / "analysis_ready_districts.csv"

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"

ID_COL = "sedalea"
NAME_COL = "sedaleaname"
OUTCOME_CANDIDATES = [
    "cs_mn_avg_eb",
    "gcs_mn_avg_mth_eb",
    "gcs_mn_avg_rla_eb",
    "cs_mn_avg_mth_eb",
    "cs_mn_avg_rla_eb",
]

METADATA_COLUMNS = [
    "query_lat",
    "query_lon",
    "status",
    "pano_id",
    "pano_lat",
    "pano_lon",
    "date",
    "district_id",
    "district_name",
    "stateabb",
    "source",
    "cache_key",
    "queried_at",
    "error_message",
]

DOWNLOADED_COLUMNS = [
    "district_id",
    "district_name",
    "stateabb",
    "pano_id",
    "pano_lat",
    "pano_lon",
    "heading",
    "image_path",
    "download_success",
    "downloaded_at",
]

REJECTED_COLUMNS = [
    "district_id",
    "district_name",
    "stateabb",
    "query_lat",
    "query_lon",
    "status",
    "pano_id",
    "pano_lat",
    "pano_lon",
    "reason",
    "source",
]

COVERAGE_COLUMNS = [
    "district_id",
    "district_name",
    "stateabb",
    "num_valid_images",
    "num_unique_panos",
    "num_candidate_points",
    "num_metadata_ok",
    "num_zero_results",
    "num_outside_district",
    "num_duplicates",
    "included_for_analysis",
]


@dataclass
class ExistingState:
    downloaded: pd.DataFrame
    metadata_cache: pd.DataFrame
    rejected: pd.DataFrame
    coverage: pd.DataFrame
    existing_paths: Set[str]
    existing_district_panos: Set[Tuple[str, str]]


def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("expand_streetview_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(OUTPUT_DIR / "expand_streetview_dataset.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


LOGGER = setup_logging()


def ensure_fiona_compatibility() -> None:
    """Patch Fiona module attributes expected by older GeoPandas releases."""
    if fiona is not None and not hasattr(fiona, "path"):
        import fiona.path as fiona_path

        fiona.path = fiona_path


def normalize_district_id(value: object) -> str:
    """Normalize district IDs so 100013 and 100013.0 compare as the same district."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def sanitize_filename_part(value: object) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(columns)]


def read_csv_if_exists(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if path.exists():
        return ensure_columns(pd.read_csv(path), columns)
    return pd.DataFrame(columns=columns)


def load_existing_metadata() -> ExistingState:
    """Load current and legacy metadata so new collection is purely incremental."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_parts: List[pd.DataFrame] = []
    for path in [DOWNLOADED_IMAGES, LEGACY_DOWNLOADED]:
        if path.exists():
            downloaded_parts.append(pd.read_csv(path))

    if LEGACY_GENERATED_METADATA.exists():
        generated = pd.read_csv(LEGACY_GENERATED_METADATA)
        if "sedalea" in generated.columns and "image_path" in generated.columns:
            generated = generated.rename(columns={"sedalea": "district_id"})
            generated["district_name"] = pd.NA
            generated["stateabb"] = pd.NA
            generated["pano_id"] = pd.NA
            generated["pano_lat"] = pd.NA
            generated["pano_lon"] = pd.NA
            generated["heading"] = 0
            generated["download_success"] = generated["image_path"].notna()
            generated["downloaded_at"] = pd.NA
            downloaded_parts.append(generated)

    downloaded = (
        pd.concat(downloaded_parts, ignore_index=True, sort=False)
        if downloaded_parts
        else pd.DataFrame(columns=DOWNLOADED_COLUMNS)
    )
    if "sedalea" in downloaded.columns and "district_id" not in downloaded.columns:
        downloaded = downloaded.rename(columns={"sedalea": "district_id"})
    downloaded = ensure_columns(downloaded, DOWNLOADED_COLUMNS)
    downloaded["district_id"] = downloaded["district_id"].map(normalize_district_id)
    downloaded["heading"] = pd.to_numeric(downloaded["heading"], errors="coerce").fillna(0).astype(int)
    downloaded = downloaded.drop_duplicates(subset=["image_path"], keep="first")

    metadata_parts: List[pd.DataFrame] = []
    if METADATA_CACHE.exists():
        metadata_parts.append(pd.read_csv(METADATA_CACHE))
    if LEGACY_STREETVIEW_METADATA.exists():
        legacy = pd.read_csv(LEGACY_STREETVIEW_METADATA)
        legacy = legacy.rename(
            columns={
                "sedalea": "district_id",
                "latitude": "query_lat",
                "longitude": "query_lon",
            }
        )
        legacy["district_name"] = pd.NA
        legacy["stateabb"] = pd.NA
        legacy["source"] = "legacy_street_view_metadata"
        legacy["cache_key"] = legacy.apply(
            lambda r: coordinate_cache_key(r.get("query_lat"), r.get("query_lon")), axis=1
        )
        legacy["queried_at"] = pd.NA
        metadata_parts.append(legacy)

    metadata_cache = (
        pd.concat(metadata_parts, ignore_index=True, sort=False)
        if metadata_parts
        else pd.DataFrame(columns=METADATA_COLUMNS)
    )
    metadata_cache = ensure_columns(metadata_cache, METADATA_COLUMNS)
    metadata_cache["district_id"] = metadata_cache["district_id"].map(normalize_district_id)
    metadata_cache = metadata_cache.drop_duplicates(subset=["cache_key"], keep="first")

    rejected = read_csv_if_exists(REJECTED_CANDIDATES, REJECTED_COLUMNS)
    rejected["district_id"] = rejected["district_id"].map(normalize_district_id)
    coverage = read_csv_if_exists(DISTRICT_COVERAGE, COVERAGE_COLUMNS)
    coverage["district_id"] = coverage["district_id"].map(normalize_district_id)

    existing_paths = {
        str(Path(path)).lower()
        for path in downloaded["image_path"].dropna().astype(str)
        if path.strip()
    }
    existing_paths.update(str(path).lower() for path in IMAGE_DIR.glob("*.jpg"))
    existing_district_panos = {
        (row["district_id"], str(row["pano_id"]))
        for _, row in downloaded.dropna(subset=["pano_id"]).iterrows()
    }

    return ExistingState(
        downloaded=downloaded,
        metadata_cache=metadata_cache,
        rejected=rejected,
        coverage=coverage,
        existing_paths=existing_paths,
        existing_district_panos=existing_district_panos,
    )


def load_district_boundaries(
    boundaries_path: Path,
    seda_path: Optional[Path],
    sample_size: int,
    target_districts: int,
    state_filter: Optional[str],
    random_seed: int,
) -> gpd.GeoDataFrame:
    """Load district geometries and keep districts with usable outcome labels."""
    if gpd is None or Point is None:
        raise RuntimeError(
            "Missing GIS dependencies. Install project requirements, including geopandas and shapely."
        )

    ensure_fiona_compatibility()

    if not boundaries_path.exists():
        raise FileNotFoundError(f"District boundary file not found: {boundaries_path}")

    gdf = gpd.read_file(boundaries_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    gdf[ID_COL] = gdf[ID_COL].map(normalize_district_id)

    if seda_path and seda_path.exists():
        seda = pd.read_csv(seda_path)
        seda[ID_COL] = seda[ID_COL].map(normalize_district_id)
        merge_cols = [c for c in [ID_COL, NAME_COL, "stateabb", "year", *OUTCOME_CANDIDATES] if c in seda.columns]
        gdf = gdf.drop(columns=[c for c in merge_cols if c in gdf.columns and c != ID_COL], errors="ignore")
        gdf = gdf.merge(seda[merge_cols].drop_duplicates(ID_COL), on=ID_COL, how="left")

    if state_filter:
        states = {s.strip().upper() for s in state_filter.split(",") if s.strip()}
        gdf = gdf[gdf.get("stateabb", "").astype(str).str.upper().isin(states)].copy()

    outcome_cols = [c for c in OUTCOME_CANDIDATES if c in gdf.columns]
    if outcome_cols:
        gdf = gdf[gdf[outcome_cols].notna().any(axis=1)].copy()
    else:
        LOGGER.warning("No expected outcome columns were found; districts will not be label-filtered.")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    keep_n = min(len(gdf), max(sample_size, target_districts))
    return gdf.head(keep_n).copy()


def largest_polygon(geometry) -> Polygon:
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda poly: poly.area)
    return geometry


def sample_random_points(polygon: Polygon, n_points: int, rng: random.Random) -> List[Point]:
    """Randomly sample inside the polygon bounds with bounded rejection attempts."""
    minx, miny, maxx, maxy = polygon.bounds
    points: List[Point] = []
    attempts = 0
    max_attempts = max(n_points * 100, 1000)

    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
        point = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if polygon.covers(point):
            points.append(point)
    return points


def sample_grid_points(polygon: Polygon, n_points: int) -> List[Point]:
    """Sample stratified grid cell centers, with centroid fallback per intersecting cell."""
    if n_points <= 0:
        return []

    minx, miny, maxx, maxy = polygon.bounds
    side = max(1, math.ceil(math.sqrt(n_points * 1.8)))
    dx = (maxx - minx) / side if maxx > minx else 0
    dy = (maxy - miny) / side if maxy > miny else 0
    points: List[Point] = []

    for ix in range(side):
        for iy in range(side):
            if len(points) >= n_points:
                return points
            center = Point(minx + (ix + 0.5) * dx, miny + (iy + 0.5) * dy)
            if polygon.covers(center):
                points.append(center)
                continue

            cell = Polygon(
                [
                    (minx + ix * dx, miny + iy * dy),
                    (minx + (ix + 1) * dx, miny + iy * dy),
                    (minx + (ix + 1) * dx, miny + (iy + 1) * dy),
                    (minx + ix * dx, miny + (iy + 1) * dy),
                ]
            )
            intersection = polygon.intersection(cell)
            if not intersection.is_empty:
                candidate = intersection.representative_point()
                if polygon.covers(candidate):
                    points.append(candidate)

    return points


def _points_along_line(line: LineString, spacing: float) -> Iterable[Point]:
    if line.length <= 0:
        return []
    count = max(1, int(line.length / spacing))
    return [line.interpolate(i / count, normalized=True) for i in range(count + 1)]


def sample_road_points_if_available(polygon: Polygon, n_points: int, logger: logging.Logger) -> List[Point]:
    """Use OSMnx road geometries when installed; fall back silently on failure."""
    if n_points <= 0:
        return []

    try:
        import osmnx as ox  # type: ignore

        graph = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
        _, edges = ox.graph_to_gdfs(graph)
        if edges.empty:
            return []

        spacing = max(0.0005, math.sqrt(polygon.area) / max(6, math.sqrt(n_points)))
        points: List[Point] = []
        for geom in edges.geometry.dropna():
            lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]
            for line in lines:
                for point in _points_along_line(line, spacing):
                    if polygon.covers(point):
                        points.append(point)

        return spread_points(points, n_points)
    except ImportError:
        return []
    except Exception as exc:
        logger.warning("OSMnx road sampling failed; using polygon sampling only: %s", exc)
        return []


def spread_points(points: Sequence[Point], n_points: int) -> List[Point]:
    """Greedy farthest-point thinning to reduce clustering."""
    unique = []
    seen = set()
    for point in points:
        key = (round(point.x, 6), round(point.y, 6))
        if key not in seen:
            unique.append(point)
            seen.add(key)

    if len(unique) <= n_points:
        return unique

    selected = [unique[0]]
    remaining = unique[1:]
    while remaining and len(selected) < n_points:
        best_idx = max(
            range(len(remaining)),
            key=lambda i: min(remaining[i].distance(existing) for existing in selected),
        )
        selected.append(remaining.pop(best_idx))
    return selected


def build_candidate_points(
    polygon: Polygon,
    n_points: int,
    rng: random.Random,
    use_roads: bool,
    logger: logging.Logger,
) -> List[Point]:
    road_target = n_points // 2 if use_roads else 0
    grid_target = max(1, n_points // 3)
    random_target = n_points

    candidates: List[Point] = []
    if use_roads:
        candidates.extend(sample_road_points_if_available(polygon, road_target, logger))
    candidates.extend(sample_grid_points(polygon, grid_target))
    candidates.extend(sample_random_points(polygon, random_target, rng))
    return spread_points(candidates, n_points)


def coordinate_cache_key(lat: object, lon: object, precision: int = 6) -> str:
    try:
        return f"{float(lat):.{precision}f},{float(lon):.{precision}f}"
    except (TypeError, ValueError):
        return ""


def query_streetview_metadata(
    lat: float,
    lon: float,
    api_key: str,
    session: requests.Session,
    retries: int,
    request_delay: float,
) -> Dict[str, object]:
    """Query the metadata endpoint with exponential backoff."""
    params = {"location": f"{lat},{lon}", "key": api_key}
    last_error = ""

    for attempt in range(retries):
        try:
            response = session.get(METADATA_URL, params=params, timeout=15)
            if response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")

            payload = response.json()
            time.sleep(request_delay)
            location = payload.get("location") or {}
            return {
                "query_lat": lat,
                "query_lon": lon,
                "status": payload.get("status", "API_ERROR"),
                "pano_id": payload.get("pano_id"),
                "pano_lat": location.get("lat"),
                "pano_lon": location.get("lng"),
                "date": payload.get("date"),
                "error_message": payload.get("error_message"),
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries - 1:
                time.sleep((2**attempt) * request_delay)

    return {
        "query_lat": lat,
        "query_lon": lon,
        "status": "API_ERROR",
        "pano_id": pd.NA,
        "pano_lat": pd.NA,
        "pano_lon": pd.NA,
        "date": pd.NA,
        "error_message": last_error,
    }


def validate_pano_inside_district(row: pd.Series, prepared_polygon) -> bool:
    try:
        point = Point(float(row["pano_lon"]), float(row["pano_lat"]))
    except (TypeError, ValueError):
        return False
    return bool(prepared_polygon.covers(point))


def download_streetview_image(
    pano_lat: float,
    pano_lon: float,
    heading: int,
    image_path: Path,
    api_key: str,
    session: requests.Session,
    retries: int,
    request_delay: float,
) -> bool:
    """Download a validated panorama image unless the path already exists."""
    if image_path.exists():
        return True

    params = {
        "size": "640x640",
        "location": f"{pano_lat},{pano_lon}",
        "heading": heading,
        "key": api_key,
    }

    last_error = ""
    for attempt in range(retries):
        try:
            response = session.get(IMAGE_URL, params=params, timeout=30)
            response.raise_for_status()
            if not response.content:
                raise ValueError("empty response body")
            image_path.write_bytes(response.content)
            time.sleep(request_delay)
            return True
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries - 1:
                time.sleep((2**attempt) * request_delay)

    LOGGER.warning("Image download failed for %s: %s", image_path.name, last_error)
    return False


def append_rows(df: pd.DataFrame, rows: List[Dict[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return ensure_columns(df, columns)
    return ensure_columns(pd.concat([df, pd.DataFrame(rows)], ignore_index=True, sort=False), columns)


def valid_image_count(downloaded: pd.DataFrame, district_id: str) -> int:
    subset = downloaded[
        (downloaded["district_id"] == district_id)
        & downloaded["download_success"].astype(str).str.lower().isin(["true", "1"])
    ]
    return int(subset["image_path"].nunique())


def truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def update_coverage_report(
    coverage: pd.DataFrame,
    district_row: pd.Series,
    downloaded: pd.DataFrame,
    metadata_cache: pd.DataFrame,
    rejected: pd.DataFrame,
    min_images_per_district: int,
) -> pd.DataFrame:
    district_id = normalize_district_id(district_row[ID_COL])
    district_meta = metadata_cache[metadata_cache["district_id"] == district_id]
    district_rejected = rejected[rejected["district_id"] == district_id]
    district_downloaded = downloaded[
        (downloaded["district_id"] == district_id)
        & downloaded["download_success"].astype(str).str.lower().isin(["true", "1"])
    ]
    metadata_coords = {
        coordinate_cache_key(row["query_lat"], row["query_lon"])
        for _, row in district_meta.iterrows()
    }
    rejected_coords = {
        coordinate_cache_key(row["query_lat"], row["query_lon"])
        for _, row in district_rejected.iterrows()
    }
    candidate_coords = {key for key in metadata_coords | rejected_coords if key}

    row = {
        "district_id": district_id,
        "district_name": district_row.get(NAME_COL, pd.NA),
        "stateabb": district_row.get("stateabb", pd.NA),
        "num_valid_images": int(district_downloaded["image_path"].nunique()),
        "num_unique_panos": int(district_downloaded["pano_id"].dropna().nunique()),
        "num_candidate_points": int(len(candidate_coords)),
        "num_metadata_ok": int((district_meta["status"] == "OK").sum()),
        "num_zero_results": int((district_meta["status"] == "ZERO_RESULTS").sum()),
        "num_outside_district": int((district_rejected["reason"] == "OUTSIDE_DISTRICT").sum()),
        "num_duplicates": int((district_rejected["reason"] == "DUPLICATE_PANO").sum()),
    }
    row["included_for_analysis"] = row["num_valid_images"] >= min_images_per_district

    coverage = ensure_columns(coverage, COVERAGE_COLUMNS)
    coverage = coverage[coverage["district_id"] != district_id].copy()
    coverage = pd.concat([coverage, pd.DataFrame([row])], ignore_index=True)
    return coverage.sort_values("district_id").reset_index(drop=True)


def save_outputs(state: ExistingState) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_columns(state.downloaded, DOWNLOADED_COLUMNS).to_csv(DOWNLOADED_IMAGES, index=False)
    ensure_columns(state.metadata_cache, METADATA_COLUMNS).to_csv(METADATA_CACHE, index=False)
    ensure_columns(state.rejected, REJECTED_COLUMNS).to_csv(REJECTED_CANDIDATES, index=False)
    ensure_columns(state.coverage, COVERAGE_COLUMNS).to_csv(DISTRICT_COVERAGE, index=False)

    analysis_ready = state.coverage[truthy_series(state.coverage["included_for_analysis"])].copy()
    analysis_ready.to_csv(ANALYSIS_READY, index=False)


def parse_headings(text: str) -> List[int]:
    headings = []
    for item in text.split(","):
        item = item.strip()
        if item:
            headings.append(int(item))
    return headings or [0]


def choose_districts_to_attempt(
    gdf: gpd.GeoDataFrame,
    state: ExistingState,
    target_districts: int,
    target_images_per_district: int,
) -> gpd.GeoDataFrame:
    current_counts = state.downloaded.groupby("district_id")["image_path"].nunique().to_dict()
    gdf = gdf.copy()
    gdf["_current_images"] = gdf[ID_COL].map(lambda value: current_counts.get(normalize_district_id(value), 0))
    needs_more = gdf[gdf["_current_images"] < target_images_per_district].copy()
    return needs_more.head(target_districts).drop(columns=["_current_images"])


def collect_for_district(
    district_row: pd.Series,
    state: ExistingState,
    args: argparse.Namespace,
    api_key: str,
    session: requests.Session,
) -> None:
    district_id = normalize_district_id(district_row[ID_COL])
    district_name = district_row.get(NAME_COL, "")
    stateabb = district_row.get("stateabb", "")
    polygon = largest_polygon(district_row.geometry)
    prepared_polygon = prep(polygon)
    rng = random.Random(args.random_seed + hash(district_id) % 100000)
    headings = parse_headings(args.headings)

    existing_panos = {
        str(pano_id)
        for pano_id in state.downloaded.loc[
            state.downloaded["district_id"] == district_id, "pano_id"
        ].dropna()
    }
    current_image_count = valid_image_count(state.downloaded, district_id)
    cache_by_key = {
        row["cache_key"]: row
        for _, row in state.metadata_cache.dropna(subset=["cache_key"]).iterrows()
    }

    candidates = build_candidate_points(
        polygon,
        args.candidate_points_per_district,
        rng,
        use_roads=not args.no_roads,
        logger=LOGGER,
    )
    LOGGER.info(
        "District %s (%s): %s candidate points, %s existing images",
        district_id,
        district_name,
        len(candidates),
        current_image_count,
    )

    metadata_rows: List[Dict[str, object]] = []
    rejected_rows: List[Dict[str, object]] = []
    downloaded_rows: List[Dict[str, object]] = []

    for point in tqdm(candidates, desc=f"district {district_id}", leave=False):
        if current_image_count >= args.target_images_per_district:
            break

        cache_key = coordinate_cache_key(point.y, point.x)
        if cache_key in cache_by_key:
            metadata = cache_by_key[cache_key].to_dict()
        else:
            metadata = query_streetview_metadata(
                point.y,
                point.x,
                api_key,
                session,
                retries=args.retries,
                request_delay=args.request_delay,
            )
            metadata.update(
                {
                    "district_id": district_id,
                    "district_name": district_name,
                    "stateabb": stateabb,
                    "source": "new_query",
                    "cache_key": cache_key,
                    "queried_at": pd.Timestamp.utcnow().isoformat(),
                }
            )
            metadata_rows.append(metadata)
            cache_by_key[cache_key] = pd.Series(metadata)

        status = str(metadata.get("status", "API_ERROR"))
        reject_base = {
            "district_id": district_id,
            "district_name": district_name,
            "stateabb": stateabb,
            "query_lat": metadata.get("query_lat", point.y),
            "query_lon": metadata.get("query_lon", point.x),
            "status": status,
            "pano_id": metadata.get("pano_id"),
            "pano_lat": metadata.get("pano_lat"),
            "pano_lon": metadata.get("pano_lon"),
            "source": metadata.get("source", "cache"),
        }

        if status != "OK":
            reason = "ZERO_RESULTS" if status == "ZERO_RESULTS" else "API_ERROR"
            rejected_rows.append({**reject_base, "reason": reason})
            continue

        if not validate_pano_inside_district(pd.Series(metadata), prepared_polygon):
            rejected_rows.append({**reject_base, "reason": "OUTSIDE_DISTRICT"})
            continue

        pano_id = str(metadata.get("pano_id"))
        if pano_id in existing_panos:
            rejected_rows.append({**reject_base, "reason": "DUPLICATE_PANO"})
            continue

        existing_panos.add(pano_id)
        for heading in headings:
            if current_image_count >= args.target_images_per_district:
                break

            filename = (
                f"{sanitize_filename_part(district_id)}_"
                f"{sanitize_filename_part(pano_id)}_{int(heading)}.jpg"
            )
            image_path = IMAGE_DIR / filename
            image_path_key = str(image_path).lower()
            if image_path_key in state.existing_paths:
                continue

            success = download_streetview_image(
                float(metadata["pano_lat"]),
                float(metadata["pano_lon"]),
                int(heading),
                image_path,
                api_key,
                session,
                retries=args.retries,
                request_delay=args.request_delay,
            )
            if success:
                row = {
                    "district_id": district_id,
                    "district_name": district_name,
                    "stateabb": stateabb,
                    "pano_id": pano_id,
                    "pano_lat": metadata.get("pano_lat"),
                    "pano_lon": metadata.get("pano_lon"),
                    "heading": int(heading),
                    "image_path": str(image_path),
                    "download_success": True,
                    "downloaded_at": pd.Timestamp.utcnow().isoformat(),
                }
                downloaded_rows.append(row)
                state.existing_paths.add(image_path_key)
                current_image_count += 1

        state.downloaded = append_rows(state.downloaded, downloaded_rows, DOWNLOADED_COLUMNS)
        downloaded_rows = []

    state.metadata_cache = append_rows(state.metadata_cache, metadata_rows, METADATA_COLUMNS)
    state.metadata_cache = state.metadata_cache.drop_duplicates(subset=["cache_key"], keep="first")
    state.rejected = append_rows(state.rejected, rejected_rows, REJECTED_COLUMNS)
    state.coverage = update_coverage_report(
        state.coverage,
        district_row,
        state.downloaded,
        state.metadata_cache,
        state.rejected,
        args.min_images_per_district,
    )
    save_outputs(state)


def summarize_final(state: ExistingState, attempted: int, min_images: int) -> None:
    coverage = state.coverage.copy()
    included = truthy_series(coverage["included_for_analysis"])
    usable = coverage[included]
    total_images = int(state.downloaded["image_path"].nunique())
    avg_images = float(usable["num_valid_images"].mean()) if not usable.empty else 0.0
    below = coverage.loc[~included, "district_id"].tolist()

    LOGGER.info("Collection complete")
    LOGGER.info("total districts attempted: %s", attempted)
    LOGGER.info("total districts with >= %s images: %s", min_images, len(usable))
    LOGGER.info("total images downloaded/known: %s", total_images)
    LOGGER.info("average images per usable district: %.2f", avg_images)
    LOGGER.info("districts still below threshold: %s", len(below))
    if below:
        LOGGER.info("below-threshold district IDs: %s", ", ".join(map(str, below[:50])))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand the district Street View dataset.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--target-districts", type=int, default=200)
    parser.add_argument("--min-images-per-district", type=int, default=15)
    parser.add_argument("--target-images-per-district", type=int, default=30)
    parser.add_argument("--candidate-points-per-district", type=int, default=150)
    parser.add_argument("--state-filter", default=None, help="Optional comma-separated state abbreviations.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--headings", default="0", help="Comma-separated headings, e.g. 0 or 0,90,180,270.")
    parser.add_argument("--boundaries-path", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--seda-path", type=Path, default=DEFAULT_SEDA)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-roads", action="store_true", help="Disable optional OSMnx road-aware sampling.")
    return parser


def main() -> None:
    load_dotenv()
    args = build_arg_parser().parse_args()
    random.seed(args.random_seed)
    ensure_fiona_compatibility()

    if pd is None or requests is None or gpd is None or Point is None:
        raise RuntimeError(
            "Missing required dependencies. Run `pip install -r requirements.txt` "
            "from the project root before collecting Street View data."
        )

    api_key = args.api_key or os.getenv("GOOGLE_STREET_VIEW_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Google API key. Set GOOGLE_STREET_VIEW_API_KEY, GOOGLE_API_KEY, or pass --api-key."
        )

    state = load_existing_metadata()
    gdf = load_district_boundaries(
        args.boundaries_path,
        args.seda_path,
        args.sample_size,
        args.target_districts,
        args.state_filter,
        args.random_seed,
    )
    districts = choose_districts_to_attempt(
        gdf,
        state,
        args.target_districts,
        args.target_images_per_district,
    )

    LOGGER.info("Attempting %s districts from %s loaded candidate districts", len(districts), len(gdf))
    attempted = 0
    with requests.Session() as session:
        for _, district_row in districts.iterrows():
            attempted += 1
            collect_for_district(district_row, state, args, api_key, session)

    save_outputs(state)
    summarize_final(state, attempted, args.min_images_per_district)


if __name__ == "__main__":
    main()
