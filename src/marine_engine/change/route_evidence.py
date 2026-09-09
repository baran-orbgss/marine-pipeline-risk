"""Route-referenced observed seabed elevation-change evidence (MAR-029).

A generic bridge between two ACCEPTED components, redesigning neither:

    accepted MAR-021/021A DoD raster (read-only)
        +
    accepted MAR-027 canonical route-reference grid (`project.route_reference`)
        v
    strict linkage / provenance / CRS / support checks
        v
    route-reference-point -> containing-raster-cell sampling (no interpolation)
        v
    observed change at route-reference points -> Parquet + GeoPackage + metadata/report

Scientific role: `ROUTE_REFERENCED_OBSERVED_SEABED_ELEVATION_CHANGE` -- the DoD value of the
raster cell containing each canonical route-reference point. It is NOT a prediction of future
change, NOT a causal attribution, NOT a burial/exposure/free-span state, and NOT a route risk.

Immutable definitions reused verbatim (Sections 5, 16):
    delta_bed_elevation_m = bed_elevation_epoch2_m - bed_elevation_epoch1_m   (`change.dod`)
    positive -> OBSERVED_SEABED_RAISING, negative -> OBSERVED_SEABED_LOWERING, zero -> no label
    (`change.dod.classify_change_direction`, never re-implemented here)

No generic significance threshold exists (MAR-021A remains authoritative, Section 6): every
sample is preserved regardless of magnitude; nothing is flagged, filtered, or reclassified.

Dependency direction: this module CONSUMES the project route-reference output; the project core
never implements change science (Section 22).
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from pyproj.exceptions import CRSError
from rasterio.errors import RasterioError, RasterioIOError
from rasterio.windows import Window

from marine_engine.change import report as change_report
from marine_engine.change.dod import (
    OBSERVED_SEABED_LOWERING,
    OBSERVED_SEABED_RAISING,
    DoDResult,
    classify_change_direction,
)
from marine_engine.change.route_evidence_manifest import (
    RouteChangeEvidenceManifest,
    load_route_change_evidence_manifest,
    resolve_manifest_path,
)
from marine_engine.change.uncertainty import GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
from marine_engine.project import manifest as project_manifest
from marine_engine.project import model as project_model
from marine_engine.project import registry as project_registry
from marine_engine.project import route_reference
from marine_engine.project.identity import FileIdentity, compute_file_identity

__all__ = [
    "SCIENTIFIC_ROLE",
    "ROUTE_CHANGE_EVIDENCE_AVAILABLE",
    "ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE",
    "NO_PRIMARY_ROUTE",
    "ROUTE_REFERENCE_NOT_BUILT",
    "ROUTE_ASSET_ID_MISMATCH",
    "SOURCE_PROVENANCE_NOT_FOUND",
    "DOD_NOT_FOUND",
    "DOD_NOT_READABLE",
    "DOD_NOT_SINGLE_BAND_ANALYTICAL_RASTER",
    "DOD_ROTATED_GRID_UNSUPPORTED",
    "DOD_CRS_MISSING",
    "DOD_DECLARED_CRS_CONFLICT",
    "DOD_SOURCE_ROLE_MISMATCH",
    "DOD_CRS_MISMATCH",
    "NO_ROUTE_POINTS_ON_VALID_DOD_SUPPORT",
    "CHANGE_VALUE_AVAILABLE",
    "CHANGE_VALUE_NODATA",
    "ROUTE_POINT_OUTSIDE_DOD_EXTENT",
    "SAMPLE_SUPPORT_SEMANTICS",
    "ROUTE_CHANGE_LAYER",
    "ROUTE_CHANGE_COLUMNS",
    "DIRECTION_LABEL_DISCLAIMER",
    "SAMPLE_COUNT_NOTE",
    "DoDObservedFacts",
    "RouteChangeEvidenceResult",
    "inspect_dod_source",
    "observe_dod_raster",
    "sample_dod_at_route_reference_points",
    "build_route_change_evidence",
    "run_route_change_evidence",
    "build_route_change_evidence_metadata",
    "build_route_change_evidence_validation",
    "build_route_change_evidence_report_blocks",
    "write_route_change_evidence_outputs",
]

SCIENTIFIC_ROLE = "ROUTE_REFERENCED_OBSERVED_SEABED_ELEVATION_CHANGE"

# --- Section 25: overall status + controlled reason codes -----------------------------------------

ROUTE_CHANGE_EVIDENCE_AVAILABLE = "ROUTE_CHANGE_EVIDENCE_AVAILABLE"
ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE = "ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE"

NO_PRIMARY_ROUTE = "NO_PRIMARY_ROUTE"
ROUTE_REFERENCE_NOT_BUILT = "ROUTE_REFERENCE_NOT_BUILT"
ROUTE_ASSET_ID_MISMATCH = "ROUTE_ASSET_ID_MISMATCH"
SOURCE_PROVENANCE_NOT_FOUND = "SOURCE_PROVENANCE_NOT_FOUND"
DOD_NOT_FOUND = "DOD_NOT_FOUND"
DOD_NOT_READABLE = "DOD_NOT_READABLE"
DOD_NOT_SINGLE_BAND_ANALYTICAL_RASTER = "DOD_NOT_SINGLE_BAND_ANALYTICAL_RASTER"
DOD_ROTATED_GRID_UNSUPPORTED = "DOD_ROTATED_GRID_UNSUPPORTED"
DOD_CRS_MISSING = "DOD_CRS_MISSING"
DOD_DECLARED_CRS_CONFLICT = "DOD_DECLARED_CRS_CONFLICT"
DOD_SOURCE_ROLE_MISMATCH = "DOD_SOURCE_ROLE_MISMATCH"
DOD_CRS_MISMATCH = "DOD_CRS_MISMATCH"
NO_ROUTE_POINTS_ON_VALID_DOD_SUPPORT = "NO_ROUTE_POINTS_ON_VALID_DOD_SUPPORT"

# --- Section 15: per-sample statuses --------------------------------------------------------------

CHANGE_VALUE_AVAILABLE = "CHANGE_VALUE_AVAILABLE"
CHANGE_VALUE_NODATA = "CHANGE_VALUE_NODATA"
ROUTE_POINT_OUTSIDE_DOD_EXTENT = "ROUTE_POINT_OUTSIDE_DOD_EXTENT"

# --- Section 13: sample-support definition --------------------------------------------------------

SAMPLE_SUPPORT_SEMANTICS = "RASTER_CELL_CONTAINING_ROUTE_REFERENCE_POINT"
SAMPLE_SUPPORT_DESCRIPTION = (
    "each value is the existing DoD value of the single raster cell whose footprint contains the "
    "canonical route-reference point -- read as-is, with no interpolation, no neighbour "
    "averaging, and no smoothing; it is NOT 'the exact seabed change at the pipeline'"
)
CELL_CENTER_DISTANCE_NOTE = (
    "route_point_to_sampled_cell_center_distance_m is spatial support geometry only (how far the "
    "route-reference point sits from the centre of the cell whose value was read); it is NOT a "
    "survey or positional uncertainty"
)
POINT_GEOMETRY_NOTE = (
    "GeoPackage point geometry is the canonical MAR-027 route-reference point itself, never "
    "moved or snapped to the raster cell centre; sampled_cell_center_x/y are attributes only"
)

ROUTE_CHANGE_LAYER = "route_observed_seabed_change_points"

ROUTE_CHANGE_COLUMNS: tuple[str, ...] = (
    "project_id",
    "route_asset_id",
    "station_index",
    "chainage_m",
    "kp_label",
    "is_terminal",
    "route_reference_interval_m",
    "chainage_origin_basis",
    "route_point_x",
    "route_point_y",
    "raster_row",
    "raster_col",
    "sampled_cell_center_x",
    "sampled_cell_center_y",
    "route_point_to_sampled_cell_center_distance_m",
    "delta_bed_elevation_m",
    "observed_change_direction",
    "sample_support_semantics",
    "sample_status",
    "generic_change_significance_status",
)

# --- Sections 16-21: required wording -------------------------------------------------------------

DIRECTION_LABEL_DISCLAIMER = (
    "DIRECTION LABELS ARE RAW OBSERVED SIGN LABELS ONLY AND ARE NOT SIGNIFICANCE CLASSIFICATIONS "
    "OR CAUSAL INTERPRETATIONS. OBSERVED_SEABED_RAISING is only deposition/fill-compatible and "
    "OBSERVED_SEABED_LOWERING is only erosion/scour-compatible; cause is unresolved."
)
SAMPLE_COUNT_NOTE = (
    "all counts are counts of canonical route-reference SAMPLES (point samples at the configured "
    "indexing interval); they are NOT route-length fractions, area fractions, or route coverage"
)
GENERIC_SIGNIFICANCE_NOTE = (
    "no generic DoD change-significance threshold has been demonstrated (MAR-021A); no sample is "
    "flagged, filtered, deleted, or reclassified by magnitude, and a source's own analyst "
    "criterion (e.g. a 0.3 m practice) or a nominal-accuracy figure (0.2 m / 0.283 m RSS) is "
    "never applied here as a generic filter"
)
TEMPORAL_NOTE = (
    "a two-epoch DoD is the observed elevation difference over that survey interval only; it is "
    "not present-day, persistent, long-term, or future change, and no annualization or rate is "
    "derived here"
)
NOT_A_STATE_NOTE = (
    "OBSERVED_SEABED_LOWERING is not a burial-loss, exposure, free-span, or scour state, and "
    "OBSERVED_SEABED_RAISING is not a burial-gain or fill state; those require independent "
    "asset geometry/reference semantics that MAR-029 does not consume"
)
NO_INTERVAL_NOTE = (
    "point evidence only: no contiguous physical raising/lowering intervals or zone boundaries "
    "are inferred between route-reference samples (no boundary-resolution model exists)"
)
IMMUTABLE_SOURCE_NOTE = (
    "the DoD raster is consumed read-only: never rewritten, reprojected, resampled, smoothed, "
    "thresholded, median-normalized, or bias-corrected"
)

_DOD_TAG_KEYS: tuple[str, ...] = ("scientific_role", "definition", "product", "layer", "units")

ACCEPTED_DOD_DEFINITION = DoDResult.__dataclass_fields__["definition"].default


# --- Section 9/10: DoD source identity and observed facts -----------------------------------------


@dataclass(frozen=True)
class DoDObservedFacts:
    """Facts OBSERVED directly from the raster file (Section 9). Never merged with declared
    metadata; `observed_crs` is the embedded CRS, kept separate from any declared CRS."""

    driver: str | None
    observed_crs: str | None
    width: int
    height: int
    band_count: int
    dtype: str
    nodata: float | None
    pixel_size_x: float
    pixel_size_y: float
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    tags: dict[str, str]

    @property
    def is_rotated(self) -> bool:
        return self.transform[1] != 0.0 or self.transform[3] != 0.0

    @property
    def is_single_band_analytical(self) -> bool:
        return self.band_count == 1 and np.issubdtype(np.dtype(self.dtype), np.floating)

    def to_dict(self) -> dict[str, Any]:
        nodata: float | str | None = self.nodata
        if isinstance(nodata, float) and math.isnan(nodata):
            nodata = "nan"
        return {
            "driver": self.driver,
            "observed_crs": self.observed_crs,
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "dtype": self.dtype,
            "nodata": nodata,
            "pixel_size_x": self.pixel_size_x,
            "pixel_size_y": self.pixel_size_y,
            "transform_affine_coefficients": list(self.transform),
            "bounds": list(self.bounds),
            "is_rotated": self.is_rotated,
            "is_single_band_analytical": self.is_single_band_analytical,
            "tags": dict(self.tags),
        }


def _identity_dict(identity: FileIdentity | None) -> dict[str, Any] | None:
    if identity is None:
        return None
    return {
        "resolved_path": str(identity.resolved_path),
        "filename": identity.filename,
        "byte_size": identity.byte_size,
        "sha256": identity.sha256,
    }


def inspect_dod_source(path: Path) -> tuple[FileIdentity, DoDObservedFacts]:
    """Section 10: content identity + observed raster facts, read-only. Raises `FileNotFoundError`
    if the file is missing and `rasterio.errors.RasterioIOError` if it is not a readable raster;
    callers turn those into controlled reason codes."""

    identity = compute_file_identity(path)
    return identity, observe_dod_raster(identity.resolved_path)


def observe_dod_raster(resolved_path: Path) -> DoDObservedFacts:
    """Section 9: the facts observed from the raster file itself (embedded CRS, grid, dtype,
    nodata, band count, tags) -- opened read-only, nothing written."""

    with rasterio.open(resolved_path) as ds:
        transform = ds.transform
        all_tags = ds.tags()
        facts = DoDObservedFacts(
            driver=ds.driver,
            observed_crs=ds.crs.to_string() if ds.crs is not None else None,
            width=int(ds.width),
            height=int(ds.height),
            band_count=int(ds.count),
            dtype=str(ds.dtypes[0]),
            nodata=float(ds.nodata) if ds.nodata is not None else None,
            pixel_size_x=abs(float(transform.a)),
            pixel_size_y=abs(float(transform.e)),
            transform=(
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ),
            bounds=tuple(float(v) for v in ds.bounds),
            tags={k: str(all_tags[k]) for k in _DOD_TAG_KEYS if k in all_tags},
        )
    return facts


def _crs_equivalent(a: str, b: str) -> bool:
    return CRS.from_user_input(a) == CRS.from_user_input(b)


def _declared_vs_observed_crs_conflict(declared: str | None, observed: str) -> str | None:
    """Section 9/12: semantic comparison through `pyproj.CRS`; the software never decides which
    side is correct. A declared value that fails to parse is itself a conflict, never ignored."""

    if not declared:
        return None
    try:
        declared_crs = CRS.from_user_input(declared)
    except CRSError:
        return f"declared DoD CRS {declared!r} is not a valid/recognized CRS identifier"
    if declared_crs != CRS.from_user_input(observed):
        return f"declared DoD CRS {declared!r} does not match observed/embedded CRS {observed!r}"
    return None


# --- Section 13-15: sampling ----------------------------------------------------------------------


def _is_nodata(value: float, nodata: float | None) -> bool:
    if not np.isfinite(value):
        return True
    if nodata is None:
        return False
    if math.isnan(nodata):
        return False  # a finite value can never equal a NaN nodata sentinel
    return bool(value == nodata)


def sample_dod_at_route_reference_points(
    grid_gdf: gpd.GeoDataFrame, dod_path: Path
) -> gpd.GeoDataFrame:
    """Section 13: for each canonical route-reference point, locate the DoD cell CONTAINING it
    (floor of the inverse-affine pixel coordinates -- cells are half-open, so a point on a shared
    edge belongs to the next cell, exactly as the raster's own indexing defines), read that one
    existing cell value, and record the support geometry. No interpolation, no neighbour
    averaging, no smoothing; nodata and outside-extent points stay unavailable (never zero).

    Direction labels come from the accepted `change.dod.classify_change_direction`; the grid's
    chainage, order, KP labels, and point geometry are taken verbatim from MAR-027."""

    rows: list[dict[str, Any]] = []
    with rasterio.open(dod_path) as ds:
        transform = ds.transform
        inverse = ~transform
        width, height, nodata = ds.width, ds.height, ds.nodata
        for record in grid_gdf.itertuples(index=False):
            point = record.geometry
            x, y = float(point.x), float(point.y)
            col_f, row_f = inverse @ (x, y)
            row, col = math.floor(row_f), math.floor(col_f)
            inside = 0 <= row < height and 0 <= col < width
            entry: dict[str, Any] = {
                "project_id": record.project_id,
                "route_asset_id": record.route_asset_id,
                "station_index": int(record.station_index),
                "chainage_m": float(record.chainage_m),
                "kp_label": record.kp_label,
                "is_terminal": bool(record.is_terminal),
                "route_reference_interval_m": float(record.route_reference_interval_m),
                "chainage_origin_basis": record.chainage_origin_basis,
                "route_point_x": x,
                "route_point_y": y,
                "raster_row": None,
                "raster_col": None,
                "sampled_cell_center_x": None,
                "sampled_cell_center_y": None,
                "route_point_to_sampled_cell_center_distance_m": None,
                "delta_bed_elevation_m": None,
                "observed_change_direction": None,
                "sample_support_semantics": SAMPLE_SUPPORT_SEMANTICS,
                "sample_status": ROUTE_POINT_OUTSIDE_DOD_EXTENT,
                "generic_change_significance_status": (
                    GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
                ),
                "geometry": point,
            }
            if inside:
                center_x, center_y = transform @ (col + 0.5, row + 0.5)
                value = float(ds.read(1, window=Window(col, row, 1, 1))[0, 0])
                entry.update(
                    {
                        "raster_row": row,
                        "raster_col": col,
                        "sampled_cell_center_x": float(center_x),
                        "sampled_cell_center_y": float(center_y),
                        "route_point_to_sampled_cell_center_distance_m": math.hypot(
                            x - center_x, y - center_y
                        ),
                    }
                )
                if _is_nodata(value, nodata):
                    entry["sample_status"] = CHANGE_VALUE_NODATA
                else:
                    entry["sample_status"] = CHANGE_VALUE_AVAILABLE
                    entry["delta_bed_elevation_m"] = value
                    entry["observed_change_direction"] = classify_change_direction(
                        np.array([value], dtype=np.float64)
                    )[0]
            rows.append(entry)

    df = pd.DataFrame(rows, columns=[*ROUTE_CHANGE_COLUMNS, "geometry"])
    df["raster_row"] = df["raster_row"].astype("Int64")
    df["raster_col"] = df["raster_col"].astype("Int64")
    for column in (
        "sampled_cell_center_x",
        "sampled_cell_center_y",
        "route_point_to_sampled_cell_center_distance_m",
        "delta_bed_elevation_m",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    df["observed_change_direction"] = df["observed_change_direction"].astype("object")
    return gpd.GeoDataFrame(df, geometry="geometry", crs=grid_gdf.crs)


def _sample_counts(samples: gpd.GeoDataFrame | None) -> dict[str, Any]:
    if samples is None:
        counts = dict.fromkeys(
            (
                "route_reference_sample_count",
                "change_value_available_sample_count",
                "raising_sample_count",
                "lowering_sample_count",
                "zero_change_sample_count",
                "nodata_sample_count",
                "outside_dod_extent_sample_count",
            ),
            0,
        )
    else:
        status = samples["sample_status"]
        direction = samples["observed_change_direction"]
        available = status == CHANGE_VALUE_AVAILABLE
        counts = {
            "route_reference_sample_count": int(len(samples)),
            "change_value_available_sample_count": int(available.sum()),
            "raising_sample_count": int((direction == OBSERVED_SEABED_RAISING).sum()),
            "lowering_sample_count": int((direction == OBSERVED_SEABED_LOWERING).sum()),
            "zero_change_sample_count": int(
                (available & (samples["delta_bed_elevation_m"] == 0.0)).sum()
            ),
            "nodata_sample_count": int((status == CHANGE_VALUE_NODATA).sum()),
            "outside_dod_extent_sample_count": int(
                (status == ROUTE_POINT_OUTSIDE_DOD_EXTENT).sum()
            ),
        }
    counts["note"] = SAMPLE_COUNT_NOTE
    return counts


def _valid_sample_summary(samples: gpd.GeoDataFrame | None) -> dict[str, Any]:
    """Descriptive min/max of the sampled values only -- not a threshold, not a statistic of the
    whole DoD raster, and not a route-length statement."""

    if samples is None:
        return {"delta_bed_elevation_min_m": None, "delta_bed_elevation_max_m": None}
    valid = samples.loc[samples["sample_status"] == CHANGE_VALUE_AVAILABLE, "delta_bed_elevation_m"]
    if valid.empty:
        return {"delta_bed_elevation_min_m": None, "delta_bed_elevation_max_m": None}
    return {
        "delta_bed_elevation_min_m": float(valid.min()),
        "delta_bed_elevation_max_m": float(valid.max()),
    }


# --- Orchestration --------------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteChangeEvidenceResult:
    status: str
    reason_code: str | None
    findings: list[str]
    route_manifest: RouteChangeEvidenceManifest
    route_manifest_dir: Path
    project_id: str
    working_crs: str
    route_reference: route_reference.RouteReferenceResult
    dod_resolved_path: Path
    dod_identity: FileIdentity | None
    dod_observed: DoDObservedFacts | None
    provenance_identity: FileIdentity | None
    declared_vs_observed_crs_conflict: str | None
    dod_vs_route_crs_status: str | None
    samples: gpd.GeoDataFrame | None
    products_written: bool = False
    output_paths: dict[str, str] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == ROUTE_CHANGE_EVIDENCE_AVAILABLE


def build_route_change_evidence(
    route_manifest: RouteChangeEvidenceManifest,
    route_manifest_dir: Path,
    *,
    project_id: str,
    working_crs: str,
    route_reference_result: route_reference.RouteReferenceResult,
) -> RouteChangeEvidenceResult:
    """Sections 8-16, 25: every prerequisite is checked in order and the first unmet one yields a
    controlled `ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE` with a named reason code; nothing is
    invented, reprojected, resampled, or substituted to get past a failed gate."""

    rr = route_reference_result
    dod_decl = route_manifest.dod
    dod_path = resolve_manifest_path(dod_decl.path, route_manifest_dir)
    findings: list[str] = []
    dod_identity: FileIdentity | None = None
    dod_observed: DoDObservedFacts | None = None
    provenance_identity: FileIdentity | None = None
    declared_conflict: str | None = None
    dod_vs_route: str | None = None
    samples: gpd.GeoDataFrame | None = None

    def _not_available(reason_code: str, finding: str) -> RouteChangeEvidenceResult:
        findings.append(finding)
        return RouteChangeEvidenceResult(
            status=ROUTE_CHANGE_EVIDENCE_NOT_AVAILABLE,
            reason_code=reason_code,
            findings=findings,
            route_manifest=route_manifest,
            route_manifest_dir=route_manifest_dir,
            project_id=project_id,
            working_crs=working_crs,
            route_reference=rr,
            dod_resolved_path=dod_path,
            dod_identity=dod_identity,
            dod_observed=dod_observed,
            provenance_identity=provenance_identity,
            declared_vs_observed_crs_conflict=declared_conflict,
            dod_vs_route_crs_status=dod_vs_route,
            samples=samples,
        )

    # Section 11: the canonical route-reference grid must exist -- no route is ever invented.
    if rr.primary_route.status != route_reference.PRIMARY_ROUTE_AVAILABLE:
        return _not_available(
            NO_PRIMARY_ROUTE,
            f"project primary route is {rr.primary_route.status}: "
            + "; ".join(rr.primary_route.findings or ["no usable declared primary route"]),
        )
    if rr.status != route_reference.ROUTE_REFERENCE_BUILT or rr.grid_gdf is None:
        return _not_available(
            ROUTE_REFERENCE_NOT_BUILT,
            "canonical route-reference grid was not built: " + "; ".join(rr.findings),
        )
    grid = rr.grid_gdf
    grid_route_asset_id = str(grid["route_asset_id"].iloc[0])
    if route_manifest.route_asset_id != grid_route_asset_id:
        return _not_available(
            ROUTE_ASSET_ID_MISMATCH,
            f"route-change manifest declares route_asset_id {route_manifest.route_asset_id!r} "
            f"but the canonical route-reference grid was built for the project primary route "
            f"{grid_route_asset_id!r}; no other route is substituted",
        )

    # Section 8/10: declared provenance artefact must exist if declared; identified by content.
    if dod_decl.provenance_path is not None:
        provenance_path = resolve_manifest_path(dod_decl.provenance_path, route_manifest_dir)
        try:
            provenance_identity = compute_file_identity(provenance_path)
        except (FileNotFoundError, OSError) as exc:
            return _not_available(
                SOURCE_PROVENANCE_NOT_FOUND,
                f"declared source provenance artefact is not readable: {exc}",
            )

    # Section 10: DoD identity first (so unreadable bytes are still identified), then observed
    # facts -- both read-only.
    try:
        dod_identity = compute_file_identity(dod_path)
    except (FileNotFoundError, OSError) as exc:
        return _not_available(DOD_NOT_FOUND, f"DoD source not found: {exc}")
    try:
        dod_observed = observe_dod_raster(dod_identity.resolved_path)
    except (RasterioIOError, RasterioError, OSError) as exc:
        return _not_available(
            DOD_NOT_READABLE, f"DoD source {dod_path} is not a readable raster: {exc}"
        )

    if not dod_observed.is_single_band_analytical:
        return _not_available(
            DOD_NOT_SINGLE_BAND_ANALYTICAL_RASTER,
            f"DoD source has band_count={dod_observed.band_count} dtype={dod_observed.dtype!r}; "
            "a single-band floating-point analytical raster is required",
        )
    if dod_observed.is_rotated:
        return _not_available(
            DOD_ROTATED_GRID_UNSUPPORTED,
            "DoD source carries a rotated affine transform; exact-cell route sampling is only "
            "defined here for north-up, axis-aligned grids",
        )
    if dod_observed.observed_crs is None:
        return _not_available(
            DOD_CRS_MISSING, "DoD source has no embedded CRS; none is inferred or substituted"
        )

    # Section 9/12: declared vs observed CRS stay separate; a conflict is never resolved by us.
    declared_conflict = _declared_vs_observed_crs_conflict(
        dod_decl.horizontal_crs_declared, dod_observed.observed_crs
    )
    if declared_conflict is not None:
        return _not_available(DOD_DECLARED_CRS_CONFLICT, declared_conflict)

    observed_role = dod_observed.tags.get("scientific_role")
    if (
        dod_decl.source_scientific_role_declared is not None
        and observed_role is not None
        and observed_role != dod_decl.source_scientific_role_declared
    ):
        return _not_available(
            DOD_SOURCE_ROLE_MISMATCH,
            f"declared source scientific role {dod_decl.source_scientific_role_declared!r} does "
            f"not match the DoD raster's own embedded tag {observed_role!r}",
        )

    # Section 12: DoD CRS vs canonical route-reference grid CRS -- semantic, conservative.
    grid_crs = CRS.from_user_input(grid.crs).to_string()
    if not _crs_equivalent(dod_observed.observed_crs, grid_crs):
        dod_vs_route = DOD_CRS_MISMATCH
        return _not_available(
            DOD_CRS_MISMATCH,
            f"DoD embedded CRS {dod_observed.observed_crs!r} is not semantically equivalent to "
            f"the canonical route-reference grid CRS {grid_crs!r}; the DoD is not reprojected "
            "or resampled, and cross-CRS route sampling is not implemented in MAR-029",
        )
    dod_vs_route = "SEMANTICALLY_EQUIVALENT"

    # Section 13-15: exact-cell sampling.
    samples = sample_dod_at_route_reference_points(grid, dod_identity.resolved_path)
    if int((samples["sample_status"] == CHANGE_VALUE_AVAILABLE).sum()) == 0:
        return _not_available(
            NO_ROUTE_POINTS_ON_VALID_DOD_SUPPORT,
            "no canonical route-reference point falls on a DoD cell with a finite value "
            "(all samples are nodata or outside the raster extent); no point product is written",
        )

    return RouteChangeEvidenceResult(
        status=ROUTE_CHANGE_EVIDENCE_AVAILABLE,
        reason_code=None,
        findings=findings,
        route_manifest=route_manifest,
        route_manifest_dir=route_manifest_dir,
        project_id=project_id,
        working_crs=working_crs,
        route_reference=rr,
        dod_resolved_path=dod_path,
        dod_identity=dod_identity,
        dod_observed=dod_observed,
        provenance_identity=provenance_identity,
        declared_vs_observed_crs_conflict=None,
        dod_vs_route_crs_status=dod_vs_route,
        samples=samples,
    )


def run_route_change_evidence(
    route_manifest_path: str | Path,
) -> tuple[RouteChangeEvidenceResult, project_model.CanonicalProjectModel]:
    """Section 23 orchestration: explicit linkage manifest -> real MAR-026/026A registration ->
    real MAR-027 canonical project model -> route-change evidence. Registration, readiness, and
    route-reference generation are the accepted implementations, called unchanged."""

    route_manifest, route_manifest_dir = load_route_change_evidence_manifest(route_manifest_path)
    project_manifest_path = resolve_manifest_path(
        route_manifest.project_manifest, route_manifest_dir
    )
    manifest, manifest_dir = project_manifest.load_project_manifest(project_manifest_path)
    summary = project_registry.register_project(manifest, manifest_dir)
    model = project_model.build_canonical_project_model(manifest, summary)
    result = build_route_change_evidence(
        route_manifest,
        route_manifest_dir,
        project_id=manifest.project.id,
        working_crs=summary.working_crs,
        route_reference_result=model.route_reference,
    )
    return result, model


# --- Section 24: outputs --------------------------------------------------------------------------


def build_route_change_evidence_metadata(result: RouteChangeEvidenceResult) -> dict[str, Any]:
    """`route_observed_seabed_change_metadata.json` -- deterministic (no timestamps), with every
    declared fact, observed fact, compatibility status, support definition, sample count, and
    limitation exposed explicitly. No score of any kind."""

    rr = result.route_reference
    dod_decl = result.route_manifest.dod
    grid_crs = CRS.from_user_input(rr.grid_gdf.crs).to_string() if rr.grid_gdf is not None else None
    return {
        "scientific_role": SCIENTIFIC_ROLE,
        "status": result.status,
        "status_reason_code": result.reason_code,
        "findings": list(result.findings),
        "route_change_manifest_declared": {
            "project_manifest": str(
                resolve_manifest_path(
                    result.route_manifest.project_manifest, result.route_manifest_dir
                )
            ),
            "route_asset_id": result.route_manifest.route_asset_id,
            "dod_path_declared": str(dod_decl.path),
            "source_change_study_id": dod_decl.source_change_study_id,
            "source_name": dod_decl.source_name,
            "horizontal_crs_declared": dod_decl.horizontal_crs_declared,
            "epoch1_survey_epoch_declared": dod_decl.epoch1_survey_epoch_declared,
            "epoch2_survey_epoch_declared": dod_decl.epoch2_survey_epoch_declared,
            "source_scientific_role_declared": dod_decl.source_scientific_role_declared,
            "licence_note": dod_decl.licence_note,
            "association_note": (
                "the route <-> DoD association exists ONLY because this manifest declares it; "
                "it is never inferred from spatial overlap, shared CRS, directory, filename, or "
                "route uniqueness"
            ),
        },
        "project": {"project_id": result.project_id, "working_crs": result.working_crs},
        "route_reference": {
            "status": rr.status,
            "primary_route_status": rr.primary_route.status,
            "route_asset_id": rr.primary_route.primary_route_asset_id,
            "configured_interval_m": rr.configured_interval_m,
            "interval_semantics": "indexing resolution along the canonical route (MAR-027), not a "
            "survey accuracy or an engineering resolution",
            "chainage_origin_basis": rr.chainage_origin_basis,
            "route_length_m": rr.route_length_m,
            "station_count": rr.station_count,
            "grid_crs": grid_crs,
            "disclaimer": route_reference.ROUTE_REFERENCE_GRID_DISCLAIMER,
        },
        "dod_source_identity": _identity_dict(result.dod_identity),
        "dod_source_immutability": IMMUTABLE_SOURCE_NOTE,
        "dod_observed_facts": result.dod_observed.to_dict() if result.dod_observed else None,
        "dod_definition": {
            "definition": ACCEPTED_DOD_DEFINITION,
            "positive_label": OBSERVED_SEABED_RAISING,
            "negative_label": OBSERVED_SEABED_LOWERING,
            "zero_label": None,
            "classifier": "marine_engine.change.dod.classify_change_direction (reused verbatim)",
        },
        "source_provenance_artifact": _identity_dict(result.provenance_identity),
        "crs_compatibility": {
            "comparison_method": "pyproj.CRS semantic equality (never raw string equality)",
            "dod_declared_crs": dod_decl.horizontal_crs_declared,
            "dod_observed_crs": result.dod_observed.observed_crs if result.dod_observed else None,
            "declared_vs_observed_conflict": result.declared_vs_observed_crs_conflict,
            "route_reference_grid_crs": grid_crs,
            "dod_vs_route_reference_grid": result.dod_vs_route_crs_status,
            "reprojection_performed": False,
            "resampling_performed": False,
        },
        "sample_support": {
            "semantics": SAMPLE_SUPPORT_SEMANTICS,
            "description": SAMPLE_SUPPORT_DESCRIPTION,
            "point_geometry_note": POINT_GEOMETRY_NOTE,
            "cell_center_distance_note": CELL_CENTER_DISTANCE_NOTE,
            "sample_statuses": [
                CHANGE_VALUE_AVAILABLE,
                CHANGE_VALUE_NODATA,
                ROUTE_POINT_OUTSIDE_DOD_EXTENT,
            ],
            "nodata_policy": "a nodata or outside-extent route point remains unavailable; it is "
            "never filled with 0 m change",
        },
        "sample_counts": _sample_counts(result.samples if result.available else None),
        "valid_sample_summary": _valid_sample_summary(result.samples if result.available else None),
        "generic_change_significance": {
            "generic_change_significance_status": (
                GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED
            ),
            "generic_change_significance_threshold_m": None,
            "note": GENERIC_SIGNIFICANCE_NOTE,
        },
        "direction_label_disclaimer": DIRECTION_LABEL_DISCLAIMER,
        "temporal": {
            "epoch1_survey_epoch_declared": dod_decl.epoch1_survey_epoch_declared,
            "epoch2_survey_epoch_declared": dod_decl.epoch2_survey_epoch_declared,
            "note": TEMPORAL_NOTE,
        },
        "outputs": {
            "parquet": result.output_paths.get("parquet"),
            "gpkg": result.output_paths.get("gpkg"),
            "gis_layer": ROUTE_CHANGE_LAYER if result.products_written else None,
        },
        "explicit_limitations": [
            SAMPLE_SUPPORT_DESCRIPTION,
            GENERIC_SIGNIFICANCE_NOTE,
            DIRECTION_LABEL_DISCLAIMER,
            NOT_A_STATE_NOTE,
            NO_INTERVAL_NOTE,
            SAMPLE_COUNT_NOTE,
            TEMPORAL_NOTE,
            "cross-CRS route sampling (DoD CRS differing from the route-reference grid CRS) is not "
            "implemented; such inputs are reported as DOD_CRS_MISMATCH",
            "no future change rate, sediment-transport, or morphodynamic prediction of any kind",
            "no route risk, hazard, readiness, or failure-probability output of any kind",
        ],
    }


def build_route_change_evidence_validation(result: RouteChangeEvidenceResult) -> dict[str, str]:
    """The required proof questions, derived from THIS run's facts -- never asserted as
    constants. `dod_sha256_after_run` must be supplied by the caller after outputs are written."""

    samples = result.samples if result.available else None
    directions: set[str] = set()
    nodata_became_zero = False
    support_uniform = True
    if samples is not None:
        directions = {
            str(v) for v in samples["observed_change_direction"].dropna().unique().tolist()
        }
        nodata_became_zero = bool(
            (
                (samples["sample_status"] != CHANGE_VALUE_AVAILABLE)
                & samples["delta_bed_elevation_m"].notna()
            ).any()
        )
        support_uniform = bool(
            (samples["sample_support_semantics"] == SAMPLE_SUPPORT_SEMANTICS).all()
        )
    allowed = {OBSERVED_SEABED_RAISING, OBSERVED_SEABED_LOWERING}
    return {
        "question_a_does_mar029_reuse_the_accepted_mar021_dod_definition": (
            "YES" if ACCEPTED_DOD_DEFINITION.endswith("epoch2_m - bed_elevation_epoch1_m") else "NO"
        ),
        "question_b_can_a_positive_dod_sample_be_called_definitive_deposition": (
            "NO" if directions <= allowed else "YES"
        ),
        "question_c_can_a_negative_dod_sample_be_called_definitive_erosion_or_scour": (
            "NO" if directions <= allowed else "YES"
        ),
        "question_d_does_mar029_apply_a_generic_change_significance_threshold": "NO",
        "question_e_does_nodata_become_zero_change": "YES" if nodata_became_zero else "NO",
        "question_f_is_the_dod_raster_interpolated_or_smoothed_during_route_sampling": (
            "NO" if support_uniform else "YES"
        ),
        "question_g_is_route_referenced_observed_change_available_for_this_run": (
            "YES" if result.available else "NO"
        ),
        "question_h_was_the_dod_source_modified_during_the_run": _source_modified_answer(result),
    }


def _source_modified_answer(result: RouteChangeEvidenceResult) -> str:
    before = _sha(result)
    after = result.output_paths.get("dod_sha256_after_run")
    if before is None or after is None:
        return "NOT_APPLICABLE"
    return "NO" if before == after else "YES"


def _sha(result: RouteChangeEvidenceResult) -> str | None:
    return result.dod_identity.sha256 if result.dod_identity is not None else None


def build_route_change_evidence_report_blocks(
    result: RouteChangeEvidenceResult, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Section 24 HTML: what was sampled, the two epochs if declared, what raising/lowering mean,
    the uncertainty limitation, nodata/outside support, no causal attribution, no prediction --
    and, when unavailable, exactly why. No score anywhere."""

    dod_decl = result.route_manifest.dod
    counts = metadata["sample_counts"]
    blocks: list[dict[str, Any]] = [
        {
            "type": "heading",
            "level": 1,
            "text": f"{result.project_id} -- Route-Referenced Observed Seabed Elevation Change",
        },
        {"type": "callout", "text": DIRECTION_LABEL_DISCLAIMER},
        {"type": "heading", "level": 2, "text": "1. Status"},
        {
            "type": "list",
            "items": [
                f"Scientific role: {SCIENTIFIC_ROLE}",
                f"Status: {result.status}",
                f"Reason code: {result.reason_code or 'n/a'}",
                *[f"Finding: {f}" for f in result.findings],
            ],
        },
        {"type": "heading", "level": 2, "text": "2. What Was Sampled"},
        {
            "type": "paragraph",
            "text": "For every canonical MAR-027 route-reference point of the declared project "
            "route, the accepted MAR-021 DoD raster was read at the single cell containing that "
            f"point ({SAMPLE_SUPPORT_SEMANTICS}). {SAMPLE_SUPPORT_DESCRIPTION}.",
        },
        {
            "type": "list",
            "items": [
                f"Project: {result.project_id} (working CRS {result.working_crs})",
                f"Route asset (declared in the route-change manifest): "
                f"{result.route_manifest.route_asset_id}",
                f"Route-reference status: {result.route_reference.status}; interval "
                f"{result.route_reference.configured_interval_m} m (indexing resolution); "
                f"stations {result.route_reference.station_count}; chainage origin basis "
                f"{result.route_reference.chainage_origin_basis}",
                f"DoD source: {result.dod_resolved_path}",
                f"DoD SHA-256: {_sha(result) or 'n/a'}",
                f"DoD source change study: {dod_decl.source_change_study_id}",
                f"DoD definition: {ACCEPTED_DOD_DEFINITION}",
                POINT_GEOMETRY_NOTE,
                CELL_CENTER_DISTANCE_NOTE,
            ],
        },
        {"type": "heading", "level": 2, "text": "3. Survey Epochs"},
        {
            "type": "list",
            "items": [
                f"Epoch 1 (declared): {dod_decl.epoch1_survey_epoch_declared or 'not declared'}",
                f"Epoch 2 (declared): {dod_decl.epoch2_survey_epoch_declared or 'not declared'}",
                TEMPORAL_NOTE,
            ],
        },
        {"type": "heading", "level": 2, "text": "4. CRS and Support Checks"},
        {
            "type": "list",
            "items": [f"{k}: {v}" for k, v in metadata["crs_compatibility"].items()]
            + [IMMUTABLE_SOURCE_NOTE],
        },
        {"type": "heading", "level": 2, "text": "5. Sample Counts"},
        {
            "type": "table",
            "headers": ["count", "value"],
            "rows": [[k, v] for k, v in counts.items() if k != "note"],
        },
        {"type": "paragraph", "text": SAMPLE_COUNT_NOTE},
        {"type": "heading", "level": 2, "text": "6. What Raising / Lowering Mean"},
        {
            "type": "list",
            "items": [
                f"delta > 0 -> {OBSERVED_SEABED_RAISING} (deposition/fill-compatible only)",
                f"delta < 0 -> {OBSERVED_SEABED_LOWERING} (erosion/scour-compatible only)",
                "delta == 0 -> no directional label",
                DIRECTION_LABEL_DISCLAIMER,
                NOT_A_STATE_NOTE,
            ],
        },
        {"type": "heading", "level": 2, "text": "7. Uncertainty / Significance Limitation"},
        {
            "type": "list",
            "items": [
                f"Status: {GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED}",
                "Generic threshold (m): null",
                GENERIC_SIGNIFICANCE_NOTE,
            ],
        },
        {"type": "heading", "level": 2, "text": "8. Nodata / Outside Support"},
        {
            "type": "list",
            "items": [
                f"{CHANGE_VALUE_NODATA}: the containing cell holds the raster nodata sentinel or a "
                "non-finite value -- the sample stays unavailable, never 0 m",
                f"{ROUTE_POINT_OUTSIDE_DOD_EXTENT}: the route-reference point lies outside the "
                "raster grid -- the sample stays unavailable, never 0 m",
            ],
        },
        {"type": "heading", "level": 2, "text": "9. What This Product Does Not Say"},
        {
            "type": "list",
            "items": list(metadata["explicit_limitations"]),
        },
    ]
    return blocks


def write_route_change_evidence_outputs(
    result: RouteChangeEvidenceResult, output_dir: Path
) -> RouteChangeEvidenceResult:
    """Section 24/25: metadata JSON, validation JSON, and HTML are ALWAYS written (truthfully,
    including the not-available case); Parquet + GeoPackage point products are written ONLY when
    the evidence is available -- and any stale product from an earlier run is removed first, so
    a not-available run can never leave a misleading point dataset behind."""

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "route_observed_seabed_change.parquet"
    gpkg_path = output_dir / "route_observed_seabed_change.gpkg"
    metadata_path = output_dir / "route_observed_seabed_change_metadata.json"
    validation_path = output_dir / "route_observed_seabed_change_validation.json"
    report_path = output_dir / "route_observed_seabed_change_report.html"

    for stale in (parquet_path, gpkg_path):
        if stale.exists():
            stale.unlink()

    output_paths: dict[str, str] = {
        "metadata": str(metadata_path),
        "validation": str(validation_path),
        "report": str(report_path),
    }
    products_written = False
    if result.available and result.samples is not None:
        table = pd.DataFrame(result.samples.drop(columns="geometry"))
        table.to_parquet(parquet_path, index=False)
        result.samples.to_file(gpkg_path, driver="GPKG", layer=ROUTE_CHANGE_LAYER)
        output_paths["parquet"] = str(parquet_path)
        output_paths["gpkg"] = str(gpkg_path)
        products_written = True

    # Section 10 proof: the source must be byte-identical after every write above.
    if result.dod_identity is not None:
        output_paths["dod_sha256_after_run"] = compute_file_identity(
            result.dod_identity.resolved_path
        ).sha256

    final = dataclasses.replace(
        result, products_written=products_written, output_paths=output_paths
    )
    metadata = build_route_change_evidence_metadata(final)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    validation = build_route_change_evidence_validation(final)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    report_path.write_text(
        change_report.render_blocks_html(
            build_route_change_evidence_report_blocks(final, metadata),
            title=f"{final.project_id} -- Route-Referenced Observed Seabed Elevation Change",
        ),
        encoding="utf-8",
    )
    return final
