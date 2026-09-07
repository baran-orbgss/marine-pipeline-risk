"""Generic, project-agnostic operator-style bathymetry data readiness (MAR-020 Section 5).

Zero dependency on any specific project or dataset. Every check takes
already-opened raster facts (never re-opens the file itself, so callers
control I/O) and returns an explicit pass/fail + severity + human-readable
reason -- never a numeric readiness score. Overall status is one of
READY / READY_WITH_LIMITATIONS / NOT_READY, derived purely from whether
any BLOCKING check failed (-> NOT_READY) or any LIMITATION-severity check
failed (-> READY_WITH_LIMITATIONS), else READY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

READY = "READY"
READY_WITH_LIMITATIONS = "READY_WITH_LIMITATIONS"
NOT_READY = "NOT_READY"

BLOCKING = "BLOCKING"
LIMITATION = "LIMITATION"

# --- Plausibility thresholds (named constants, never inline magic numbers) ------------------

MAX_PLAUSIBLE_ABS_ELEVATION_M = 11000.0  # generously beyond the Challenger Deep; a sanity guard
MIN_VALID_CELL_FRACTION_FOR_READY = 0.05  # below this, the raster is functionally empty
MIN_VALID_CELL_FRACTION_FOR_LIMITATION = 0.30  # below this (but above the floor), flag sparsity


@dataclass(frozen=True)
class ReadinessCheck:
    check_id: str
    passed: bool
    severity: str  # BLOCKING or LIMITATION -- only meaningful when passed=False
    detail: str


@dataclass(frozen=True)
class RasterFacts:
    """Plain facts about an already-opened raster -- callers extract these
    with their own I/O (e.g. rasterio) so this module never opens a file
    itself and stays trivially unit-testable with synthetic values."""

    band_count: int
    dtype: str
    color_interpretations: tuple[str, ...]
    crs_is_present: bool
    crs_is_geographic: bool | None
    crs_linear_units: str | None
    width: int
    height: int
    pixel_size_x_m: float | None
    pixel_size_y_m: float | None
    bounds: tuple[float, float, float, float] | None  # (minx, miny, maxx, maxy)
    nodata_value: float | None
    vertical_datum: str | None
    survey_epoch: str | None
    data_min: float | None
    data_max: float | None
    data_std: float | None
    valid_cell_fraction: float | None


@dataclass(frozen=True)
class ReadinessResult:
    status: str
    checks: tuple[ReadinessCheck, ...]

    def reasons(self) -> list[str]:
        return [f"{c.check_id} ({c.severity}): {c.detail}" for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": [
                {
                    "check_id": c.check_id,
                    "passed": c.passed,
                    "severity": c.severity if not c.passed else None,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "blocking_reasons": [
                c.detail for c in self.checks if not c.passed and c.severity == BLOCKING
            ],
            "limitation_reasons": [
                c.detail for c in self.checks if not c.passed and c.severity == LIMITATION
            ],
        }


def assess_bathymetry_readiness(facts: RasterFacts) -> ReadinessResult:
    """Run every operator-style readiness check against `facts` and derive
    an explicit status. Never a numeric score."""

    checks: list[ReadinessCheck] = []

    checks.append(
        ReadinessCheck(
            "file_readable", True, "", "raster opened successfully (caller-provided facts)"
        )
    )

    is_render = facts.band_count >= 3 and facts.dtype in ("uint8", "int8")
    checks.append(
        ReadinessCheck(
            "analytical_not_render",
            not is_render,
            BLOCKING,
            f"raster looks like an RGB/render product (bands={facts.band_count}, "
            f"dtype={facts.dtype}, colorinterp={facts.color_interpretations}), not analytical "
            "elevation/depth data"
            if is_render
            else f"single-band ({facts.band_count}), non-8-bit ({facts.dtype}) -- consistent with "
            "analytical elevation/depth data",
        )
    )

    checks.append(
        ReadinessCheck(
            "crs_present",
            facts.crs_is_present,
            BLOCKING,
            "no CRS found on the raster" if not facts.crs_is_present else "CRS is present",
        )
    )

    crs_ok = facts.crs_is_present and facts.crs_is_geographic is False
    checks.append(
        ReadinessCheck(
            "crs_projected_metric",
            crs_ok,
            BLOCKING,
            f"CRS is geographic (degrees) or unknown -- not a projected metric CRS "
            f"(is_geographic={facts.crs_is_geographic}, linear_units={facts.crs_linear_units})"
            if not crs_ok
            else f"projected CRS, linear units = {facts.crs_linear_units}",
        )
    )

    # bool(...) at the end: `np.isfinite` returns numpy.bool_, and an `and` chain returns
    # whichever operand it last evaluates -- left as numpy.bool_, `json.dumps(default=str)`
    # would serialize `passed` as the STRING "True"/"False" instead of a JSON boolean for this
    # one check, silently breaking any strict-typed consumer of this "machine-readable" report.
    resolution_known = bool(
        facts.pixel_size_x_m is not None
        and facts.pixel_size_y_m is not None
        and facts.pixel_size_x_m > 0
        and facts.pixel_size_y_m > 0
        and np.isfinite(facts.pixel_size_x_m)
        and np.isfinite(facts.pixel_size_y_m)
    )
    checks.append(
        ReadinessCheck(
            "native_resolution_known",
            resolution_known,
            BLOCKING,
            f"pixel size not a finite positive value: x={facts.pixel_size_x_m}, "
            f"y={facts.pixel_size_y_m}"
            if not resolution_known
            else f"native pixel size {facts.pixel_size_x_m} x {facts.pixel_size_y_m} m",
        )
    )

    extent_valid = (
        facts.bounds is not None
        and all(np.isfinite(v) for v in facts.bounds)
        and facts.width > 0
        and facts.height > 0
    )
    checks.append(
        ReadinessCheck(
            "spatial_extent_valid",
            extent_valid,
            BLOCKING,
            f"raster bounds/dimensions are not valid: bounds={facts.bounds}, "
            f"width={facts.width}, height={facts.height}"
            if not extent_valid
            else f"bounds={facts.bounds}, {facts.width}x{facts.height} px",
        )
    )

    coord_plausible = True
    if extent_valid and not facts.crs_is_geographic:
        minx, miny, maxx, maxy = facts.bounds
        # A projected CRS's coordinates should be on the order of tens of thousands to a few
        # million metres (UTM-style eastings/northings), never tiny (looks like degrees mistaken
        # for metres) or absurdly large (looks like a units/CRS mismatch).
        coord_plausible = all(1000.0 <= abs(v) <= 1.0e8 for v in (minx, miny, maxx, maxy) if v != 0)
    checks.append(
        ReadinessCheck(
            "coordinate_magnitude_plausible",
            coord_plausible,
            BLOCKING,
            f"raster bounds {facts.bounds} do not look like plausible projected-CRS coordinates"
            if not coord_plausible
            else "bounds magnitude consistent with a real projected CRS",
        )
    )

    checks.append(
        ReadinessCheck(
            "vertical_datum_known",
            facts.vertical_datum is not None,
            LIMITATION,
            "no vertical datum recorded for this raster (common for plain GeoTIFFs -- "
            "provenance/filename evidence should be recorded separately, never invented)"
            if facts.vertical_datum is None
            else f"vertical datum: {facts.vertical_datum}",
        )
    )

    checks.append(
        ReadinessCheck(
            "survey_epoch_known",
            facts.survey_epoch is not None,
            LIMITATION,
            "no survey epoch supplied (the raster itself rarely carries this -- must come from "
            "source provenance)"
            if facts.survey_epoch is None
            else f"survey epoch: {facts.survey_epoch}",
        )
    )

    checks.append(
        ReadinessCheck(
            "nodata_defined",
            facts.nodata_value is not None,
            LIMITATION,
            "no explicit nodata sentinel defined on the raster"
            if facts.nodata_value is None
            else f"nodata = {facts.nodata_value}",
        )
    )

    vcf = facts.valid_cell_fraction
    vcf_blocking = vcf is not None and vcf < MIN_VALID_CELL_FRACTION_FOR_READY
    checks.append(
        ReadinessCheck(
            "valid_cell_fraction_nonzero",
            not vcf_blocking,
            BLOCKING,
            f"only {vcf:.1%} of cells are valid -- functionally empty"
            if vcf_blocking
            else ("valid_cell_fraction unknown" if vcf is None else f"{vcf:.1%} valid cells"),
        )
    )
    vcf_sparse = (
        vcf is not None and not vcf_blocking and vcf < MIN_VALID_CELL_FRACTION_FOR_LIMITATION
    )
    checks.append(
        ReadinessCheck(
            "valid_cell_fraction_dense_enough",
            not vcf_sparse,
            LIMITATION,
            f"only {vcf:.1%} of cells are valid -- sparse coverage (a narrow survey swath within "
            "a larger bounding box is expected and not itself a defect)"
            if vcf_sparse
            else "valid-cell coverage is not unusually sparse",
        )
    )

    range_plausible = True
    if facts.data_min is not None and facts.data_max is not None:
        # bool(...): same numpy.bool_-leaks-into-JSON-as-a-string reasoning as resolution_known.
        range_plausible = bool(
            abs(facts.data_min) <= MAX_PLAUSIBLE_ABS_ELEVATION_M
            and abs(facts.data_max) <= MAX_PLAUSIBLE_ABS_ELEVATION_M
            and np.isfinite(facts.data_min)
            and np.isfinite(facts.data_max)
        )
    checks.append(
        ReadinessCheck(
            "data_range_plausible",
            range_plausible,
            BLOCKING,
            f"data range [{facts.data_min}, {facts.data_max}] is not physically plausible"
            if not range_plausible
            else f"data range [{facts.data_min}, {facts.data_max}] is physically plausible",
        )
    )

    is_constant = (
        facts.data_std is not None and facts.data_std == 0.0 and vcf is not None and vcf > 0.5
    )
    checks.append(
        ReadinessCheck(
            "not_corrupted_constant_sentinel",
            not is_constant,
            BLOCKING,
            "a large fraction of the raster is a single constant value -- consistent with a "
            "corrupted or placeholder file"
            if is_constant
            else f"data standard deviation {facts.data_std} is not a suspicious constant",
        )
    )

    checks_t = tuple(checks)
    if any(not c.passed and c.severity == BLOCKING for c in checks_t):
        status = NOT_READY
    elif any(not c.passed and c.severity == LIMITATION for c in checks_t):
        status = READY_WITH_LIMITATIONS
    else:
        status = READY
    return ReadinessResult(status=status, checks=checks_t)
