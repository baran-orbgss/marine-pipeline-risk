"""Generic epoch-compatibility gates (MAR-021 Sections 6 and 8).

Zero dependency on any specific project or dataset. Two independent gates:

1. The HARD vertical-datum gate (Section 6): mandatory, categorical, and
   evidence-based ONLY -- never inferred from the data itself (Section 6
   is explicit: "Do NOT estimate a datum offset from the median difference
   and call that harmonization"). This module never even sees a DoD array;
   it only ever compares datum LABELS and (optionally) a caller-supplied
   evidence string documenting a real, source-stated equivalence.

2. The horizontal/grid compatibility classification (Section 8): compares
   two rasters' CRS, pixel spacing, rotation, and origin to classify how
   (or whether) they can be brought onto a common grid, preferring exact
   cell alignment or integer-pixel cropping over any resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

VERTICAL_DATUM_HARMONIZED = "VERTICAL_DATUM_HARMONIZED"
VERTICAL_DATUM_NOT_HARMONIZED = "VERTICAL_DATUM_NOT_HARMONIZED"

EXACT_GRID_ALIGNMENT = "EXACT_GRID_ALIGNMENT"
INTEGER_PIXEL_OFFSET_ALIGNMENT = "INTEGER_PIXEL_OFFSET_ALIGNMENT"
RESAMPLING_REQUIRED = "RESAMPLING_REQUIRED"
INCOMPATIBLE_HORIZONTAL_REFERENCE = "INCOMPATIBLE_HORIZONTAL_REFERENCE"

# A sub-pixel/rotation tolerance for floating-point affine comparisons -- NOT a scientific
# threshold, purely a numerical-noise guard (real GeoTIFF affines are rarely bit-exact even when
# conceptually "the same" grid).
_COORDINATE_TOLERANCE_M = 1e-6


@dataclass(frozen=True)
class VerticalDatumCompatibilityResult:
    status: str
    epoch1_datum: str | None
    epoch2_datum: str | None
    evidence: str | None
    reason: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "epoch1_datum": self.epoch1_datum,
            "epoch2_datum": self.epoch2_datum,
            "evidence": self.evidence,
            "reason": self.reason,
        }


def assess_vertical_datum_compatibility(
    epoch1_datum: str | None,
    epoch2_datum: str | None,
    *,
    harmonization_evidence: str | None = None,
) -> VerticalDatumCompatibilityResult:
    """The MANDATORY hard gate (Section 6). Purely categorical/evidence-
    based -- takes no elevation data of any kind, so it is structurally
    impossible for this function to derive "harmonization" from a
    computed difference.

    Passes only if:
      (a) both datum labels are known and identical (trivial case), OR
      (b) both labels are known, differ, AND the caller supplies a real,
          non-empty `harmonization_evidence` string documenting a
          defensible source-stated transformation/equivalence (e.g. a
          quoted survey-report statement) -- never inferred, never
          auto-generated, never merely "both surveys are the same site".

    Any other case (either datum unknown, or differing datums with no
    supplied evidence) fails with VERTICAL_DATUM_NOT_HARMONIZED.
    """

    if epoch1_datum is None or epoch2_datum is None:
        return VerticalDatumCompatibilityResult(
            status=VERTICAL_DATUM_NOT_HARMONIZED,
            epoch1_datum=epoch1_datum,
            epoch2_datum=epoch2_datum,
            evidence=harmonization_evidence,
            reason="one or both epochs have an unknown vertical datum -- cannot demonstrate "
            "compatibility",
        )

    if epoch1_datum == epoch2_datum:
        return VerticalDatumCompatibilityResult(
            status=VERTICAL_DATUM_HARMONIZED,
            epoch1_datum=epoch1_datum,
            epoch2_datum=epoch2_datum,
            evidence=harmonization_evidence,
            reason=f"both epochs share the identical stated vertical datum: {epoch1_datum!r}",
        )

    if harmonization_evidence:
        return VerticalDatumCompatibilityResult(
            status=VERTICAL_DATUM_HARMONIZED,
            epoch1_datum=epoch1_datum,
            epoch2_datum=epoch2_datum,
            evidence=harmonization_evidence,
            reason="datum labels differ but a source-stated transformation/equivalence was "
            "supplied as explicit evidence",
        )

    return VerticalDatumCompatibilityResult(
        status=VERTICAL_DATUM_NOT_HARMONIZED,
        epoch1_datum=epoch1_datum,
        epoch2_datum=epoch2_datum,
        evidence=None,
        reason=f"datum labels differ ({epoch1_datum!r} vs {epoch2_datum!r}) and no source-stated "
        "transformation/equivalence evidence was supplied",
    )


@dataclass(frozen=True)
class GridCompatibilityResult:
    status: str
    row_offset: float | None  # rows: epoch2 - epoch1, in PIXELS (positive = epoch2 grid starts
    col_offset: float | None  # further south/east); None when CRS/rotation are incompatible.
    reason: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "row_offset_px": self.row_offset,
            "col_offset_px": self.col_offset,
            "reason": self.reason,
        }


def classify_grid_alignment(
    *,
    crs1: str,
    transform1,
    crs2: str,
    transform2,
) -> GridCompatibilityResult:
    """Classifies how two rasters' grids relate, preferring exact
    alignment or integer-pixel cropping over any resampling (Section 8)."""

    if crs1 != crs2:
        return GridCompatibilityResult(
            status=INCOMPATIBLE_HORIZONTAL_REFERENCE,
            row_offset=None,
            col_offset=None,
            reason=f"CRS differs: epoch1={crs1!r}, epoch2={crs2!r}",
        )

    rotation_present = any(
        abs(t.b) > _COORDINATE_TOLERANCE_M or abs(t.d) > _COORDINATE_TOLERANCE_M
        for t in (transform1, transform2)
    )
    if rotation_present:
        return GridCompatibilityResult(
            status=INCOMPATIBLE_HORIZONTAL_REFERENCE,
            row_offset=None,
            col_offset=None,
            reason="one or both rasters carry a rotated affine transform -- not supported by "
            "this generic engine's integer-pixel/no-rotation assumption",
        )

    pixel_size_matches = (
        abs(abs(transform1.a) - abs(transform2.a)) < _COORDINATE_TOLERANCE_M
        and abs(abs(transform1.e) - abs(transform2.e)) < _COORDINATE_TOLERANCE_M
    )
    if not pixel_size_matches:
        return GridCompatibilityResult(
            status=RESAMPLING_REQUIRED,
            row_offset=None,
            col_offset=None,
            reason=f"pixel size differs: epoch1=({transform1.a:g},{transform1.e:g}), "
            f"epoch2=({transform2.a:g},{transform2.e:g})",
        )

    cell_size = abs(transform1.a)
    col_offset = (transform2.c - transform1.c) / cell_size
    row_offset = (transform1.f - transform2.f) / cell_size  # north-up: row 0 is the max northing

    col_is_integer = abs(col_offset - round(col_offset)) < 1e-3
    row_is_integer = abs(row_offset - round(row_offset)) < 1e-3
    if not (col_is_integer and row_is_integer):
        return GridCompatibilityResult(
            status=RESAMPLING_REQUIRED,
            row_offset=row_offset,
            col_offset=col_offset,
            reason=f"grid origins are offset by a FRACTIONAL pixel amount "
            f"(row={row_offset:.4f}, col={col_offset:.4f} px) -- cannot crop-align without "
            "resampling",
        )

    row_offset, col_offset = round(row_offset), round(col_offset)
    if row_offset == 0 and col_offset == 0:
        return GridCompatibilityResult(
            status=EXACT_GRID_ALIGNMENT,
            row_offset=0,
            col_offset=0,
            reason="identical origin and pixel size -- the two rasters already share the same grid",
        )

    return GridCompatibilityResult(
        status=INTEGER_PIXEL_OFFSET_ALIGNMENT,
        row_offset=row_offset,
        col_offset=col_offset,
        reason=f"grids share pixel size and CRS, offset by a whole number of pixels "
        f"(row={row_offset}, col={col_offset}) -- alignable by direct cropping, no "
        "interpolation required",
    )
