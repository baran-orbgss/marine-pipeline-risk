"""Layer catalog assembly + GeoJSON/tile URL wiring.

A project's layer catalog comes from two independent sources:

1. **Registered assets** -- files declared in the project's `ProjectManifest` (if it has one).
   These have an established semantic role because a human/operator declared it explicitly
   (`asset.category`); `role_established` is always True here.
2. **Known output files** -- a small, curated table (`_OUTPUT_RULES`) of glob patterns matched
   relative to the project's output root, one entry per accepted engine output family actually
   verified to exist on disk for at least one of the example projects. A file matching no rule is
   never silently displayed. A file already covered by a registered asset is not duplicated.

This module does not try to enumerate every possible engine output -- only output families this
ticket's manual-acceptance scenarios require, plus a few obvious siblings. Anything not covered
stays inspectable through the Engineering Workbench instead of appearing as a map layer; that is a
deliberate, reported scope limitation, not a silent gap.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import rasterio

from api import display_specs as ds
from api import settings
from api.discovery import (
    RASTER_EXTENSIONS,
    bounds_to_wgs84,
    inspect_file,
    reconstruct_known_tabular_geometry,
)
from api.models import (
    BoundingBox,
    DisplaySpec,
    LayerCatalog,
    LayerDescriptor,
    LayerGroup,
    SupportType,
    TooltipField,
)
from api.paths import repo_relative
from api.projects import get_project_summary
from marine_engine.project import categories as cat
from marine_engine.project.manifest import load_project_manifest, resolve_asset_path

_ASSET_CATEGORY_SUPPORT: dict[str, SupportType] = {
    cat.PIPELINE_ROUTE: "LINEAR_ASSET",
    cat.BATHYMETRY_RASTER: "AREA_SURFACE",
    cat.BURIAL_PROFILE: "LINEAR_ANALYSIS",
    cat.CPT: "POINT_EVIDENCE",
}
_ASSET_CATEGORY_GROUP: dict[str, LayerGroup] = {
    cat.PIPELINE_ROUTE: "ASSETS",
    cat.BATHYMETRY_RASTER: "DATA",
    cat.BURIAL_PROFILE: "EVIDENCE",
    cat.CPT: "EVIDENCE",
}


def _repo_rel(path: Path) -> str:
    return repo_relative(path, settings.REPO_ROOT)


def _raster_domain(path: Path, *, max_pixels: int = 512) -> tuple[float, float]:
    with rasterio.open(path) as src:
        scale = max(1, int(math.sqrt((src.width * src.height) / max_pixels**2)))
        out_shape = (max(1, src.height // scale), max(1, src.width // scale))
        data = src.read(1, out_shape=out_shape, masked=True)
        if data.count() == 0:
            return (0.0, 1.0)
        return (float(data.min()), float(data.max()))


def _url_segment(layer_id: str) -> str:
    """A `layer_id` routinely contains '/' and ':' (e.g. "proj:output:terrain/aspect.tif").
    Percent-encoding alone is not enough to embed it as a single opaque path segment: ASGI servers
    (uvicorn) percent-decode `scope["path"]` -- including a `%2F` back into a literal '/' -- before
    Starlette matches routes, so a percent-encoded '/' still splits the path into extra segments
    that no `{layer_id}` route matches, and on Windows the request then falls through to the
    static-file mount and fails with an invalid-path OSError (verified against a live server, not
    just a theory: this was observed happening for every output layer whose id embeds a nested
    path, e.g. "terrain/aspect.tif" or "bedforms/crest_match_candidates.parquet").

    Base64url-encoding it instead produces a token drawn only from `[A-Za-z0-9_-]`, which has no
    character that could ever decode (at any layer -- browser, proxy, ASGI) into '/' or ':'. The
    frontend always uses the server-provided URL template verbatim rather than rebuilding one from
    a raw `layer_id`, so this encoding only ever needs to be reversed by `decode_layer_id` below."""
    return base64.urlsafe_b64encode(layer_id.encode("utf-8")).decode("ascii").rstrip("=")


def decode_layer_id(token: str) -> str:
    """Reverse `_url_segment`. Raises `ValueError` for a malformed token (e.g. a hand-typed URL)."""
    padded = token + "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")


def _tile_urls(project_id: str, layer_id: str) -> tuple[str, str]:
    base = f"/api/projects/{quote(project_id, safe='')}/layers/{_url_segment(layer_id)}"
    return f"{base}/tiles/{{z}}/{{x}}/{{y}}.png", f"{base}/tilejson.json"


def _vector_url(project_id: str, layer_id: str) -> str:
    return f"/api/projects/{quote(project_id, safe='')}/layers/{_url_segment(layer_id)}/features"


def _cpt_profile_url_template(project_id: str, layer_id: str) -> str:
    base = f"/api/projects/{quote(project_id, safe='')}/layers/{_url_segment(layer_id)}"
    return f"{base}/cpt-tests/{{test_id}}/profile"


def _raster_descriptor(
    *,
    project_id: str,
    layer_id: str,
    group: LayerGroup,
    support_type: SupportType,
    capability_key: str | None,
    semantic_role: str | None,
    role_established: bool,
    path: Path,
    display: DisplaySpec,
) -> LayerDescriptor | None:
    try:
        inspection = inspect_file(path)
    except Exception:
        return None
    tile_template, tilejson_url = _tile_urls(project_id, layer_id)
    return LayerDescriptor(
        layer_id=layer_id,
        project_id=project_id,
        group=group,
        capability_key=capability_key,
        layer_type="raster",
        support_type=support_type,
        semantic_role=semantic_role,
        role_established=role_established,
        relative_path=_repo_rel(path),
        crs_observed=inspection.observed_crs,
        bounds_native=inspection.bounds_native,
        bounds_wgs84=inspection.bounds_wgs84,
        display=display,
        tile_url_template=tile_template,
        tilejson_url=tilejson_url,
        band_count=inspection.band_count,
    )


def _vector_descriptor(
    *,
    project_id: str,
    layer_id: str,
    group: LayerGroup,
    support_type: SupportType,
    capability_key: str | None,
    semantic_role: str | None,
    role_established: bool,
    path: Path,
    display: DisplaySpec,
    layer: str | None = None,
) -> LayerDescriptor | None:
    try:
        inspection = inspect_file(path, layer=layer)
    except Exception:
        return None
    if inspection.kind != "vector":
        return None
    return LayerDescriptor(
        layer_id=layer_id,
        project_id=project_id,
        group=group,
        capability_key=capability_key,
        layer_type="vector",
        support_type=support_type,
        semantic_role=semantic_role,
        role_established=role_established,
        relative_path=_repo_rel(path),
        gpkg_layer=layer,
        crs_observed=inspection.observed_crs,
        bounds_native=inspection.bounds_native,
        bounds_wgs84=inspection.bounds_wgs84,
        display=display,
        vector_url=_vector_url(project_id, layer_id),
        cpt_profile_url_template=_cpt_profile_url_template(project_id, layer_id),
        feature_count=inspection.feature_count,
    )


def _asset_layers(
    project_id: str, output_root: Path | None
) -> tuple[list[LayerDescriptor], set[Path]]:
    """Layers declared by a ProjectManifest -- role_established is always True here because a
    human declared the category. Returns the layers plus the set of resolved file paths they
    cover, so output-file discovery below never double-registers the same file."""
    manifest_path = None
    if output_root is not None:
        candidate = output_root / "project" / settings.AD_HOC_MANIFEST_FILENAME
        if candidate.is_file():
            manifest_path = candidate
    if manifest_path is None:
        for candidate in settings.PROJECT_MANIFESTS_DIR.glob("*.yaml"):
            try:
                manifest, _ = load_project_manifest(candidate)
            except Exception:
                continue
            if manifest.project.id.strip().lower() == project_id:
                manifest_path = candidate
                break
    if manifest_path is None:
        return [], set()

    manifest, manifest_dir = load_project_manifest(manifest_path)
    layers: list[LayerDescriptor] = []
    covered: set[Path] = set()
    for asset in manifest.assets:
        try:
            path = resolve_asset_path(asset, manifest_dir)
        except Exception:
            continue
        if not path.is_file():
            continue
        covered.add(path.resolve())
        layer_id = f"{project_id}:asset:{asset.asset_id}"
        is_raster = path.suffix.lower() in RASTER_EXTENSIONS
        support_type = _ASSET_CATEGORY_SUPPORT.get(
            asset.category, "AREA_SURFACE" if is_raster else "AREA_VECTOR"
        )
        group = _ASSET_CATEGORY_GROUP.get(asset.category, "DATA")

        if asset.category == cat.BATHYMETRY_RASTER and is_raster:
            display = ds.bathymetry_display_spec(_raster_domain(path))
        elif asset.category == cat.PIPELINE_ROUTE:
            display = ds.asset_display_spec("Pipeline / cable route")
        elif asset.category == cat.CPT:
            display = ds.point_evidence_display_spec(
                "CPT test locations", [TooltipField(key="test_id", label="Test ID")]
            )
        elif asset.category == cat.BURIAL_PROFILE:
            display = ds.route_analysis_display_spec(
                "Burial profile evidence",
                [TooltipField(key="chainage_m", label="Chainage", unit="m")],
            )
        else:
            display = ds.unclassified_display_spec(asset.asset_id)

        descriptor = (
            _raster_descriptor(
                project_id=project_id,
                layer_id=layer_id,
                group=group,
                support_type=support_type,
                capability_key=None,
                semantic_role=asset.category,
                role_established=True,
                path=path,
                display=display,
            )
            if is_raster
            else _vector_descriptor(
                project_id=project_id,
                layer_id=layer_id,
                group=group,
                support_type=support_type,
                capability_key=None,
                semantic_role=asset.category,
                role_established=True,
                path=path,
                display=display,
                layer=asset.layer,
            )
        )
        if descriptor is not None:
            layers.append(descriptor)
    return layers, covered


def _terrain_spec(path: Path) -> DisplaySpec:
    stem = path.stem.lower()
    domain = _raster_domain(path)
    if "hillshade" in stem:
        return ds.hillshade_display_spec()
    if any(token in stem for token in ("elevation", "bathy", "depth")):
        return ds.bathymetry_display_spec(domain)
    unit = "deg" if any(token in stem for token in ("aspect", "slope")) else ""
    return ds.terrain_derivative_display_spec(path.stem.replace("_", " ").title(), unit, domain)


_KP_FIELDS = [
    TooltipField(key="kp_start", label="KP start"),
    TooltipField(key="kp_end", label="KP end"),
]


@dataclass(frozen=True)
class _OutputRule:
    capability_key: str | None
    group: LayerGroup
    support_type: SupportType
    layer_type: str  # "raster" | "vector"
    relative_glob: str
    display_name: str | Callable[[Path], str]
    spec_factory: Callable[[Path], DisplaySpec]
    # For a multi-layer GeoPackage: which named layer actually holds the data this rule wants
    # (verified per file -- geopandas/pyogrio otherwise silently reads whichever layer the
    # container lists first, which is not always the meaningful one).
    gpkg_layer: str | None = None


_OUTPUT_RULES: list[_OutputRule] = [
    _OutputRule(
        "terrain",
        "DERIVED",
        "AREA_SURFACE",
        "raster",
        "terrain/*.tif",
        lambda p: p.stem.replace("_", " ").title(),
        _terrain_spec,
    ),
    _OutputRule(
        "erosion_deposition",
        "DERIVED",
        "AREA_SURFACE",
        "raster",
        "change/*.tif",
        lambda p: p.stem.replace("_", " ").title(),
        lambda p: ds.seabed_change_display_spec(_raster_domain(p)),
    ),
    _OutputRule(
        "sediment_mobility",
        "ANALYSIS",
        "LINEAR_ANALYSIS",
        "vector",
        "sediment/*mobility*segments.gpkg",
        "Sediment mobility capacity",
        lambda p: ds.route_analysis_display_spec(
            "Sediment mobility capacity",
            [
                *_KP_FIELDS,
                TooltipField(
                    key="largest_tested_d50_with_p95_mobility_ratio_ge_1_mm",
                    label="Largest mobile D50 (P95)",
                    unit="mm",
                ),
            ],
        ),
    ),
    _OutputRule(
        "sediment_mobility",
        "ANALYSIS",
        "LINEAR_ANALYSIS",
        "vector",
        "sediment/*transport_intensity*segments.gpkg",
        "Sediment transport intensity",
        lambda p: ds.route_analysis_display_spec(
            "Sediment transport intensity",
            [
                *_KP_FIELDS,
                TooltipField(
                    key="relative_excess_intensity_p95", label="Relative excess intensity (P95)"
                ),
            ],
        ),
    ),
    _OutputRule(
        "scour",
        "ANALYSIS",
        "LINEAR_ANALYSIS",
        "vector",
        "scour/pipeline_scour_screening.gpkg",
        "Scour susceptibility",
        lambda p: ds.route_analysis_display_spec(
            "Scour susceptibility",
            [*_KP_FIELDS, TooltipField(key="screening_state", label="Screening state")],
        ),
        gpkg_layer="site_specific_susceptibility_screening",
    ),
    _OutputRule(
        "scour",
        "ANALYSIS",
        "LINEAR_ANALYSIS",
        "vector",
        "scour/*embedment_segments.gpkg",
        "Scour onset embedment",
        lambda p: ds.route_analysis_display_spec(
            "Scour onset embedment",
            [
                *_KP_FIELDS,
                TooltipField(
                    key="p95_required_embedment_lower_m",
                    label="Required embedment, lower (P95)",
                    unit="m",
                ),
                TooltipField(
                    key="p95_required_embedment_upper_m",
                    label="Required embedment, upper (P95)",
                    unit="m",
                ),
            ],
        ),
    ),
    _OutputRule(
        "burial_exposure",
        "EVIDENCE",
        "POINT_EVIDENCE",
        "vector",
        "burial/burial_exposure_poc.gpkg",
        "Burial / exposure screening",
        lambda p: ds.point_evidence_display_spec(
            "Burial / exposure screening",
            [
                TooltipField(key="chainage_m", label="Chainage", unit="m"),
                TooltipField(key="measured_burial_state", label="Measured burial state"),
                TooltipField(key="cover_above_asset_m", label="Cover above asset", unit="m"),
            ],
        ),
        gpkg_layer="source_interpreted_exposure",
    ),
    _OutputRule(
        None,
        "ASSETS",
        "LINEAR_ASSET",
        "vector",
        "burial/canonical_asset_route.gpkg",
        "Asset route",
        lambda p: ds.asset_display_spec("Asset route"),
    ),
    _OutputRule(
        None,
        "EVIDENCE",
        "LINEAR_ANALYSIS",
        "vector",
        "freespan_evidence/*freespan_spatial_evidence.gpkg",
        "Free span evidence",
        lambda p: ds.route_analysis_display_spec("Free span evidence", _KP_FIELDS),
    ),
    _OutputRule(
        None,
        "EVIDENCE",
        "POINT_EVIDENCE",
        "vector",
        "pipeline_condition/*freespan_registry.gpkg",
        "Observed free spans (NSTA registry)",
        lambda p: ds.point_evidence_display_spec(
            "Observed free spans", [TooltipField(key="kp", label="KP")]
        ),
    ),
    _OutputRule(
        None,
        "DATA",
        "CORRIDOR",
        "vector",
        "aoi.gpkg",
        "Analysis corridor (AOI)",
        lambda p: ds.area_vector_display_spec("Analysis corridor (AOI)"),
    ),
    _OutputRule(
        None,
        "DATA",
        "SUPPORT_NODE",
        "vector",
        "chainage_25m.gpkg",
        "Chainage reference points",
        lambda p: ds.point_evidence_display_spec(
            "Chainage reference points",
            [TooltipField(key="chainage_m", label="Chainage", unit="m")],
        ),
    ),
    # Sheringham Shoal 2008 CPTU has no ProjectManifest; the engine's own canonical CPT output
    # shape (cpt_locations.gpkg at the output root) is the reliable, capability-defined signal.
    _OutputRule(
        "cpt_evidence",
        "EVIDENCE",
        "POINT_EVIDENCE",
        "vector",
        "cpt_locations.gpkg",
        "CPT test locations",
        lambda p: ds.point_evidence_display_spec(
            "CPT test locations", [TooltipField(key="test_id", label="Test ID")]
        ),
    ),
]


def _matched_display_name(rule: _OutputRule, path: Path) -> str:
    return rule.display_name(path) if callable(rule.display_name) else rule.display_name


def _output_layers(project_id: str, output_root: Path, covered: set[Path]) -> list[LayerDescriptor]:
    layers: list[LayerDescriptor] = []
    seen_ids: set[str] = set()
    for rule in _OUTPUT_RULES:
        for path in sorted(output_root.glob(rule.relative_glob)):
            if not path.is_file() or path.resolve() in covered:
                continue
            layer_id = f"{project_id}:output:{path.relative_to(output_root).as_posix()}"
            if layer_id in seen_ids:
                continue
            try:
                display = rule.spec_factory(path)
            except Exception:
                continue
            display.display_name = _matched_display_name(rule, path)
            descriptor = (
                _raster_descriptor(
                    project_id=project_id,
                    layer_id=layer_id,
                    group=rule.group,
                    support_type=rule.support_type,
                    capability_key=rule.capability_key,
                    semantic_role=rule.capability_key,
                    role_established=True,
                    path=path,
                    display=display,
                )
                if rule.layer_type == "raster"
                else _vector_descriptor(
                    project_id=project_id,
                    layer_id=layer_id,
                    group=rule.group,
                    support_type=rule.support_type,
                    capability_key=rule.capability_key,
                    semantic_role=rule.capability_key,
                    role_established=True,
                    path=path,
                    display=display,
                    layer=rule.gpkg_layer,
                )
            )
            if descriptor is not None:
                seen_ids.add(layer_id)
                layers.append(descriptor)
    return layers


def _bedform_crest_layer(
    project_id: str, output_root: Path, working_crs: str | None
) -> LayerDescriptor | None:
    """`bedforms/crest_match_candidates.parquet` is plain tabular parquet, not geoparquet -- but
    its `epoch1_x_m`/`epoch1_y_m` columns are a known, verified property of
    `build-bedform-morphodynamics-poc`'s output (see
    `api.discovery.reconstruct_known_tabular_geometry`), unlike the along-transect
    `crest_position_m`-style columns in the other bedform tables, which are not map coordinates
    and are intentionally not surfaced as a layer in this release."""
    path = output_root / "bedforms" / "crest_match_candidates.parquet"
    if not path.is_file() or working_crs is None:
        return None
    try:
        gdf = reconstruct_known_tabular_geometry(path, working_crs)
    except Exception:
        return None
    minx, miny, maxx, maxy = gdf.total_bounds
    layer_id = f"{project_id}:output:bedforms/crest_match_candidates.parquet"
    return LayerDescriptor(
        layer_id=layer_id,
        project_id=project_id,
        group="DERIVED",
        capability_key="bedforms",
        layer_type="vector",
        support_type="AREA_VECTOR",
        semantic_role="bedforms",
        role_established=True,
        relative_path=_repo_rel(path),
        crs_observed=working_crs,
        bounds_native=BoundingBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=working_crs),
        bounds_wgs84=bounds_to_wgs84(minx, miny, maxx, maxy, working_crs),
        display=ds.bedform_display_spec(
            "Bedforms (matched crests, 2018-2020)",
            [
                TooltipField(key="match_status", label="Match status"),
                TooltipField(key="normal_displacement_m", label="Migration distance", unit="m"),
                TooltipField(key="apparent_rate_m_per_year", label="Migration rate", unit="m/yr"),
                TooltipField(key="wavelength_1_m", label="Wavelength", unit="m"),
            ],
        ),
        vector_url=_vector_url(project_id, layer_id),
        feature_count=len(gdf),
    )


def build_layer_catalog(project_id: str) -> LayerCatalog:
    summary = get_project_summary(project_id)
    if summary is None:
        return LayerCatalog(project_id=project_id, layers=[])
    output_root = settings.REPO_ROOT / summary.output_root if summary.output_root else None

    asset_layers, covered = _asset_layers(project_id, output_root)
    output_layers: list[LayerDescriptor] = []
    if output_root and output_root.is_dir():
        output_layers = _output_layers(project_id, output_root, covered)
        bedform_layer = _bedform_crest_layer(project_id, output_root, summary.working_crs)
        if bedform_layer is not None:
            output_layers.append(bedform_layer)
    return LayerCatalog(project_id=project_id, layers=[*asset_layers, *output_layers])


def get_layer(project_id: str, layer_id: str) -> LayerDescriptor | None:
    catalog = build_layer_catalog(project_id)
    for layer in catalog.layers:
        if layer.layer_id == layer_id:
            return layer
    return None


def catalog_extent_wgs84(catalog: LayerCatalog) -> BoundingBox | None:
    boxes = [layer.bounds_wgs84 for layer in catalog.layers if layer.bounds_wgs84 is not None]
    if not boxes:
        return None
    return BoundingBox(
        minx=min(b.minx for b in boxes),
        miny=min(b.miny for b in boxes),
        maxx=max(b.maxx for b in boxes),
        maxy=max(b.maxy for b in boxes),
    )


def list_figure_paths(project_id: str) -> list[str]:
    """Existing scientific PNG figures, kept as reference material (ticket S40) -- never deleted,
    never the primary spatial UX. Returns paths relative to the project's own output root."""
    summary = get_project_summary(project_id)
    if summary is None or summary.output_root is None:
        return []
    output_root = settings.REPO_ROOT / summary.output_root
    if not output_root.is_dir():
        return []
    return sorted(path.relative_to(output_root).as_posix() for path in output_root.rglob("*.png"))
