"""Curated display-specification table: output-kind -> full cartography spec.

Presentation metadata only (palette / legend / units / opacity / z-index / tooltip fields /
default visibility). It never computes, thresholds, or reinterprets a scientific value -- domains
come from the raster/vector's own observed statistics, passed in by the caller.

Z-index bands keep the same generic ordering for every project: area rasters and area/corridor
vectors sit lowest, route-only analysis lines above them, point evidence above that, and asset
(pipeline/cable) geometry always on top -- see `computeLayerOrder` in the frontend, which mirrors
this exactly.
"""

from __future__ import annotations

from api.models import DisplaySpec, LegendSpec, PaletteSpec, PaletteStop, TooltipField

Z_AREA_RASTER = 10
Z_AREA_VECTOR = 15
Z_ROUTE_ANALYSIS = 30
Z_POINT_EVIDENCE = 40
Z_ASSET = 50

_BATHYMETRY_STOPS = [
    PaletteStop(value=0.0, color="#08306b", label="deep"),
    PaletteStop(value=0.25, color="#2171b5"),
    PaletteStop(value=0.5, color="#6baed6"),
    PaletteStop(value=0.75, color="#c6dbef"),
    PaletteStop(value=1.0, color="#f7fbff", label="shallow"),
]

_DIVERGING_STOPS = [
    PaletteStop(value=-1.0, color="#67001f", label="erosion"),
    PaletteStop(value=0.0, color="#f7f7f7", label="no change"),
    PaletteStop(value=1.0, color="#053061", label="accretion"),
]

_TERRAIN_STOPS = [
    PaletteStop(value=0.0, color="#f7fcb9"),
    PaletteStop(value=0.5, color="#addd8e"),
    PaletteStop(value=1.0, color="#238443"),
]

_MOBILITY_STOPS = [
    PaletteStop(value=0.0, color="#fee8c8", label="low"),
    PaletteStop(value=0.5, color="#fdbb84"),
    PaletteStop(value=1.0, color="#e34a33", label="high"),
]


def _scaled_stops(stops: list[PaletteStop], domain: tuple[float, float]) -> list[PaletteStop]:
    lo, hi = domain
    span = hi - lo or 1.0
    return [
        PaletteStop(value=lo + stop.value * span, color=stop.color, label=stop.label)
        for stop in stops
    ]


def bathymetry_display_spec(domain: tuple[float, float]) -> DisplaySpec:
    stops = _scaled_stops(_BATHYMETRY_STOPS, domain)
    return DisplaySpec(
        display_name="Bathymetry",
        palette=PaletteSpec(
            kind="continuous", colormap_name="marine_depth", domain=domain, stops=stops
        ),
        legend=LegendSpec(title="Seabed elevation", unit="m", kind="continuous", stops=stops),
        units="m",
        opacity=0.95,
        z_index=Z_AREA_RASTER,
        tooltip_fields=[TooltipField(key="value", label="Elevation", unit="m")],
        default_visible=True,
        scientific_limitations=[
            "Canonical bed elevation. Any hillshade blending is a display-only overlay, never a "
            "separate scientific product.",
        ],
    )


def hillshade_display_spec() -> DisplaySpec:
    return DisplaySpec(
        display_name="Display hillshade",
        palette=PaletteSpec(kind="single_color", color="#000000"),
        legend=LegendSpec(
            title="Display hillshade",
            kind="single_color",
            note="Display hillshade — visualization only",
        ),
        opacity=0.35,
        z_index=Z_AREA_RASTER + 1,
        default_visible=False,
        scientific_limitations=["Visualization aid only; not a scientific output."],
    )


def terrain_derivative_display_spec(
    display_name: str, unit: str, domain: tuple[float, float]
) -> DisplaySpec:
    stops = _scaled_stops(_TERRAIN_STOPS, domain)
    return DisplaySpec(
        display_name=display_name,
        palette=PaletteSpec(kind="continuous", colormap_name="terrain", domain=domain, stops=stops),
        legend=LegendSpec(title=display_name, unit=unit, kind="continuous", stops=stops),
        units=unit,
        opacity=0.9,
        z_index=Z_AREA_RASTER,
        tooltip_fields=[TooltipField(key="value", label=display_name, unit=unit)],
    )


def seabed_change_display_spec(domain: tuple[float, float]) -> DisplaySpec:
    magnitude = max(abs(domain[0]), abs(domain[1])) or 1.0
    symmetric = (-magnitude, magnitude)
    stops = _scaled_stops(_DIVERGING_STOPS, symmetric)
    return DisplaySpec(
        display_name="Seabed elevation change",
        palette=PaletteSpec(
            kind="diverging", colormap_name="RdBu", domain=symmetric, midpoint=0.0, stops=stops
        ),
        legend=LegendSpec(
            title="Seabed elevation change (m)",
            unit="m",
            kind="diverging",
            stops=stops,
            note="negative ← 0 → positive",
        ),
        units="m",
        opacity=0.9,
        z_index=Z_AREA_RASTER + 1,
        tooltip_fields=[TooltipField(key="value", label="Change", unit="m")],
        scientific_limitations=[
            "Signed multi-epoch bed-elevation difference between two surveys. Never a risk score.",
        ],
    )


def bedform_display_spec(display_name: str, tooltip_fields: list[TooltipField]) -> DisplaySpec:
    stop = PaletteStop(value=0, color="#8c6bb1", label="bedform")
    return DisplaySpec(
        display_name=display_name,
        palette=PaletteSpec(kind="categorical", stops=[stop]),
        legend=LegendSpec(title=display_name, kind="categorical", stops=[stop]),
        opacity=0.85,
        z_index=Z_AREA_VECTOR,
        tooltip_fields=tooltip_fields,
    )


def route_analysis_display_spec(
    display_name: str, tooltip_fields: list[TooltipField], *, color: str = "#e34a33"
) -> DisplaySpec:
    """A flat, distinctly-coloured route-only line. Real route-analysis outputs (mobility,
    transport intensity, scour) each carry several domain-specific numeric/categorical fields with
    no single universal "the value" column across them -- rather than invent a fake shared scale,
    this stays a single colour and lets the real fields speak for themselves via `tooltip_fields`
    (a genuine per-capability choropleth is a follow-up, not this pass)."""
    stop = PaletteStop(value=0.0, color=color, label=display_name)
    return DisplaySpec(
        display_name=display_name,
        palette=PaletteSpec(kind="single_color", color=color, stops=[stop]),
        legend=LegendSpec(
            title=display_name, kind="single_color", stops=[stop], note="Route analysis only"
        ),
        opacity=1.0,
        z_index=Z_ROUTE_ANALYSIS,
        tooltip_fields=tooltip_fields,
        scientific_limitations=[
            "Defined only along the surveyed pipeline route; not an area-wide result.",
        ],
    )


def point_evidence_display_spec(
    display_name: str, tooltip_fields: list[TooltipField]
) -> DisplaySpec:
    stop = PaletteStop(value=0.0, color="#d94801", label=display_name)
    return DisplaySpec(
        display_name=display_name,
        palette=PaletteSpec(kind="single_color", color="#d94801", stops=[stop]),
        legend=LegendSpec(title=display_name, kind="single_color", stops=[stop]),
        opacity=1.0,
        z_index=Z_POINT_EVIDENCE,
        tooltip_fields=tooltip_fields,
    )


def asset_display_spec(display_name: str) -> DisplaySpec:
    stop = PaletteStop(value=0.0, color="#111111", label=display_name)
    return DisplaySpec(
        display_name=display_name,
        palette=PaletteSpec(kind="single_color", color="#111111", stops=[stop]),
        legend=LegendSpec(title=display_name, kind="single_color", stops=[stop]),
        opacity=1.0,
        z_index=Z_ASSET,
        tooltip_fields=[TooltipField(key="chainage_m", label="Chainage", unit="m")],
    )


def area_vector_display_spec(display_name: str, color: str = "#2c7fb8") -> DisplaySpec:
    stop = PaletteStop(value=0.0, color=color, label=display_name)
    return DisplaySpec(
        display_name=display_name,
        palette=PaletteSpec(kind="single_color", color=color, stops=[stop]),
        legend=LegendSpec(title=display_name, kind="single_color", stops=[stop]),
        opacity=0.25,
        z_index=Z_AREA_VECTOR,
        tooltip_fields=[],
    )


def staging_preview_display_spec(domain: tuple[float, float]) -> DisplaySpec:
    """Generic neutral preview for a staged (not-yet-classified) raster: a plain greyscale ramp,
    labeled as a preview -- never a scientific palette, since semantic role is not established."""
    stops = [
        PaletteStop(value=domain[0], color="#1a1a1a"),
        PaletteStop(value=(domain[0] + domain[1]) / 2, color="#808080"),
        PaletteStop(value=domain[1], color="#e6e6e6"),
    ]
    return DisplaySpec(
        display_name="Staged preview",
        palette=PaletteSpec(
            kind="continuous", colormap_name="greyscale", domain=domain, stops=stops
        ),
        legend=LegendSpec(
            title="Staged preview",
            kind="continuous",
            stops=stops,
            note="Unclassified — preview only",
        ),
        opacity=0.85,
        z_index=Z_AREA_RASTER,
        default_visible=True,
        scientific_limitations=["Semantic role not yet established for this staged file."],
    )


def unclassified_display_spec(display_name: str) -> DisplaySpec:
    return DisplaySpec(
        display_name=display_name,
        palette=PaletteSpec(kind="single_color", color="#999999"),
        legend=LegendSpec(
            title="Unclassified", kind="single_color", note="Semantic role not established"
        ),
        opacity=0.6,
        z_index=Z_AREA_RASTER,
        default_visible=False,
        scientific_limitations=[
            "Spatial structure detected automatically, but no source/config/metadata establishes "
            "what this data represents. Classify it explicitly before relying on it.",
        ],
    )
