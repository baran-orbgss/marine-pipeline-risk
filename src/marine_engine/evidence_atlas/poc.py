"""OrbGSS Marine POC v0.1 external-reviewer package (MAR-019).

Product-framing / presentation content only -- no new geohazard physics, no
risk/susceptibility score, no ML, no new external-data research. Reuses the
same format-neutral block renderer as `report.py` (`render_blocks_html`/
`render_blocks_markdown`) so the POC overview's HTML and Markdown can never
drift apart from each other.
"""

from __future__ import annotations

import html as html_module
from pathlib import Path
from typing import Any

import geopandas as gpd

ATLAS_GPKG_LAYER_NAMES: tuple[str, ...] = (
    "pipeline_route",
    "engineering_support_sections",
    "observed_freespans_2018",
    "historical_freespans_2012_2018",
    "observed_psa_d50_points",
    "highres_survey_inventory",
    "chainage_reference_points",
)


def read_atlas_gpkg_layers(gpkg_path: Path, *, working_crs: str) -> dict[str, gpd.GeoDataFrame]:
    """Reload every already-built MAR-018 atlas layer (read-only reuse, no
    recomputation). A layer absent from the GeoPackage (it is only ever
    written when non-empty) reloads as an empty, correctly-CRS'd
    GeoDataFrame rather than raising."""

    layers: dict[str, gpd.GeoDataFrame] = {}
    for layer_name in ATLAS_GPKG_LAYER_NAMES:
        try:
            layers[layer_name] = gpd.read_file(gpkg_path, layer=layer_name)
        except Exception:  # noqa: BLE001 -- an optional layer may legitimately be absent
            layers[layer_name] = gpd.GeoDataFrame(geometry=[], crs=working_crs)
    return layers


# --- Upstream integrity: MAR-018 outputs, read-only reuse (never recomputed) ----------------


def required_mar018_outputs(study_dir: Path) -> dict[str, Path]:
    """Every already-accepted MAR-018 output this package reuses verbatim."""

    return {
        "section_evidence_parquet": (
            study_dir / "evidence_atlas" / "pl854_section_evidence.parquet"
        ),
        "atlas_gpkg": study_dir / "evidence_atlas" / "pl854_engineering_evidence_atlas.gpkg",
        "engineering_report_html": (
            study_dir / "report" / "pl854_engineering_evidence_report.html"
        ),
        "engineering_report_md": study_dir / "report" / "pl854_engineering_evidence_report.md",
        "chainage_sediment_evidence": (
            study_dir / "sediment" / "chainage_sediment_evidence.parquet"
        ),
        "condition_benchmark": (
            study_dir / "pipeline_condition" / "anglia_2018_condition_benchmark.json"
        ),
    }


def check_mar018_outputs_present(study_dir: Path) -> list[str]:
    """`f"{name}: {path}"` for every required MAR-018 output that does not exist."""

    return [
        f"{name}: {path}"
        for name, path in required_mar018_outputs(study_dir).items()
        if not path.exists()
    ]


# --- POC overview content (Sections 15-18) ---------------------------------------------------

POC_TITLE = "OrbGSS Marine Module -- POC v0.1"

# Verbatim product-definition negatives (ticket Section 1) -- what the Module is NOT.
PRODUCT_NEGATIVES: tuple[str, ...] = (
    "a survey acquisition company",
    "a geophysical contractor",
    "a drilling / CPT / borehole contractor",
    "a replacement for primary geological/geophysical interpretation",
    "a generic AI black-box hazard predictor",
)

# Hazard-map product-concept statuses (Section 18): "demonstrated" / "screening prototype" /
# "planned" only -- deliberately no percentage-complete metric.
HAZARD_MAP_STATUS_ROWS: tuple[tuple[str, str], ...] = (
    (
        "Seabed morphology / bedforms",
        "demonstrated (regional context); high-resolution engine synthetically validated only",
    ),
    ("Sediment mobility", "demonstrated"),
    ("Scour", "screening prototype"),
    ("Burial/exposure", "demonstrated (2018 observed evidence only, not predictive)"),
    ("Freespan", "demonstrated (2018 observed evidence only, not predictive)"),
    ("Slope instability", "planned"),
    ("Shallow gas", "planned"),
    ("Faults / structural features", "planned"),
    ("Existing infrastructure / crossings", "planned"),
)


def build_poc_overview_blocks() -> list[dict[str, Any]]:
    """Format-neutral blocks for the 3-5 minute external-reviewer overview."""

    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "heading", "level": 1, "text": POC_TITLE})

    blocks.append({"type": "heading", "level": 2, "text": "What this is"})
    blocks.append(
        {
            "type": "paragraph",
            "text": "A GIS-based offshore pipeline geohazard analytics software concept.",
        }
    )
    blocks.append(
        {
            "type": "paragraph",
            "text": "The OrbGSS Marine Module is NOT: " + "; ".join(PRODUCT_NEGATIVES) + ".",
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "Intended user"})
    blocks.append(
        {
            "type": "paragraph",
            "text": "Operator / pipeline engineer / georisk engineer working with existing "
            "project survey and engineering data.",
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "Intended workflow"})
    blocks.append(
        {
            "type": "paragraph",
            "text": "Operator data -> QA -> standardized project data model -> individual "
            "geohazard analyses -> map -> KP view -> engineering report.",
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "What this POC demonstrates today"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "Route-based GIS analysis",
                "Bathymetric/regional terrain context",
                "Metocean forcing processing",
                "Combined bed shear",
                "Noncohesive sediment mobility",
                "Scour-onset screening",
                "Observed freespan evidence integration",
                "Provenance / evidence governance",
                "Map + KP strip + engineering report",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "What it does NOT yet demonstrate"})
    blocks.append(
        {
            "type": "list",
            "items": [
                "Generic customer project creation",
                "Customer file upload",
                "Automated MBES ingestion workflow",
                "SSS/SBP ingestion UI",
                "Shallow-gas/fault interpretation ingestion workflow",
                "Full geohazard catalogue",
                "Route optimisation",
                "Validated freespan susceptibility",
                "Lifecycle reliability",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "Why PL854 is used"})
    blocks.append(
        {
            "type": "paragraph",
            "text": "Public-data development case only. PL854 is used because no "
            "operator-supplied, project-grade survey package is available for this POC. A "
            "real deployment assumes the operator provides authorized project-grade survey / "
            "interpreted / engineering data; OrbGSS does not intend to replace those surveys "
            "with public datasets.",
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "Future operator-data concept"})
    blocks.append(
        {
            "type": "paragraph",
            "text": "A production deployment is built around three distinct data "
            "categories, described below.",
        }
    )

    blocks.append(
        {"type": "heading", "level": 2, "text": "Measured / Interpreted / Derived Data Model"}
    )
    blocks.append(
        {
            "type": "list",
            "items": [
                "MEASURED DATA -- direct physical observations. Examples: MBES, CPT, "
                "boreholes, grab samples, current observations.",
                "INTERPRETED DATA -- expert or contractor interpretation of measured data. "
                "Examples: shallow gas polygons, faults, buried channels, boulders, seabed "
                "feature interpretation.",
                "DERIVED ENGINEERING LAYERS -- software-computed analytics from measured/"
                "interpreted data. Examples: terrain morphology, sediment mobility, scour "
                "susceptibility, eventual free-span susceptibility, slope-instability "
                "screening, route constraints. Not all of these derived layers exist today -- "
                "see the product-concept status table below.",
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "Future Hazard-Map Concept"})
    blocks.append(
        {
            "type": "paragraph",
            "text": "Future Marine projects may contain separate maps for the hazard classes "
            "below. Status reflects what exists TODAY in this POC -- never a commitment, and "
            "never a percentage-complete metric.",
        }
    )
    blocks.append(
        {
            "type": "table",
            "headers": ["Hazard class", "Status"],
            "rows": [[name, status] for name, status in HAZARD_MAP_STATUS_ROWS],
        }
    )

    blocks.append(
        {
            "type": "callout",
            "text": "ORBGSS MARINE MODULE POC ASSUMES OPERATOR-SUPPLIED PROJECT-GRADE SURVEY "
            "/ GEOPHYSICAL / GEOTECHNICAL DATA IN A PRODUCTION DEPLOYMENT.",
        }
    )
    blocks.append(
        {
            "type": "callout",
            "text": "ORBGSS IS THE SOFTWARE ANALYTICS / GIS / ENGINEERING-COMMUNICATION LAYER; "
            "THIS POC DOES NOT POSITION ORBGSS AS THE PRIMARY SURVEY OR DRILLING CONTRACTOR.",
        }
    )
    blocks.append({"type": "callout", "text": "PL854 IS A PUBLIC-DATA DEVELOPMENT CASE ONLY."})

    return blocks


# --- External reviewer guide (Sections 19-21) ------------------------------------------------


def build_review_guide_markdown() -> str:
    """Pure Markdown (no HTML twin required) -- engineering/product questions, never
    a bare 'do you like it?' (Section 19), plus the required data-authorization
    and privacy caveat (Section 21)."""

    return """# External Georisk Review Guide -- OrbGSS Marine POC v0.1

Thank you for reviewing this proof of concept. This is not a request for general
feedback -- please answer the engineering/product questions below from your own
practical experience running or reviewing offshore pipeline geohazard workflows.

## How to review

Open `index.html` first, then the POC Overview, then the Engineering Evidence
Atlas, the KP Evidence Strip, and the Engineering Evidence Report, in that order.
A thorough review typically takes 30-60 minutes.

## Engineering Relevance

- Which outputs are actually useful in a pipeline geohazard workflow?
- Which are redundant?
- Which major hazard classes are missing?

## Data Input

- Which datasets would normally be available at the point this software would
  be used?
- Which formats are common in practice?
- Which metadata are essential before analysis?
- How are CRS / vertical datum / survey epoch normally supplied?

## Scientific Interpretation

- Which outputs should remain expert-interpreted?
- Which can defensibly be software-derived?
- Which terminology is misleading or non-standard?

## Map / GIS Usability

- Is Map + KP View the right primary interaction model?
- Which variables need to be visible when selecting a route section?
- Which outputs would normally be exported to QGIS / ArcGIS?

## Reporting

- Which outputs would belong in a client/engineering report?
- What evidence/provenance must accompany a hazard map?
- What would prevent using an OrbGSS output in a real project?

## Workflow

- At which stage of a real pipeline project would this platform provide the
  most value?
- What currently requires repeated manual GIS/Excel work?
- Where would software automation save the most engineering time?

## Safety / Misinterpretation

- Which current outputs could lead a less experienced user to the wrong
  engineering conclusion?
- Which warnings/limitations should always remain visible?

## If you want to test a future data-ingestion version

PL854 public data are sufficient for this first product/scientific review --
no additional data is needed or requested for this round.

If, in a future round, you wish to test a data-ingestion concept with your own
data, any data you provide must be:

- owned or authorized by you/your organisation for this purpose, and
- non-confidential, OR properly anonymized / cleared for sharing.

We will not request proprietary client data casually, and you should never feel
obliged to provide it.
"""


# --- POC package index landing page ----------------------------------------------------------

_INDEX_STYLE = """
body { font-family: Georgia, 'Times New Roman', serif; max-width: 960px; margin: 2.5rem auto;
       padding: 0 1.5rem; color: #1a1a1a; line-height: 1.55; }
h1 { font-size: 2rem; margin-bottom: 0.2rem; }
.subtitle { color: #444; font-style: italic; margin-top: 0; margin-bottom: 2rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 1.1rem; }
.card { display: block; border: 1px solid #999; border-radius: 6px; padding: 1.1rem 1.3rem;
        text-decoration: none; color: inherit; background: #FAFAF6; transition: background 0.15s; }
.card:hover { background: #F0EEE4; }
.card h2 { margin: 0 0 0.4rem 0; font-size: 1.1rem; color: #2a3d5c; }
.card p { margin: 0; font-size: 0.92rem; color: #333; }
footer { margin-top: 2.5rem; font-size: 0.82rem; color: #555; font-style: italic; }
"""


def build_poc_index_html(*, deliverables: list[dict[str, str]]) -> str:
    """`deliverables`: ordered list of {title, href, description} -- the first
    file an external reviewer opens (Section 22)."""

    cards = "".join(
        f'<a class="card" href="{html_module.escape(d["href"])}">'
        f"<h2>{html_module.escape(d['title'])}</h2>"
        f"<p>{html_module.escape(d['description'])}</p>"
        "</a>"
        for d in deliverables
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>OrbGSS Marine POC -- Review Package</title>"
        f"<style>{_INDEX_STYLE}</style></head><body>"
        f"<h1>{POC_TITLE}</h1>"
        '<p class="subtitle">External reviewer package -- PL854 public-data demonstrator '
        "(offshore pipeline geohazard analytics, engineering decision support)</p>"
        f'<div class="grid">{cards}</div>'
        "<footer>No fused risk/susceptibility score. Operator-supplied project-grade data "
        "is assumed for any production deployment.</footer>"
        "</body></html>"
    )
