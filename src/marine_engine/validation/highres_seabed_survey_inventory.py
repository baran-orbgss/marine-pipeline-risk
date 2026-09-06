"""High-resolution seabed survey recovery + access audit (MAR-016).

A DATA-DISCOVERY / ACCESS-RECOVERY milestone, never a new model (Section 1)
----------------------------------------------------------------------------
MAR-015 demonstrated that the primary demonstrated evidence gap for
freespan context is PIPELINE-SCALE / CURRENT-ERA SEABED MORPHOLOGY: MAR-007
is derived from legacy ~1991-1992 EMODnet source bathymetry, temporally
mismatched with the 2018 official freespan survey and the current
(2024-2026) hydrodynamic forcing. This module audits whether
substantially better route-specific seabed data can actually be
RECOVERED from authoritative public metadata/data services -- it never
creates a freespan model, a susceptibility score, morphology weighting, an
ML/regression/classification model, a scour prediction, a sandwave
migration prediction, or a canonical PL854 raster, and never replaces
MAR-006/MAR-007.

Survey metadata availability != bathymetry-data availability (Section 20)
-----------------------------------------------------------------------------
BGS's own `offshore-oil-gas-site-surveys` collection description (confirmed
live) states: "This layer shows the geographic location of oil and gas
industry site surveys ... BGS do not hold the data. For further information
contact the Custodian of the data." Every BGS-sourced candidate therefore
defaults to `METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED` unless an actual
open file/API product is independently verified (never assumed).

Do not call a block-number match a route overlap (Section 3)
-------------------------------------------------------------------
Real, confirmed-live example: `bgs_ref_no=GB02SS0003` ("Anglia A") and
`decc_ref_no=GS_807` ("Bedevere rig site survey") both cite block 48/18,
yet GS_807's own real footprint sits ~12.9 km from the PL854 route while
GB02SS0003's overlaps it. Route/AOI overlap here is ALWAYS computed from
the survey's own real returned geometry, never inferred from a shared
block number or title text. Several real BGS footprints are also
near-perfect rectangles (`footprint_rectangularity` >= 0.9) consistent with
a licensed block/quadrant extent rather than an as-run vessel track line;
this is flagged explicitly and excluded from ever justifying
`CAN_ADDRESS_PIPELINE_SCALE_MORPHOLOGY_GAP` on its own.
"""

import sys
from datetime import datetime
from typing import Any

import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

# --- Section 3: route/AOI overlap vocabulary ---------------------------------------------
ROUTE_INTERSECTING = "ROUTE_INTERSECTING"
AOI_INTERSECTING_NOT_ROUTE = "AOI_INTERSECTING_NOT_ROUTE"
NEARBY_NOT_AOI = "NEARBY_NOT_AOI"

# --- Section 8: access classification vocabulary ------------------------------------------
OPEN_DIRECT_DOWNLOAD = "OPEN_DIRECT_DOWNLOAD"
OPEN_API_DATA = "OPEN_API_DATA"
METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED = "METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED"
REGISTRATION_REQUIRED = "REGISTRATION_REQUIRED"
OWNER_PERMISSION_REQUIRED = "OWNER_PERMISSION_REQUIRED"
REPORT_ONLY_NO_RAW_GRID_FOUND = "REPORT_ONLY_NO_RAW_GRID_FOUND"
ACCESS_UNKNOWN = "ACCESS_UNKNOWN"

# --- Section 11: MAR-015-gap relevance vocabulary ------------------------------------------
CAN_ADDRESS_PIPELINE_SCALE_MORPHOLOGY_GAP = "CAN_ADDRESS_PIPELINE_SCALE_MORPHOLOGY_GAP"
PARTIAL_ENDPOINT_CONTEXT_ONLY = "PARTIAL_ENDPOINT_CONTEXT_ONLY"
REGIONAL_CONTEXT_ONLY = "REGIONAL_CONTEXT_ONLY"
NO_USEFUL_ROUTE_OVERLAP = "NO_USEFUL_ROUTE_OVERLAP"

# --- Section 22: analog-dataset vocabulary -------------------------------------------------
METHOD_DEVELOPMENT_ANALOG_ONLY = "METHOD_DEVELOPMENT_ANALOG_ONLY"

# --- Section 16: Fugro 2018 dossier status vocabulary --------------------------------------
OPEN_DATA_FOUND = "OPEN_DATA_FOUND"
METADATA_FOUND_RAW_DATA_NOT_OPEN = "METADATA_FOUND_RAW_DATA_NOT_OPEN"
REPORT_EVIDENCE_ONLY = "REPORT_EVIDENCE_ONLY"
ACCESS_PATH_REQUIRES_CUSTODIAN_REQUEST = "ACCESS_PATH_REQUIRES_CUSTODIAN_REQUEST"

SOURCE_CATALOGUE_BGS = "BGS OGC API (offshore-oil-gas-site-surveys)"
FOOTPRINT_RECTANGULARITY_LICENSED_BLOCK_THRESHOLD = 0.90

# Section 18's fixed temporal reference markers.
MAR007_MORPHOLOGY_SOURCE_YEARS = (1991, 1992)
FUGRO_2018_SURVEY_YEAR = 2018
CURRENT_HYDRODYNAMIC_FORCING_YEARS = (2024, 2026)

SURVEY_INVENTORY_COLUMNS = (
    "survey_id",
    "decc_ref_no",
    "title",
    "operator",
    "custodian",
    "acquisition_start",
    "acquisition_end",
    "acquisition_year",
    "publication_or_update_date",
    "source_catalogue",
    "footprint_source",
    "route_overlap_class",
    "route_distance_m",
    "route_coverage_fraction",
    "footprint_rectangularity",
    "footprint_likely_licensed_block_extent",
    "equipment",
    "actual_available_data_types",
    "access_class",
    "download_or_enquiry_reference",
    "crs_status",
    "vertical_datum_status",
    "stated_resolution",
    "morphology_gap_relevance_class",
    "limitations",
    "survey_footprint_metadata_does_not_imply_data_custody",
)

FILE_INVENTORY_COLUMNS = (
    "survey_id",
    "filename",
    "file_type",
    "byte_size",
    "sha256",
    "stated_crs",
    "stated_vertical_datum",
    "grid_spacing_or_sounding_density",
    "acquisition_epoch",
    "licence_or_access_terms",
)

# --- Section 9: equipment keyword parser (real free-text seen in `additional_info`) --------
_EQUIPMENT_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("multibeam echo sounder", "MBES"),
    ("single beam echo sounder", "SBES"),
    ("side scan sonar", "SIDESCAN_SONAR"),
    ("sub bottom profiler", "SUB_BOTTOM_PROFILER"),
    ("deep tow boomer", "BOOMER_OR_PINGER"),
    ("surface tow boomer", "BOOMER_OR_PINGER"),
    ("hull-mounted pinger", "BOOMER_OR_PINGER"),
    ("towed pinger", "BOOMER_OR_PINGER"),
    ("magnetometer", "MAGNETOMETER"),
    ("gradiometer", "MAGNETOMETER"),
    ("grab samples", "SEDIMENT_SAMPLES"),
    ("piston cores", "SEDIMENT_SAMPLES"),
    ("gravity/piston cores", "SEDIMENT_SAMPLES"),
    ("seabed photography", "SEABED_PHOTOGRAPHY"),
    ("high resolution seismic", "HIGH_RES_SEISMIC"),
    ("pipeline tracker", "PIPELINE_TRACKER_OR_BURIAL_DATA"),
    ("burial", "PIPELINE_TRACKER_OR_BURIAL_DATA"),
    ("interpreted seabed feature", "INTERPRETED_SEABED_FEATURES"),
)


def parse_survey_equipment(additional_info: str | None) -> tuple[str, ...]:
    """`SURVEY_USED_EQUIPMENT` (Section 9) parsed from BGS's own free-text
    `additional_info` field -- never inflated into `ACTUAL_DOWNLOADABLE_
    DATA_TYPE`, which this module always tracks as a SEPARATE column."""

    if not additional_info:
        return ()
    text = additional_info.lower()
    found = {canonical for keyword, canonical in _EQUIPMENT_KEYWORDS if keyword in text}
    return tuple(sorted(found))


def survey_report_exists(additional_info: str | None) -> bool:
    """Whether BGS's own text confirms a legacy site survey REPORT was
    lodged -- distinct from, and never implying, an actual raw data grid."""

    return bool(additional_info) and "site survey report" in additional_info.lower()


# --- Section 3: real-geometry classification (never text/block-number based) --------------


def reproject_wgs84_to_working_crs(geometry_wgs84: BaseGeometry, working_crs: str) -> BaseGeometry:
    transformer = pyproj.Transformer.from_crs("EPSG:4326", working_crs, always_xy=True)
    return shapely_transform(transformer.transform, geometry_wgs84)


def compute_aoi_bbox_wgs84(
    aoi_geometry: BaseGeometry, working_crs: str
) -> tuple[float, float, float, float]:
    """The real AOI's own bounds, reprojected to WGS84 lon/lat -- the live BGS
    bbox query parameter this ticket's provider needs (Section 3: query BGS
    spatially against the real AOI, never a frozen/guessed bbox)."""

    transformer = pyproj.Transformer.from_crs(working_crs, "EPSG:4326", always_xy=True)
    aoi_wgs84 = shapely_transform(transformer.transform, aoi_geometry)
    return aoi_wgs84.bounds


def compute_footprint_rectangularity(geometry: BaseGeometry) -> float | None:
    """1.0 == a perfect rectangle. BGS's site-survey layer records survey
    LOCATION, never vessel trackline geometry -- a high rectangularity is
    consistent with a licensed block/quadrant extent (Section 3's
    "do not call a block-number match a route overlap" discipline, applied
    to the footprint shape itself, not just its text)."""

    minx, miny, maxx, maxy = geometry.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area <= 0:
        return None
    return geometry.area / bbox_area


def classify_route_overlap(*, intersects_route: bool, intersects_aoi: bool) -> str:
    if intersects_route:
        return ROUTE_INTERSECTING
    if intersects_aoi:
        return AOI_INTERSECTING_NOT_ROUTE
    return NEARBY_NOT_AOI


def classify_access(properties: dict[str, Any]) -> tuple[str, str]:
    """Every BGS-sourced record is `METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED`
    (Section 20's own collection-description statement) unless an actual
    open product has been independently verified elsewhere -- never
    invented from an optimistic reading of the abstract."""

    custodian_name = properties.get("custodian_name")
    custodian_email = properties.get("custodian_email")
    custodian_company = properties.get("custodian") or properties.get("originator")
    contact_bits = [
        v
        for v in (custodian_name, custodian_email)
        if v and str(v).strip().lower() not in ("not available", "none", "")
    ]
    if contact_bits:
        reference = f"Contact {custodian_company}: {', '.join(str(b) for b in contact_bits)}."
    else:
        reference = (
            f"No direct custodian contact published; the named custodian/originator company "
            f"is '{custodian_company}'. BGS OGC API states BGS does not hold the underlying "
            f"data -- request via BGS National Geoscience Data Centre citing "
            f"bgs_ref_no={properties.get('bgs_ref_no')}."
        )
    return METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED, reference


def classify_morphology_gap_relevance(
    *,
    route_overlap_class: str,
    access_class: str,
    route_coverage_fraction: float | None,
) -> str:
    """Section 11 -- geometric touch alone never justifies `CAN_ADDRESS_
    PIPELINE_SCALE_MORPHOLOGY_GAP`; that requires BOTH route intersection
    AND independently-verified open data (Section 20/12's discipline)."""

    if route_overlap_class == ROUTE_INTERSECTING:
        if (
            access_class in (OPEN_DIRECT_DOWNLOAD, OPEN_API_DATA)
            and (route_coverage_fraction or 0.0) > 0
        ):
            return CAN_ADDRESS_PIPELINE_SCALE_MORPHOLOGY_GAP
        return PARTIAL_ENDPOINT_CONTEXT_ONLY
    if route_overlap_class == AOI_INTERSECTING_NOT_ROUTE:
        return REGIONAL_CONTEXT_ONLY
    return NO_USEFUL_ROUTE_OVERLAP


def _parse_bgs_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_survey_candidate_row(
    feature: dict[str, Any], *, route: BaseGeometry, aoi_geometry: BaseGeometry, working_crs: str
) -> dict[str, Any]:
    """One fully-classified inventory row (Section 14's schema) from one raw
    BGS GeoJSON feature -- real geometry intersection/distance/coverage,
    never a block-number or title-text shortcut."""

    props = feature["properties"]
    geometry_wgs84 = shapely_shape(feature["geometry"])
    geometry = reproject_wgs84_to_working_crs(geometry_wgs84, working_crs)

    intersects_route = geometry.intersects(route)
    intersects_aoi = geometry.intersects(aoi_geometry)
    distance_to_route_m = 0.0 if intersects_route else geometry.distance(route)

    route_overlap_class = classify_route_overlap(
        intersects_route=intersects_route, intersects_aoi=intersects_aoi
    )
    route_intersection = geometry.intersection(route) if intersects_route else None
    route_coverage_fraction = None
    if route_intersection is not None and not route_intersection.is_empty and route.length > 0:
        route_coverage_fraction = route_intersection.length / route.length

    rectangularity = compute_footprint_rectangularity(geometry)
    likely_block_extent = (
        rectangularity is not None
        and rectangularity >= FOOTPRINT_RECTANGULARITY_LICENSED_BLOCK_THRESHOLD
    )

    access_class, download_or_enquiry_reference = classify_access(props)
    morphology_gap_relevance_class = classify_morphology_gap_relevance(
        route_overlap_class=route_overlap_class,
        access_class=access_class,
        route_coverage_fraction=route_coverage_fraction,
    )

    start = _parse_bgs_date(props.get("svy_start_date"))
    end = _parse_bgs_date(props.get("svy_end_date"))

    limitations = [
        "BGS OGC API collection description: 'BGS do not hold the data' -- metadata/footprint "
        "only, never verified raw grid data.",
    ]
    if likely_block_extent:
        limitations.append(
            f"Footprint rectangularity={rectangularity:.2f} is consistent with a licensed "
            "block/quadrant extent, not a verified as-run vessel track line; any nominal route "
            "coverage fraction reflects block geometry, not confirmed continuous survey coverage."
        )
    if route_overlap_class == ROUTE_INTERSECTING and route_coverage_fraction is not None:
        limitations.append(
            f"Nominal route coverage fraction {route_coverage_fraction:.3f} is a footprint-vs-"
            "route geometric overlap only, never a statement of actual data density/quality "
            "along that coverage."
        )

    return {
        "survey_id": props.get("bgs_ref_no"),
        "decc_ref_no": props.get("decc_ref_no"),
        "title": props.get("site_svy_name"),
        "operator": props.get("originator"),
        "custodian": props.get("custodian"),
        "acquisition_start": start,
        "acquisition_end": end,
        "acquisition_year": start.year if start is not None else None,
        "publication_or_update_date": None,  # not exposed by this collection's own schema
        "source_catalogue": SOURCE_CATALOGUE_BGS,
        "footprint_source": (
            f"BGS GeoIndex Offshore / NGDC (site_svy_id={props.get('site_svy_id')})"
        ),
        "route_overlap_class": route_overlap_class,
        "route_distance_m": distance_to_route_m,
        "route_coverage_fraction": route_coverage_fraction,
        "footprint_rectangularity": rectangularity,
        "footprint_likely_licensed_block_extent": likely_block_extent,
        "equipment": parse_survey_equipment(props.get("additional_info")),
        "actual_available_data_types": (),  # Section 9/12: never assumed from equipment alone
        "access_class": access_class,
        "download_or_enquiry_reference": download_or_enquiry_reference,
        "crs_status": "Published WGS84 (OGC:CRS84); reprojected here for intersection only.",
        "vertical_datum_status": "unknown",  # Section 13: this collection carries no datum field
        "stated_resolution": None,  # Section 10/20: never fabricated -- not documented anywhere
        "morphology_gap_relevance_class": morphology_gap_relevance_class,
        "limitations": " ".join(limitations),
        "survey_footprint_metadata_does_not_imply_data_custody": True,
    }


def build_survey_inventory_table(
    features: list[dict[str, Any]],
    *,
    route: BaseGeometry,
    aoi_geometry: BaseGeometry,
    working_crs: str,
) -> pd.DataFrame:
    rows = [
        build_survey_candidate_row(
            f, route=route, aoi_geometry=aoi_geometry, working_crs=working_crs
        )
        for f in features
    ]
    if not rows:
        return pd.DataFrame(columns=list(SURVEY_INVENTORY_COLUMNS))
    df = pd.DataFrame(rows)
    return df[list(SURVEY_INVENTORY_COLUMNS)]


def build_survey_footprints_gdf(
    features: list[dict[str, Any]], *, working_crs: str
) -> gpd.GeoDataFrame:
    """`survey_id` + real reprojected footprint geometry only -- kept SEPARATE
    from `SURVEY_INVENTORY_COLUMNS` (which stays a flat, geometry-free
    schema matching Section 14 exactly); used only by the map renderers."""

    records = [
        {
            "survey_id": f["properties"].get("bgs_ref_no"),
            "geometry": reproject_wgs84_to_working_crs(shapely_shape(f["geometry"]), working_crs),
        }
        for f in features
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs=working_crs)


def build_empty_file_inventory_table() -> pd.DataFrame:
    """Section 15: an empty canonical schema is acceptable and REQUIRED when
    (as here) no actual downloadable candidate data file was found --
    never fabricated rows."""

    return pd.DataFrame(columns=list(FILE_INVENTORY_COLUMNS))


# --- Section 21: preserve the MAR-015 morphology finding as motivation only ----------------

MAR015_APPARENT_TOPOGRAPHIC_CO_LOCATION_STATEMENT = (
    "APPARENT DESCRIPTIVE TOPOGRAPHIC CO-LOCATION: the 2018 event-containing hydrodynamic "
    "support sections lie in the top-quartile (77-92 percentile) empirical route context for "
    "legacy MAR-007 local relief/slope (MAR-015). This is a descriptive spatial observation "
    "only -- never 'morphology predicts freespan', and no model has been fitted to it."
)


# --- Section 16: the official 2018 Fugro pre-decommissioning survey ------------------------
#
# Verified via two real, public gov.uk decommissioning filings (retrieved 2026-09-06), never
# fabricated: the Anglia decommissioning programme (Ithaca Energy (UK) Limited,
# ITH-ANG-DCOM-PLN-0001 Rev C2, 13/12/2019) and its Environmental Appraisal
# (ITH-ANG-DCOM-EA-0001, Hartley Anderson Limited for Ithaca Energy, April 2020). The EA's own
# reference list cites the source survey precisely as Fugro (2018a)/(2018b) below; fieldwork
# dates to ~January 2018 (per embedded seabed-photo timestamps), reports finalised April 2018.
FUGRO_2018_TARGET_ID = "ANGLIA_FUGRO_2018_PRE_DECOMMISSIONING_SURVEY"

FUGRO_2018_OFFICIAL_REFERENCES: tuple[dict[str, str], ...] = (
    {
        "title": (
            "Decommissioning Programmes -- Anglia Field: Normally Unattended Platform "
            "Topsides, Jacket, Subsea Installations and Associated Pipelines"
        ),
        "document_id": "ITH-ANG-DCOM-PLN-0001 Rev C2",
        "publisher": "Ithaca Energy (UK) Limited",
        "date": "2019-12-13",
        "url": (
            "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/"
            "attachment_data/file/853407/Anglia_DP.pdf"
        ),
        "document_type": "Petroleum Act 1998 Section 29 decommissioning programme",
    },
    {
        "title": "Anglia Decommissioning Environmental Appraisal",
        "document_id": "ITH-ANG-DCOM-EA-0001",
        "publisher": "Hartley Anderson Limited (for Ithaca Energy (UK) Limited)",
        "date": "2020-04",
        "url": (
            "https://assets.publishing.service.gov.uk/media/5ee78179e90e07043743e353/"
            "Ithaca_Anglia_Decom_Environmental_Appraisal_April_2020.pdf"
        ),
        "document_type": "Environmental appraisal",
    },
)

FUGRO_2018_SOURCE_SURVEY_DOCUMENTS: tuple[dict[str, str], ...] = (
    {
        "citation": "Fugro (2018a)",
        "title": "Habitat Assessment -- Anglia Subsea System Survey 2017, UKCS Block 48/19b",
        "document_id": "ITA-SKO-FUG-170230-R-011(01)",
        "date": "2018-04-30",
    },
    {
        "citation": "Fugro (2018b)",
        "title": (
            "Pre-decommissioning Environmental Baseline Survey Report, "
            "Anglia Subsea System Survey 2017"
        ),
        "document_id": "ITA-SKO-FUG-170230-R-012(01)",
        "date": None,
    },
)

FUGRO_2018_KNOWN_SURVEY_FACTS: dict[str, str] = {
    "survey_fieldwork_epoch": (
        "~January 2018 (per embedded seabed-photo timestamps); reports finalised April 2018"
    ),
    "water_depth_m": "approximately 20-28 m along the export corridor (EA Section 4.1)",
    "sandwave_amplitude_m": "up to ~5 m recorded along the export route (EA Section 4.1)",
    "freespans_pl854_pl855_export_methanol_line": (
        "8 freespans, 97 m total, <1% of route length (EA Table 3.4)"
    ),
    "exposed_sections_pl854_pl855_export_methanol_line": (
        "19 exposed sections, 519 m total, 2% of route length (EA Table 3.5)"
    ),
    "sediment_sample": (
        "one pipeline-route sample described as poorly sorted coarse sand, attributed to "
        "Fugro (2018a) (EA Section 4.1)"
    ),
}

FUGRO_2018_CATALOGUES_SEARCHED: tuple[str, ...] = (
    "BGS OGC API / GeoIndex Offshore (offshore-oil-gas-site-surveys collection)",
    "data.gov.uk (being transitioned to the UK National Data Library as of early 2026, per "
    "data.gov.uk's own About page -- not a separate catalogue today)",
    "MEDIN Discovery Metadata Portal (portal.medin.org.uk, ~19,700 records)",
    "The Crown Estate Marine Data Exchange (marinedataexchange.co.uk)",
    "UK government decommissioning-programme filings (gov.uk / NSTA / OPRED)",
)


def build_fugro_2018_recovery_dossier(
    *, survey_inventory_df: pd.DataFrame, retrieved_at_utc: str
) -> dict[str, Any]:
    """Section 16: the dedicated Fugro-2018 recovery dossier. Every claim here
    traces to a real, cited public document -- the 2018 survey itself has NO
    discoverable standalone catalogue entry anywhere searched (confirmed, not
    assumed): it is known only via citation inside the Environmental
    Appraisal. Never claims raw bathymetry is open (Section 6's explicit
    instruction)."""

    bgs_candidate_ids = (
        survey_inventory_df["survey_id"].dropna().tolist() if not survey_inventory_df.empty else []
    )

    return {
        "target_id": FUGRO_2018_TARGET_ID,
        "retrieved_at_utc": retrieved_at_utc,
        "official_references": list(FUGRO_2018_OFFICIAL_REFERENCES),
        "source_survey_documents": list(FUGRO_2018_SOURCE_SURVEY_DOCUMENTS),
        "known_survey_facts": dict(FUGRO_2018_KNOWN_SURVEY_FACTS),
        "known_inferred_coverage": (
            "Export corridor, primarily UKCS block 48/19b per the Fugro (2018a) survey title; "
            "water-depth/sandwave/freespan/exposure facts above are stated for the PL854/PL855 "
            "export-methanol pipeline specifically."
        ),
        "raw_data_availability_status": (
            "Only the descriptive Environmental Appraisal PDF is publicly downloadable -- it "
            "reproduces summary tables (freespan/exposure counts), seabed photographs "
            "(Appendix B), sediment-sample summary tables, and plotted freespan-location maps "
            "(Appendix A, not a georeferenced dataset). No MBES grid, XYZ soundings, GeoTIFF, "
            "or shapefile from the 2018 survey was found published anywhere."
        ),
        "catalogues_searched": list(FUGRO_2018_CATALOGUES_SEARCHED),
        "candidate_dataset_identifiers_checked": bgs_candidate_ids,
        "candidate_dataset_identifiers_note": (
            "None of the BGS-catalogued candidates ARE the 2018 Fugro survey -- the closest "
            "catalogued analogs (GB02SS0001, GB02SS0003, GB03SS0002) are 2002-2003 GDF "
            "Suez-era surveys by a different contractor (Gardline, not Fugro)."
        ),
        "prior_ticket_cross_check": (
            "MAR-005 (an earlier, separate bathymetry-source discovery ticket) already queried "
            "BGS's OWN DIFFERENT GeoNetwork CSW 2.0.2 catalogue (metadata.bgs.ac.uk, not this "
            "ticket's ogcapi.bgs.ac.uk OGC API Features service) for GB02SS0001 and CS03SS0003 "
            "directly. Re-run live for this dossier: both independently corroborate this "
            "module's own OGC API Features findings -- same survey dates, "
            "access_type='restricted', download_available=False -- via a completely different "
            "protocol, and CS03SS0003's CSW footprint bounds (1.8976, 53.3909, 2.0026, 53.7251) "
            "match the ticket's own stated extent exactly."
        ),
        "custodian_contact_route": (
            "Ithaca Energy (UK) Limited, as the decommissioning operator that commissioned the "
            "2018 Fugro survey, is the most plausible practical contact route; no formally "
            "published data-request procedure for this specific survey was found."
        ),
        "exact_unresolved_access_blocker": (
            "The 2018 Fugro survey's raw deliverables (MBES grid / XYZ soundings) were never "
            "submitted to a public archive (BGS NGDC or otherwise) under a discoverable "
            "identifier; access would require directly requesting them from Ithaca Energy (UK) "
            "Limited or Fugro."
        ),
        "final_status": REPORT_EVIDENCE_ONLY,
    }


# --- Section 22: optional analog datasets (method development only, never PL854 evidence) --
#
# Verified live via web research (2026-09-06) -- Southern North Sea, real, open, high-
# resolution multibeam-grade datasets. None of these enter PL854 features or validation.

ANALOG_DATASET_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "name": "East Coast Regional Environmental Characterisation (ECREC)",
        "custodian": "Cefas",
        "approximate_area": "~3,300 km2 off Norfolk/Suffolk, East Anglia coast",
        "survey_years": "2008-2009",
        "data_types": (
            "MBES bathymetry (Kongsberg EM3002D, 0.5 m and 1 m grids)",
            "sidescan sonar",
            "grab samples",
            "magnetometer",
            "vibrocore",
            "video",
        ),
        "formats": ("GeoTIFF", "XYZ", "Fledermaus SD", "GSF", "XTF", "SEGY"),
        "access_class": OPEN_DIRECT_DOWNLOAD,
        "access_reference": (
            "Crown Estate Marine Data Exchange, package 1224 -- per-package download, no "
            "registration observed to be required."
        ),
        "classification": METHOD_DEVELOPMENT_ANALOG_ONLY,
    },
    {
        "name": "Inner Dowsing, Race Bank & North Ridge cSAC -- Processed Bathymetry",
        "custodian": "JNCC / Cefas",
        "approximate_area": "Greater Wash area, bbox ~53.09-53.42N, 0.51-1.06E",
        "survey_years": "published 2015",
        "data_types": ("processed MBES-derived bathymetry grid", "habitat ground-truth"),
        "formats": ("ZIP",),
        "access_class": OPEN_DIRECT_DOWNLOAD,
        "access_reference": "data.gov.uk, UK Open Government Licence, no registration required.",
        "classification": METHOD_DEVELOPMENT_ANALOG_ONLY,
    },
    {
        "name": "Sheringham Shoal Offshore Wind Farm bathymetric survey",
        "custodian": "Fugro EMU Ltd (for Scira Offshore Energy Ltd)",
        "approximate_area": "Wind farm site + export cable corridor, North Norfolk coast",
        "survey_years": "2014-04 to 2014-05",
        "data_types": ("bathymetry survey", "post-construction monitoring report"),
        "formats": (),
        "access_class": ACCESS_UNKNOWN,
        "access_reference": (
            "Crown Estate Marine Data Exchange, record TCE-1977 -- downloadable via the detail "
            "page, but exact grid format and whether registration is genuinely required for "
            "this specific download were NOT confirmed either way."
        ),
        "classification": METHOD_DEVELOPMENT_ANALOG_ONLY,
    },
)


# --- Section 19: access-gap report -----------------------------------------------------


def build_access_gap_report(
    *, survey_inventory_df: pd.DataFrame, fugro_dossier: dict[str, Any]
) -> dict[str, Any]:
    """Section 19: literal answers only, no score anywhere."""

    route_df = survey_inventory_df[survey_inventory_df["route_overlap_class"] == ROUTE_INTERSECTING]
    open_route_df = route_df[route_df["access_class"].isin((OPEN_DIRECT_DOWNLOAD, OPEN_API_DATA))]

    best_coverage_row = (
        route_df.sort_values("route_coverage_fraction", ascending=False).iloc[0]
        if not route_df.empty
        else None
    )

    return {
        "question_A": (
            "Is any OPEN route-intersecting pipeline-scale MBES/bathymetric grid available?"
        ),
        "answer_A": (
            "NO -- every route-intersecting BGS-catalogued survey is metadata-only "
            "(METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED); no open MBES/bathymetric grid was "
            "verified."
            if open_route_df.empty
            else f"YES -- {open_route_df['survey_id'].tolist()}"
        ),
        "question_B": "Is a 2018-contemporaneous bathymetric dataset available?",
        "answer_B": (
            "NO -- the 2018 Fugro pre-decommissioning survey exists only as a citation inside "
            "the Anglia Environmental Appraisal (summary tables, photos, and plotted maps); no "
            "raw 2018 bathymetric grid/soundings dataset was found published anywhere. "
            f"Dossier status: {fugro_dossier['final_status']}."
        ),
        "question_C": "Are only metadata footprints available for relevant oil/gas surveys?",
        "answer_C": (
            "YES for every BGS-catalogued 2001-2005 survey checked (BGS's own collection "
            "description states BGS does not hold the underlying data); the 2018 Fugro survey "
            "itself does not even have a metadata footprint catalogued anywhere searched."
        ),
        "question_D": "Which survey currently has the best route coverage?",
        "answer_D": (
            f"{best_coverage_row['survey_id']} ({best_coverage_row['title']}), nominal route "
            f"coverage fraction {best_coverage_row['route_coverage_fraction']:.3f} -- but its "
            f"footprint rectangularity is {best_coverage_row['footprint_rectangularity']:.2f}, "
            "consistent with a licensed block extent rather than confirmed continuous survey "
            "coverage."
            if best_coverage_row is not None
            else "NO route-intersecting candidate was found."
        ),
        "question_E": (
            "Which survey has the best temporal alignment with the 2018 observed freespans?"
        ),
        "answer_E": (
            "None of the BGS-catalogued, checked surveys are contemporaneous with 2018 (all are "
            "2001-2005, 13-17 years earlier); the only 2018-contemporaneous survey is the Fugro "
            "pre-decommissioning survey itself, which is real and well-documented but has no "
            "open dataset -- only report evidence."
        ),
        "question_F": "What exact access step is required for the best unavailable candidate?",
        "answer_F": fugro_dossier["exact_unresolved_access_blocker"],
    }


def print_survey_inventory_report(
    *,
    survey_inventory_df: pd.DataFrame,
    access_gap_report: dict[str, Any],
    outputs: dict[str, Any],
    file: Any = None,
) -> None:
    file = file or sys.stdout
    lines = ["=== PL854 High-Resolution Seabed Survey Inventory (MAR-016) ===", ""]

    lines.append("## Survey discovery")
    lines.append(f"  Total candidates checked: {len(survey_inventory_df)}")
    for overlap_class in (ROUTE_INTERSECTING, AOI_INTERSECTING_NOT_ROUTE, NEARBY_NOT_AOI):
        subset = survey_inventory_df[survey_inventory_df["route_overlap_class"] == overlap_class]
        lines.append(f"  {overlap_class}: {len(subset)}")
    lines.append("")

    lines.append("## Route-intersecting candidates")
    route_df = survey_inventory_df[survey_inventory_df["route_overlap_class"] == ROUTE_INTERSECTING]
    for _, row in route_df.iterrows():
        lines.append(
            f"  {row['survey_id']} ({row['acquisition_year']}, {row['operator']}): "
            f"coverage={row['route_coverage_fraction']}, access={row['access_class']}, "
            f"morphology_gap_relevance={row['morphology_gap_relevance_class']}"
        )
    lines.append("")

    lines.append("## Access-gap answers")
    for key, value in access_gap_report.items():
        if key.startswith(("question_", "answer_")):
            lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append(f"  {MAR015_APPARENT_TOPOGRAPHIC_CO_LOCATION_STATEMENT}")
    lines.append("")

    lines.append("## Outputs")
    for key, value in outputs.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append(
        "MAR-016 INVENTORIES EVIDENCE AND ACCESS ONLY. NO NEW MORPHOLOGY, FREESPAN PREDICTION "
        "OR SUSCEPTIBILITY SCORE HAS BEEN CREATED."
    )
    print("\n".join(lines), file=file)
