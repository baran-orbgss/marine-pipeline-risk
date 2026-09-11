"""Explicit presentation registry for the engineering workbench (UI-001 Sections 7, 8, 11, 12).

This module is PRESENTATION METADATA ONLY. It states which capabilities exist, how mature they are
as SOFTWARE / SCIENTIFIC CAPABILITIES, which tests and configs exercise them and which local output
files they are known to write. It is not a scientific authority: nothing here computes, thresholds,
reinterprets or overrides any engine result. Maturity is never hazard severity, risk level,
engineering acceptance or a safe/unsafe classification.

Nothing in this registry is inferred from README prose or from directory names.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

__all__ = [
    "QUALIFIED_POC",
    "FOUNDATION_READY",
    "PARTIAL",
    "UNDER_CONSTRUCTION",
    "MATURITY_VOCABULARY",
    "MATURITY_MEANING",
    "STATUS_SEMANTICS_NOTICE",
    "REAL_SOURCE",
    "SYNTHETIC_TEST_FIXTURE",
    "CACHED_REAL_OUTPUT",
    "NOT_AVAILABLE",
    "EVIDENCE_KINDS",
    "DEFAULT_FLOW",
    "OutputPattern",
    "Capability",
    "SupportingCapability",
    "Project",
    "HAZARDS",
    "SUPPORTING",
    "PROJECTS",
    "LIQUEFACTION_CARD",
    "hazard_by_id",
    "project_by_id",
    "maturity_counts",
    "all_registered_test_files",
    "capabilities_for_test_file",
    "capabilities_for_project",
]

# --- maturity vocabulary (UI-001 Section 8-9) ----------------------------------------------------

QUALIFIED_POC = "QUALIFIED_POC"
FOUNDATION_READY = "FOUNDATION_READY"
PARTIAL = "PARTIAL"
UNDER_CONSTRUCTION = "UNDER_CONSTRUCTION"
MATURITY_VOCABULARY = (QUALIFIED_POC, FOUNDATION_READY, PARTIAL, UNDER_CONSTRUCTION)

MATURITY_MEANING = {
    QUALIFIED_POC: (
        "Implemented generic engine + real-data proof-of-concept run + offline tests + documented "
        "limitations. A research POC; not an engineering qualification."
    ),
    FOUNDATION_READY: (
        "Evidence / input foundation implemented and verified on real data; the hazard physics "
        "itself is not implemented."
    ),
    PARTIAL: (
        "Some inputs or adjacent building blocks exist in the engine; no dedicated hazard product "
        "exists."
    ),
    UNDER_CONSTRUCTION: "Planned capability; nothing implemented yet.",
}

STATUS_SEMANTICS_NOTICE = (
    "Colours and statuses on this workbench describe SOFTWARE / SCIENTIFIC CAPABILITY MATURITY. "
    "They are NOT hazard severity, NOT a risk level, NOT engineering acceptance and NOT a "
    "safe / unsafe classification. Do not read the maturity colour as a risk-map legend."
)

# --- evidence kind vocabulary (UI-001 Section 20) --------------------------------------------------

REAL_SOURCE = "REAL_SOURCE"
SYNTHETIC_TEST_FIXTURE = "SYNTHETIC_TEST_FIXTURE"
CACHED_REAL_OUTPUT = "CACHED_REAL_OUTPUT"
NOT_AVAILABLE = "NOT_AVAILABLE"
EVIDENCE_KINDS = (
    REAL_SOURCE,
    SYNTHETIC_TEST_FIXTURE,
    CACHED_REAL_OUTPUT,
    UNDER_CONSTRUCTION,
    NOT_AVAILABLE,
)

DEFAULT_FLOW = ("INPUT", "CANONICALIZATION", "ANALYSIS", "QA / READINESS", "OUTPUT")


@dataclass(frozen=True)
class OutputPattern:
    """A known output location, as a glob relative to `data/` (e.g. `processed/pl854/scour/*.gpkg`).
    `evidence_kind` says what a file matching the pattern IS if it exists locally; the workbench
    never fabricates a file that does not match."""

    label: str
    glob: str
    evidence_kind: str = CACHED_REAL_OUTPUT
    project_id: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {self.evidence_kind!r}")
        if self.glob.startswith(("/", "\\")) or ".." in self.glob.split("/"):
            raise ValueError(f"output pattern must be relative to data/ without '..': {self.glob}")


@dataclass(frozen=True)
class Capability:
    id: str
    hazard_number: int
    title: str
    maturity: str
    what_it_does: str
    scientific_role: tuple[str, ...]
    inputs: tuple[str, ...]
    method: tuple[str, ...]
    outputs: tuple[str, ...]
    real_benchmark: str
    tests: tuple[str, ...]
    configs: tuple[str, ...]
    output_patterns: tuple[OutputPattern, ...]
    limitations: tuple[str, ...]
    next_planned: str
    tickets: tuple[str, ...] = ()
    cli_commands: tuple[str, ...] = ()
    flow: tuple[str, ...] = DEFAULT_FLOW

    def __post_init__(self) -> None:
        if self.maturity not in MATURITY_VOCABULARY:
            raise ValueError(f"{self.id}: unknown maturity {self.maturity!r}")
        if not 1 <= self.hazard_number <= 15:
            raise ValueError(f"{self.id}: hazard number out of range")
        if not self.limitations:
            raise ValueError(f"{self.id}: the scientific-limits panel is mandatory")
        for t in self.tests:
            if not (t.startswith("tests/") and t.endswith(".py")):
                raise ValueError(f"{self.id}: test path must be tests/<name>.py, got {t!r}")


@dataclass(frozen=True)
class SupportingCapability:
    id: str
    title: str
    maturity: str
    description: str
    tests: tuple[str, ...]
    tickets: tuple[str, ...] = ()
    cli_commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.maturity not in MATURITY_VOCABULARY:
            raise ValueError(f"{self.id}: unknown maturity {self.maturity!r}")


@dataclass(frozen=True)
class Project:
    """A known project / demo context. `evidence_kind` describes the SOURCE data lineage the
    context is built on; every displayed file still carries its own evidence kind."""

    id: str
    title: str
    description: str
    evidence_kind: str
    configs: tuple[str, ...]
    processed_dirs: tuple[str, ...]
    interim_dirs: tuple[str, ...]
    readiness_files: tuple[str, ...]
    capability_ids: tuple[str, ...]
    source_note: str = ""

    def __post_init__(self) -> None:
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"{self.id}: unknown evidence kind {self.evidence_kind!r}")


def _p(
    label: str, glob: str, project_id: str | None, kind: str = CACHED_REAL_OUTPUT
) -> OutputPattern:
    return OutputPattern(label=label, glob=glob, evidence_kind=kind, project_id=project_id)


_SS20 = "sheringham_shoal_2020"
_PL854 = "pl854"
_BARROW = "barrow_2016"
_CPTU = "sheringham_shoal_2008_cptu"

# --- the 15 hazard capabilities (UI-001 Section 8) -------------------------------------------------

HAZARDS: tuple[Capability, ...] = (
    Capability(
        id="terrain",
        hazard_number=1,
        title="Terrain",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Builds a canonical high-resolution seabed terrain raster from an operator bathymetry "
            "survey and derives slope, aspect, curvature, local relief, ruggedness and terrain "
            "standard deviation at explicit analysis windows, with an operator-style bathymetry "
            "readiness assessment."
        ),
        scientific_role=(
            "Measured seabed geometry (survey-derived), no morphological interpretation.",
            "Readiness = data readiness of the bathymetry input, not survey acceptance.",
        ),
        inputs=("Operator MBES bathymetry GeoTIFF (canonical CRS/datum/units declared)",),
        method=(
            "Canonical raster semantics (CRS, vertical reference, nodata) under an explicit input "
            "contract",
            "Neighbourhood terrain derivatives at 10 m and 50 m windows",
            "Tiled processing for large rasters (MAR-031)",
        ),
        outputs=(
            "terrain/*.tif derivative rasters",
            "gis/seabed_terrain_poc.gpkg",
            "maps/*_terrain_atlas.png, *_bathymetry_qa.png",
            "report/*_terrain_poc.html, terrain_poc_validation.json",
        ),
        real_benchmark="Sheringham Shoal 2020 Fugro MBES 1 m survey (TCE-1986, Marine Data Exchange).",
        tests=("tests/test_terrain_poc.py",),
        configs=("configs/sheringham_shoal_2020.yaml",),
        output_patterns=(
            _p("Terrain derivative rasters", f"processed/{_SS20}/terrain/*.tif", _SS20),
            _p("Terrain GIS package", f"processed/{_SS20}/gis/seabed_terrain_poc.gpkg", _SS20),
            _p("Terrain atlas map", f"processed/{_SS20}/maps/*terrain_atlas.png", _SS20),
            _p("Bathymetry QA map", f"processed/{_SS20}/maps/*bathymetry_qa.png", _SS20),
            _p("Bathymetry readiness", f"processed/{_SS20}/readiness/bathymetry*.json", _SS20),
            _p("Terrain POC validation", f"processed/{_SS20}/terrain_poc_validation.json", _SS20),
            _p("Terrain POC report", f"processed/{_SS20}/report/*terrain_poc.html", _SS20),
        ),
        limitations=(
            "Derivatives describe the surveyed surface at the survey epoch only.",
            "No geomorphological interpretation; slope is geometry, not a hazard.",
            "Readiness states data readiness; it is not survey acceptance or QC sign-off.",
        ),
        next_planned="Terrain derivatives feed slope-instability and bedform engines (done); no new terrain physics planned.",
        tickets=("MAR-020", "MAR-031"),
        cli_commands=("build-highres-terrain-poc",),
    ),
    Capability(
        id="bedforms",
        hazard_number=2,
        title="Bedforms",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Extracts sand-wave / bedform morphometry (tile spectral scale, transect geometry, "
            "individual bedforms) from high-resolution bathymetry and matches crests independently "
            "across epochs, restricted to natural-context tiles."
        ),
        scientific_role=(
            "Measured bedform geometry and OBSERVED crest displacement between two epochs.",
            "Source interpretation (operator bedform layers) kept separate from measured geometry.",
        ),
        inputs=(
            "Canonical terrain rasters for one or two epochs",
            "Optional source bedform interpretation layers (comparator only)",
        ),
        method=(
            "Tile spectral morphometry and transect morphometry (MAR-017 engine)",
            "Independent multi-epoch crest matching (MAR-022)",
            "Natural-vs-anthropogenic tile context gate; disturbed tiles reported non-canonically",
        ),
        outputs=(
            "bedforms/*.parquet observation, morphometry and crest-match tables",
            "gis/bedform_morphodynamics_poc.gpkg",
            "maps/*bedform*.png",
            "report/*bedform_morphodynamics_poc.html",
        ),
        real_benchmark=(
            "Sheringham Shoal 2018 vs 2020 Fugro MBES; engine validated on three open analogs "
            "(HHW CEND 11/11, IDRBNR CEND 11/11, Greater Gabbard 2014)."
        ),
        tests=(
            "tests/test_bedform_morphodynamics_poc.py",
            "tests/test_sandwave_morphometry.py",
            "tests/test_sandwave_morphometry_map.py",
            "tests/test_hhw_cend1111_analog.py",
            "tests/test_idr_bnr_cend1111_analog.py",
            "tests/test_greater_gabbard_2014_analog.py",
        ),
        configs=("configs/sheringham_shoal_2020.yaml",),
        output_patterns=(
            _p("Bedform tables", f"processed/{_SS20}/bedforms/*.parquet", _SS20),
            _p(
                "Bedform GIS package",
                f"processed/{_SS20}/gis/bedform_morphodynamics_poc.gpkg",
                _SS20,
            ),
            _p("Bedform maps", f"processed/{_SS20}/maps/*bedform*.png", _SS20),
            _p(
                "Crest displacement statistics",
                f"processed/{_SS20}/maps/*crest_displacement*.png",
                _SS20,
            ),
            _p(
                "Bedform POC validation",
                f"processed/{_SS20}/bedform_morphodynamics_poc_validation.json",
                _SS20,
            ),
            _p("Analog validation summary", "processed/analogs/*.parquet", None),
            _p("Analog validation status", "processed/analogs/*.json", None),
            _p("Analog maps (method development)", "processed/analogs/*/maps/*.png", None),
        ),
        limitations=(
            "Observed crest displacement is geometry between two epochs; it is not a migration-rate "
            "prediction or a sediment-transport rate.",
            "Canonical results are restricted to natural-context tiles; anthropogenically disturbed "
            "tiles are diagnostic only.",
            "Analog datasets are method-development evidence, never project evidence.",
        ),
        next_planned="Route-referenced bedform interaction products; no new bedform physics authorized.",
        tickets=("MAR-017", "MAR-017B", "MAR-017C", "MAR-022"),
        cli_commands=(
            "build-bedform-morphodynamics-poc",
            "build-analog-sandwave-morphometry",
            "build-idrbnr-sandwave-validation",
            "build-greater-gabbard-sandwave-validation",
        ),
    ),
    Capability(
        id="sediment_mobility",
        hazard_number=3,
        title="Sediment Mobility",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Screens noncohesive sediment mobility capacity along a route by comparing combined "
            "wave-current bed shear against a critical Shields threshold for explicitly tested D50 "
            "scenarios."
        ),
        scientific_role=(
            "Scenario-based mobility CAPACITY screening; not a transport rate.",
            "Grain-size context from BGS particle-size analysis observations.",
        ),
        inputs=(
            "Combined wave-current bed shear (MAR-012) along chainage",
            "Tested D50 scenarios; BGS PSA D50 context (MAR-008)",
        ),
        method=(
            "Soulsby-Whitehouse critical Shields parameter for each tested D50",
            "Exceedance statistics per hydro-pair route segment",
        ),
        outputs=(
            "sediment/noncohesive_mobility_*.{gpkg,parquet,json}",
            "maps/*noncohesive_mobility_capacity.png, *mobility_capacity_profile.png",
        ),
        real_benchmark="PL854 (Anglia A -> LOGGS) with Copernicus Marine forcing and BGS PSA D50 context.",
        tests=(
            "tests/test_noncohesive_mobility.py",
            "tests/test_noncohesive_mobility_map.py",
            "tests/test_grain_size.py",
        ),
        configs=("configs/pl854.yaml",),
        output_patterns=(
            _p(
                "Mobility capacity segments",
                f"processed/{_PL854}/sediment/noncohesive_mobility_*",
                _PL854,
            ),
            _p(
                "Observed D50 context",
                f"processed/{_PL854}/sediment/observed_d50_context.parquet",
                _PL854,
            ),
            _p("Mobility capacity map", f"processed/{_PL854}/maps/*mobility_capacity*.png", _PL854),
        ),
        limitations=(
            "Tested-D50 scenarios only; not a site-specific continuous D50 field.",
            "Mobility capacity is not a transport rate, volume or direction.",
            "Forcing is regional-model derived (Copernicus), not in-situ measured.",
        ),
        next_planned="Transport intensity (Hazard 08) builds on this; no new mobility physics planned.",
        tickets=("MAR-008", "MAR-013"),
        cli_commands=("build-noncohesive-mobility",),
    ),
    Capability(
        id="scour",
        hazard_number=4,
        title="Scour",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Screens pipeline scour-onset susceptibility under combined wave-current forcing for "
            "explicit embedment scenarios and ingests real observed-scour evidence separately."
        ),
        scientific_role=(
            "Track A: model-based onset susceptibility screening (scenario inference).",
            "Track B: source-interpreted observed scour evidence (Sheringham Shoal 2024), kept apart.",
        ),
        inputs=(
            "Near-bed wave orbital velocity and current (MAR-011/012)",
            "Pipe diameter and embedment scenarios (explicit)",
            "Observed scour evidence layers (source interpreted)",
        ),
        method=(
            "Marini et al. (2024) combined wave-current onset criterion",
            "Sensitivity across embedment scenarios; 2018 PL854 condition benchmark context",
        ),
        outputs=(
            "scour/pipeline_scour_*.{gpkg,parquet}, scour_onset_embedment_*",
            "maps/*scour_onset*.png; sheringham_shoal_2024/maps/*observed_scour_evidence.png",
            "scour_poc/report/generic_linear_asset_scour_poc.html",
        ),
        real_benchmark=(
            "PL854 with the official 2018 pipeline condition benchmark; Sheringham Shoal 2024 "
            "XOCEAN observed scour evidence."
        ),
        tests=(
            "tests/test_scour_onset.py",
            "tests/test_scour_onset_map.py",
            "tests/test_scour_susceptibility_poc.py",
            "tests/test_pipeline_condition.py",
        ),
        configs=("configs/pl854.yaml", "configs/sheringham_shoal_2020.yaml"),
        output_patterns=(
            _p("Scour screening products", f"processed/{_PL854}/scour/*", _PL854),
            _p("Scour onset maps", f"processed/{_PL854}/maps/*scour_onset*.png", _PL854),
            _p(
                "Observed scour evidence (2024)",
                "processed/sheringham_shoal_2024/scour/*",
                "sheringham_shoal_2024",
            ),
            _p(
                "Observed scour evidence map",
                "processed/sheringham_shoal_2024/maps/*.png",
                "sheringham_shoal_2024",
            ),
            _p("Scour POC validation", "processed/scour_poc/scour_poc_validation.json", None),
            _p("Scour POC report", "processed/scour_poc/report/*.html", None),
        ),
        limitations=(
            "Onset susceptibility only; no scour depth, extent or time development.",
            "Embedment is an explicit scenario input, not a measured state.",
            "Observed scour evidence is source-interpreted; it does not validate the onset model.",
        ),
        next_planned="Coupling with burial/exposure state; no scour-depth model authorized.",
        tickets=("MAR-014", "MAR-023"),
        cli_commands=("build-scour-onset-screening", "build-scour-susceptibility-poc"),
    ),
    Capability(
        id="burial_exposure",
        hazard_number=5,
        title="Burial / Exposure",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Normalizes an operator depth-of-burial survey into a canonical burial profile, "
            "classifies burial / exposure state and screens exposure susceptibility against an "
            "explicit target-burial margin."
        ),
        scientific_role=(
            "Measured burial state at the survey epoch (after an explicit source-semantics gate).",
            "Target burial and margin are engineering inputs kept distinct from measured burial.",
        ),
        inputs=(
            "Operator burial-profile table with declared sign convention and reference point",
            "Canonical asset route; target burial specification",
        ),
        method=(
            "Source burial-measurement semantics gate before interpretation",
            "Sign / reference-point normalization to canonical cover",
            "Exposure susceptibility screening contract",
        ),
        outputs=(
            "burial/canonical_burial_profile.parquet, burial_exposure_poc.gpkg",
            "maps/*observed_burial_state.png, *burial_kp_profile.png",
            "readiness/burial_profile_readiness.json; report/*burial_exposure_poc.html",
        ),
        real_benchmark="Barrow Offshore Wind Farm 2016 export-cable depth-of-burial survey (Deep BV).",
        tests=("tests/test_burial_exposure_poc.py",),
        configs=("configs/barrow_2016.yaml", "configs/project_manifests/barrow_2016.yaml"),
        output_patterns=(
            _p("Burial products", f"processed/{_BARROW}/burial/*", _BARROW),
            _p("Burial maps", f"processed/{_BARROW}/maps/*.png", _BARROW),
            _p("Burial profile readiness", f"processed/{_BARROW}/readiness/*.json", _BARROW),
            _p("Burial POC report", f"processed/{_BARROW}/report/*.html", _BARROW),
        ),
        limitations=(
            "Single survey epoch; no burial-change or re-exposure prediction.",
            "Exposure susceptibility is a screening contract, not a forecast.",
            "Missing sign convention or reference point stays unresolved; never inferred.",
        ),
        next_planned="Multi-epoch burial change once a second real survey epoch is available.",
        tickets=("MAR-024",),
        cli_commands=("build-burial-exposure-poc",),
    ),
    Capability(
        id="free_span",
        hazard_number=6,
        title="Free Span",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Computes canonical pipe-underside clearance, classifies measured support state, "
            "extracts free-span intervals and screens support-loss susceptibility scenarios; "
            "recovers and reconciles real official free-span evidence onto the canonical route."
        ),
        scientific_role=(
            "Measured support state / clearance geometry (generic engine).",
            "Official free-span registries (NSTA, Table B.1) are source evidence, reconciled not modelled.",
            "Structural assessment is a handoff contract only.",
        ),
        inputs=(
            "Pipe and seabed vertical profiles with declared references",
            "Official free-span records (NSTA UKCS registry, PL854 Table B.1)",
        ),
        method=(
            "Vertical reference normalization; clearance threshold classification",
            "Support-loss scenario screening",
            "Registry reconciliation, temporal provenance and positive-only context audit",
        ),
        outputs=(
            "freespan_poc/freespan/*.json, gis/*.gpkg, synthetic/* (SYNTHETIC case)",
            "pl854/freespan_evidence/*, pl854/pipeline_condition/*, pl854/validation/*",
            "maps/pl854_*freespan*.png, ukcs_observed_pipeline_freespan_evidence.png",
        ),
        real_benchmark=(
            "Real NSTA UKCS pipeline free-span registry (cached) and PL854 official 2018 free-span "
            "events; the generic engine's geometry case is an explicitly SYNTHETIC validation."
        ),
        tests=(
            "tests/test_free_span_poc.py",
            "tests/test_freespan_evidence.py",
            "tests/test_freespan_evidence_map.py",
            "tests/test_freespan_temporal_provenance.py",
            "tests/test_nsta_freespan_provider.py",
            "tests/test_nsta_freespan_reconciliation.py",
            "tests/test_nsta_freespan_reconciliation_map.py",
            "tests/test_freespan_context_audit.py",
            "tests/test_freespan_context_audit_map.py",
            "tests/test_freespan_model_context.py",
        ),
        configs=("configs/pl854.yaml",),
        output_patterns=(
            _p("Free-span contract / handoff", "processed/freespan_poc/freespan/*.json", None),
            _p(
                "Free-span POC validation",
                "processed/freespan_poc/freespan_poc_validation.json",
                None,
            ),
            _p(
                "NSTA registry audit",
                "processed/freespan_poc/nsta_freespan_registry_audit.parquet",
                None,
            ),
            _p(
                "Observed free-span GIS",
                "processed/freespan_poc/gis/nsta_observed_freespan_evidence.gpkg",
                None,
            ),
            _p("Observed free-span maps", "processed/freespan_poc/maps/*.png", None),
            _p(
                "Synthetic engine validation case",
                "processed/freespan_poc/synthetic/*",
                None,
                SYNTHETIC_TEST_FIXTURE,
            ),
            _p(
                "Synthetic support-screening GIS",
                "processed/freespan_poc/gis/synthetic_free_span_support_screening.gpkg",
                None,
                SYNTHETIC_TEST_FIXTURE,
            ),
            _p(
                "PL854 free-span spatial evidence",
                f"processed/{_PL854}/freespan_evidence/*",
                _PL854,
            ),
            _p(
                "PL854 pipeline condition benchmark",
                f"processed/{_PL854}/pipeline_condition/*",
                _PL854,
            ),
            _p("PL854 free-span context audit", f"processed/{_PL854}/validation/*", _PL854),
            _p("PL854 free-span maps", f"processed/{_PL854}/maps/*freespan*.png", _PL854),
            _p("Free-span POC report", "processed/freespan_poc/report/*.html", None),
        ),
        limitations=(
            "Support-loss susceptibility screening; no VIV, fatigue or structural free-span "
            "assessment (handoff contract only).",
            "The generic engine's geometry case is SYNTHETIC; it is not a real pipeline profile.",
            "Official registries are reconciled as evidence; positive-only context audit cannot prove "
            "absence of spans.",
        ),
        next_planned="Structural assessment handoff to a qualified engineering workflow (outside this engine).",
        tickets=("MAR-014A", "MAR-014B", "MAR-014C", "MAR-015", "MAR-025"),
        cli_commands=(
            "build-free-span-poc",
            "build-freespan-spatial-evidence",
            "build-freespan-registry-reconciliation",
            "audit-freespan-context",
        ),
    ),
    Capability(
        id="erosion_deposition",
        hazard_number=7,
        title="Erosion / Deposition",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Computes a canonical DEM-of-Difference between two aligned bathymetry epochs with "
            "common-support masking, misregistration QA and uncertainty evidence, and references "
            "observed seabed change onto a project route."
        ),
        scientific_role=(
            "Observed elevation change between two survey epochs (measured minus measured).",
            "Comparison against a source-produced difference raster is validation, not calibration.",
        ),
        inputs=(
            "Two canonical terrain rasters with declared epochs and vertical references",
            "Optional source-produced difference raster (comparator)",
            "Route + DoD linkage manifest for route evidence",
        ),
        method=(
            "Common-grid alignment and horizontal misregistration QA",
            "Common-valid-support masking; DoD and epoch-interval annualization",
            "Uncertainty evidence inventory and propagation",
        ),
        outputs=(
            "change/delta_bed_elevation_*.tif, annualized_*.tif",
            "gis/seabed_change_poc.gpkg; maps/*seabed_change*.png, *change_qa.png",
            "validation/dod_vs_source_difference_comparison.json; uncertainty/*.json",
        ),
        real_benchmark="Sheringham Shoal 2018 vs 2020 Fugro MBES, with the source MBESDIFF raster as comparator.",
        tests=("tests/test_seabed_change_poc.py", "tests/test_route_change_evidence.py"),
        configs=("configs/sheringham_shoal_2020.yaml",),
        output_patterns=(
            _p("Seabed change rasters", f"processed/{_SS20}/change/*.tif", _SS20),
            _p("Seabed change GIS", f"processed/{_SS20}/gis/seabed_change_poc.gpkg", _SS20),
            _p("Seabed change maps", f"processed/{_SS20}/maps/*change*.png", _SS20),
            _p("DoD vs source comparison", f"processed/{_SS20}/validation/*.json", _SS20),
            _p("Uncertainty evidence", f"processed/{_SS20}/uncertainty/*.json", _SS20),
            _p(
                "Seabed change POC validation",
                f"processed/{_SS20}/seabed_change_poc_validation.json",
                _SS20,
            ),
            _p(
                "Seabed change POC report",
                f"processed/{_SS20}/report/*seabed_change_poc.html",
                _SS20,
            ),
        ),
        limitations=(
            "Two-epoch difference only; annualized change divides by the epoch interval and is not a "
            "trend or forecast.",
            "No sediment budget, volume or transport pathway inference.",
            "Change below the propagated uncertainty is reported as not significant, not as zero.",
        ),
        next_planned="Route-referenced change evidence for operator projects (MAR-029 done); multi-epoch series when available.",
        tickets=("MAR-021", "MAR-029"),
        cli_commands=("build-seabed-change-poc", "build-route-seabed-change-evidence"),
    ),
    Capability(
        id="transport_intensity",
        hazard_number=8,
        title="Sediment Transport Intensity",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Derives a dimensionless relative excess-Shields transport-potential intensity from "
            "the MAR-013 mobility screening across tested D50 scenarios."
        ),
        scientific_role=(
            "Relative transport-potential intensity (dimensionless); scenario inference.",
        ),
        inputs=(
            "MAR-013 noncohesive mobility outputs (tested D50 scenarios)",
            "Combined bed shear",
        ),
        method=(
            "Relative excess Shields parameter per segment and scenario",
            "Scenario matrix reporting",
        ),
        outputs=(
            "sediment/noncohesive_transport_intensity_*.{gpkg,parquet,json}",
            "maps/noncohesive_transport_intensity_scenario_matrix.png",
        ),
        real_benchmark="PL854 route with Copernicus Marine forcing (same evidence base as Hazard 03).",
        tests=("tests/test_transport_intensity.py",),
        configs=("configs/pl854.yaml",),
        output_patterns=(
            _p(
                "Transport intensity products",
                f"processed/{_PL854}/sediment/noncohesive_transport_intensity_*",
                _PL854,
            ),
            _p(
                "Scenario matrix figure",
                f"processed/{_PL854}/maps/noncohesive_transport_intensity_scenario_matrix.png",
                _PL854,
            ),
        ),
        limitations=(
            "Dimensionless relative excess Shields; not volumetric or mass transport.",
            "No net transport direction.",
            "Inherits the tested-D50 scenario limitation of Hazard 03.",
        ),
        next_planned="No transport-rate formulation is authorized; intensity remains relative.",
        tickets=("MAR-030",),
        cli_commands=("build-sediment-transport-intensity",),
    ),
    Capability(
        id="slope_instability",
        hazard_number=9,
        title="Slope Instability",
        maturity=QUALIFIED_POC,
        what_it_does=(
            "Screens submarine slope-instability susceptibility from real terrain demand (slope at "
            "explicit windows) against explicit hypothetical geotechnical scenarios declared in a "
            "typed scenario manifest."
        ),
        scientific_role=(
            "Terrain demand is REAL (measured slope).",
            "Factor of safety exists only for explicit, hypothetical geotechnical scenarios.",
        ),
        inputs=(
            "Canonical terrain raster (MAR-020) in a metric projected CRS",
            "Slope-stability scenario manifest (explicit strength / unit weight scenarios)",
        ),
        method=(
            "Slope at 10 m and 50 m windows; normalized strength demand",
            "Infinite-slope style scenario screening as declared by the manifest",
            "Readiness vocabulary, no numeric hazard score",
        ),
        outputs=(
            "slope_stability/slope_*_deg.tif, normalized_strength_demand_*.tif",
            "slope_stability/slope_instability_{contract,readiness,screening_metadata}.json",
            "maps/*_slope_instability_screening.png",
        ),
        real_benchmark="Sheringham Shoal 2020 terrain (real demand); PL854 regional context (readiness only).",
        tests=("tests/test_slope_stability.py",),
        configs=("configs/sheringham_shoal_2020.yaml", "configs/pl854.yaml"),
        output_patterns=(
            _p("Slope-instability rasters", f"processed/{_SS20}/slope_stability/*.tif", _SS20),
            _p(
                "Slope-instability contract / readiness",
                f"processed/{_SS20}/slope_stability/*.json",
                _SS20,
            ),
            _p(
                "Slope-instability figure", f"processed/{_SS20}/maps/*slope_instability*.png", _SS20
            ),
            _p(
                "PL854 slope-instability readiness",
                f"processed/{_PL854}/slope_stability/*.json",
                _PL854,
            ),
        ),
        limitations=(
            "Terrain demand real; FoS only for explicit hypothetical geotechnical scenarios.",
            "No runout, no trigger mechanics, no seismic or wave loading.",
            "Foot-based or non-metric CRS is rejected, never converted silently.",
        ),
        next_planned="Site-specific geotechnical parameters when measured CPT-derived strength becomes available.",
        tickets=("MAR-031", "MAR-031A"),
        cli_commands=("build-slope-instability-screening",),
    ),
    Capability(
        id="liquefaction",
        hazard_number=10,
        title="Liquefaction",
        maturity=FOUNDATION_READY,
        what_it_does=(
            "Acquires, inventories and normalizes real offshore CPT/CPTU evidence into a canonical "
            "measurement profile with verified product identity, schema, lineage and CRS integrity, "
            "and reports liquefaction-INPUT readiness. No liquefaction physics is computed."
        ),
        scientific_role=(
            "Measured CPT/CPTU evidence (qc, fs, u2) and its readiness as a future resistance input.",
            "Earthquake-induced and wave-induced mechanisms kept separate; both NOT_EVALUABLE.",
        ),
        inputs=("Marine Data Exchange CPT package (GEO CSV logs) and Part B/C reports",),
        method=(
            "Strict source-specific parse with explicit channel and unit declarations",
            "Canonical CPT_CANONICAL_PROFILE_V1 product marker, full schema and source_id lineage",
            "Semantic CRS check against EPSG:32631 (metre)",
        ),
        outputs=(
            "cpt_measurements.parquet, cpt_tests.parquet, cpt_locations.gpkg",
            "cpt_metadata.json, cpt_readiness.json, liquefaction_readiness.json",
        ),
        real_benchmark="Sheringham Shoal 2008 GEO CPT/CPTU investigation (TCE-1964): 138,514 rows, 100 tests.",
        tests=("tests/test_cpt_evidence.py",),
        configs=("configs/geotechnical/sheringham_shoal_2008_cptu.yaml",),
        output_patterns=(
            _p("Canonical CPT measurements", f"processed/{_CPTU}/cpt_measurements.parquet", _CPTU),
            _p("CPT tests table", f"processed/{_CPTU}/cpt_tests.parquet", _CPTU),
            _p("CPT locations", f"processed/{_CPTU}/cpt_locations.gpkg", _CPTU),
            _p("CPT metadata / readiness", f"processed/{_CPTU}/*.json", _CPTU),
            _p("CPT source evidence (interim)", f"interim/{_CPTU}/*.json", _CPTU),
        ),
        limitations=(
            "Evidence foundation only.",
            "No CSR, no CRR, no triggering result, no factor of safety.",
            "qt is never derived from qc; it stays null where the source did not provide it.",
        ),
        next_planned="CPT resistance normalization, CRR/CSR and earthquake triggering FS under explicit method authority (MAR-033+).",
        tickets=("MAR-032", "MAR-032A", "MAR-032B"),
        cli_commands=("build-cpt-evidence-poc",),
    ),
    Capability(
        id="shallow_gas",
        hazard_number=11,
        title="Shallow Gas",
        maturity=UNDER_CONSTRUCTION,
        what_it_does="Planned: shallow-gas indicator evidence and susceptibility screening. Nothing implemented.",
        scientific_role=("Not defined yet.",),
        inputs=("Not defined yet.",),
        method=("Not defined yet.",),
        outputs=(),
        real_benchmark="None.",
        tests=(),
        configs=(),
        output_patterns=(),
        limitations=("Under construction; no capability, no outputs, no tests.",),
        next_planned="Scope and method authority to be defined in a future ticket.",
    ),
    Capability(
        id="fault_structural_crossing",
        hazard_number=12,
        title="Fault / Structural Crossing",
        maturity=UNDER_CONSTRUCTION,
        what_it_does="Planned: fault and structural-feature crossing evidence along a route. Nothing implemented.",
        scientific_role=("Not defined yet.",),
        inputs=("Not defined yet.",),
        method=("Not defined yet.",),
        outputs=(),
        real_benchmark="None.",
        tests=(),
        configs=(),
        output_patterns=(),
        limitations=("Under construction; no capability, no outputs, no tests.",),
        next_planned="Scope and method authority to be defined in a future ticket.",
    ),
    Capability(
        id="boulder_hardground",
        hazard_number=13,
        title="Boulder / Hardground",
        maturity=PARTIAL,
        what_it_does=(
            "Partial: substrate-class evidence from BGS seabed sediment products (MAR-008) and "
            "high-resolution ruggedness / local-relief derivatives (MAR-020) exist as inputs. No "
            "boulder detection, hardground classification or clearance product exists."
        ),
        scientific_role=("Existing inputs only; no dedicated boulder / hardground product.",),
        inputs=(
            "BGS 250k seabed sediment classes (source interpreted)",
            "Terrain ruggedness rasters",
        ),
        method=("Not defined yet.",),
        outputs=(),
        real_benchmark="None for this hazard (inputs proven under Hazards 01 and 03 only).",
        tests=("tests/test_sediment_evidence.py", "tests/test_bgs_sediment.py"),
        configs=("configs/pl854.yaml",),
        output_patterns=(
            _p(
                "Sediment evidence (input only)",
                f"processed/{_PL854}/sediment/chainage_sediment_evidence.parquet",
                _PL854,
            ),
        ),
        limitations=(
            "No boulder or hardground product; listed inputs are not a hazard result.",
            "Substrate classes are source interpretations, not measured hardness.",
        ),
        next_planned="Boulder / hardground evidence contract to be defined in a future ticket.",
    ),
    Capability(
        id="infrastructure_crossings",
        hazard_number=14,
        title="Infrastructure / Crossings",
        maturity=PARTIAL,
        what_it_does=(
            "Partial: canonical route ingestion, a project route-reference grid with explicit "
            "cross-asset linkage (MAR-027) and anthropogenic / infrastructure context inventories "
            "exist. No crossing-geometry or crossing-conflict product exists."
        ),
        scientific_role=("Project geometry and linkage only; no crossing assessment.",),
        inputs=(
            "Canonical project route",
            "Registered project assets",
            "Infrastructure context inventories",
        ),
        method=("Not defined yet.",),
        outputs=(),
        real_benchmark="None for this hazard (route linkage proven under MAR-027 only).",
        tests=("tests/test_project_route_reference.py", "tests/test_nsta.py"),
        configs=("configs/project_manifests/pl854_route_reference.yaml",),
        output_patterns=(
            _p(
                "Project route reference",
                f"processed/{_PL854}/project/project_route_reference.gpkg",
                _PL854,
            ),
            _p("Project model", f"processed/{_PL854}/project/canonical_project_model.json", _PL854),
        ),
        limitations=(
            "No crossing detection, separation or conflict product.",
            "Route linkage is geometry bookkeeping, not an engineering crossing assessment.",
        ),
        next_planned="Crossing-geometry evidence contract to be defined in a future ticket.",
    ),
    Capability(
        id="route_suitability",
        hazard_number=15,
        title="Route Constraint / Suitability",
        maturity=UNDER_CONSTRUCTION,
        what_it_does=(
            "Planned: route-constraint and suitability synthesis across hazard families. Nothing "
            "implemented; existing per-asset readiness is not a suitability statement."
        ),
        scientific_role=("Not defined yet.",),
        inputs=("Not defined yet.",),
        method=("Not defined yet.",),
        outputs=(),
        real_benchmark="None.",
        tests=(),
        configs=(),
        output_patterns=(),
        limitations=(
            "Under construction; no capability, no outputs, no tests.",
            "No universal marine-project readiness or suitability claim is permitted.",
        ),
        next_planned="Scope and method authority to be defined in a future ticket.",
    ),
)

# --- supporting engine capabilities (UI-001 Section 8, second list) --------------------------------

SUPPORTING: tuple[SupportingCapability, ...] = (
    SupportingCapability(
        "route_ingestion",
        "Route ingestion",
        QUALIFIED_POC,
        "Canonical pipeline geometry ingested from the NSTA offshore pipeline service.",
        ("tests/test_nsta.py", "tests/test_config.py"),
        ("MAR-001",),
        ("ingest-pipeline",),
    ),
    SupportingCapability(
        "aoi",
        "AOI",
        QUALIFIED_POC,
        "Corridor area-of-interest generation from the canonical pipeline.",
        ("tests/test_aoi.py",),
        ("MAR-002",),
        ("build-aoi",),
    ),
    SupportingCapability(
        "chainage_kp",
        "Chainage / KP",
        QUALIFIED_POC,
        "25 m chainage / KP linear-reference system along the canonical pipeline.",
        ("tests/test_chainage.py",),
        ("MAR-002",),
        ("build-chainage",),
    ),
    SupportingCapability(
        "bathymetry_discovery",
        "Bathymetry discovery",
        QUALIFIED_POC,
        "Discovery, spatial verification and ranking of approved bathymetry sources; CDI provenance.",
        (
            "tests/test_bathymetry_sources.py",
            "tests/test_bathymetry_inventory.py",
            "tests/test_cdi.py",
            "tests/test_source_resolution.py",
        ),
        ("MAR-005", "MAR-006B", "MAR-006C"),
        ("discover-bathymetry", "resolve-bathymetry-sources"),
    ),
    SupportingCapability(
        "bathymetry_acquisition",
        "Bathymetry acquisition",
        QUALIFIED_POC,
        "Idempotent, checksummed raw acquisition bookkeeping and high-resolution survey access audit.",
        (
            "tests/test_bathymetry_acquisition.py",
            "tests/test_highres_seabed_survey_inventory.py",
            "tests/test_highres_seabed_survey_inventory_map.py",
        ),
        ("MAR-006", "MAR-016"),
        ("fetch-bathymetry", "inventory-highres-seabed-data"),
    ),
    SupportingCapability(
        "canonical_bathymetry",
        "Canonical bathymetry",
        QUALIFIED_POC,
        "Canonical EMODnet baseline DTM with verified depth semantics and regional morphology context.",
        ("tests/test_bathymetry_canonical.py", "tests/test_morphology_regional.py"),
        ("MAR-006", "MAR-007"),
        ("build-bathymetry", "build-regional-morphology"),
    ),
    SupportingCapability(
        "metocean_currents",
        "Metocean currents",
        QUALIFIED_POC,
        "Copernicus Marine current evidence, deepest-valid-level selection and near-bed normalization.",
        (
            "tests/test_current.py",
            "tests/test_current_map.py",
            "tests/test_current_normalization.py",
            "tests/test_metocean_evidence.py",
            "tests/test_metocean_acquisition.py",
            "tests/test_copernicus_provider.py",
        ),
        ("MAR-009", "MAR-009A", "MAR-010"),
        ("build-metocean-evidence", "build-current-normalization"),
    ),
    SupportingCapability(
        "wave_orbital",
        "Wave orbital forcing",
        QUALIFIED_POC,
        "Wave-only spectral near-bed orbital velocity (Soulsby & Smallman) along the route.",
        ("tests/test_wave.py", "tests/test_wave_orbital.py", "tests/test_wave_orbital_map.py"),
        ("MAR-011", "MAR-011A"),
        ("build-wave-orbital-forcing",),
    ),
    SupportingCapability(
        "combined_bed_shear",
        "Combined wave-current bed shear",
        QUALIFIED_POC,
        "Soulsby algebraic wave-current bed shear stress sensitivity.",
        ("tests/test_combined_bed_shear.py", "tests/test_combined_bed_shear_map.py"),
        ("MAR-012",),
        ("build-combined-bed-shear",),
    ),
    SupportingCapability(
        "sediment_evidence",
        "Sediment evidence",
        QUALIFIED_POC,
        "BGS PSA, 250k and predictive sediment evidence base with D10/D50/D90 derivation; no physics.",
        (
            "tests/test_sediment_evidence.py",
            "tests/test_bgs_sediment.py",
            "tests/test_grain_size.py",
        ),
        ("MAR-008",),
        ("build-sediment-evidence",),
    ),
    SupportingCapability(
        "project_ingestion",
        "Generic project ingestion",
        QUALIFIED_POC,
        "Operator project manifest, content-based asset identity, declared-vs-observed metadata, "
        "intrinsic vs effective readiness.",
        ("tests/test_project_ingestion.py",),
        ("MAR-026", "MAR-026A"),
        ("build-project-readiness",),
    ),
    SupportingCapability(
        "crs_integrity",
        "CRS integrity",
        QUALIFIED_POC,
        "Semantic declared-vs-observed CRS comparison; material conflicts block effective readiness; "
        "metric projected working CRS enforced.",
        ("tests/test_project_ingestion.py",),
        ("MAR-026A", "MAR-031A", "MAR-032A"),
    ),
    SupportingCapability(
        "route_linkage",
        "Cross-asset route linkage",
        QUALIFIED_POC,
        "Canonical project model with an explicit route-reference grid and per-asset linkage.",
        ("tests/test_project_route_reference.py",),
        ("MAR-027",),
        ("build-project-model",),
    ),
    SupportingCapability(
        "evidence_atlas",
        "Engineering Evidence Atlas",
        QUALIFIED_POC,
        "Map-first PL854 engineering evidence atlas and external-reviewer POC package.",
        ("tests/test_evidence_atlas.py", "tests/test_marine_poc.py"),
        ("MAR-018", "MAR-019"),
        ("build-engineering-evidence-atlas", "build-marine-poc-review-package"),
    ),
    SupportingCapability(
        "ci_audit",
        "CI / audit",
        QUALIFIED_POC,
        "GitHub Actions gate (.github/workflows/ci.yml): lock check, frozen sync, format, lint, "
        "offline pytest, dependency audit. Real-data regression stays local.",
        ("tests/test_imports.py",),
        ("MAR-028",),
    ),
    SupportingCapability(
        "cpt_evidence",
        "CPT/CPTU evidence",
        FOUNDATION_READY,
        "Real Sheringham 2008 CPT/CPTU evidence with verified canonical identity, schema and lineage; "
        "the input foundation of Hazard 10.",
        ("tests/test_cpt_evidence.py",),
        ("MAR-032", "MAR-032A", "MAR-032B"),
        ("build-cpt-evidence-poc",),
    ),
)

# --- project / demo contexts (UI-001 Section 7) ----------------------------------------------------

PROJECTS: tuple[Project, ...] = (
    Project(
        id=_PL854,
        title="PL854 (Anglia A -> LOGGS)",
        description=(
            "First fully worked study case: NSTA route, EMODnet baseline bathymetry, Copernicus "
            "Marine forcing, BGS sediment evidence, official free-span and condition records."
        ),
        evidence_kind=REAL_SOURCE,
        configs=("configs/pl854.yaml", "configs/project_manifests/pl854_route_reference.yaml"),
        processed_dirs=("pl854", "freespan_poc", "scour_poc"),
        interim_dirs=("pl854",),
        readiness_files=(
            "processed/pl854/validation/freespan_evidence_readiness.json",
            "processed/pl854/slope_stability/slope_instability_readiness.json",
            "processed/pl854/project/project_model_validation.json",
        ),
        capability_ids=(
            "sediment_mobility",
            "scour",
            "free_span",
            "transport_intensity",
            "slope_instability",
            "boulder_hardground",
            "infrastructure_crossings",
        ),
        source_note="Regional (100 m EMODnet) baseline; no high-resolution PL854 survey is available locally.",
    ),
    Project(
        id=_SS20,
        title="Sheringham Shoal 2020",
        description=(
            "Independent real-data demonstration: Fugro 2018 and 2020 MBES surveys (terrain, change, "
            "bedforms, slope instability) and the 2024 XOCEAN observed-scour evidence."
        ),
        evidence_kind=REAL_SOURCE,
        configs=(
            "configs/sheringham_shoal_2020.yaml",
            "configs/project_manifests/sheringham_shoal_2020.yaml",
        ),
        processed_dirs=("sheringham_shoal_2020", "sheringham_shoal_2024"),
        interim_dirs=(),
        readiness_files=(
            "processed/sheringham_shoal_2020/readiness/bathymetry_readiness.json",
            "processed/sheringham_shoal_2020/readiness/bathymetry_2018_readiness.json",
            "processed/sheringham_shoal_2020/readiness/bathymetry_2020_readiness.json",
            "processed/sheringham_shoal_2020/slope_stability/slope_instability_readiness.json",
            "processed/sheringham_shoal_2020/project/project_readiness.json",
        ),
        capability_ids=("terrain", "bedforms", "erosion_deposition", "scour", "slope_instability"),
        source_note="Raw surveys are gitignored; canonical rasters exceed 250 million cells -- metadata only on load.",
    ),
    Project(
        id=_CPTU,
        title="Sheringham Shoal 2008 CPTU",
        description="Real GEO CPT/CPTU geotechnical investigation (Marine Data Exchange TCE-1964).",
        evidence_kind=REAL_SOURCE,
        configs=("configs/geotechnical/sheringham_shoal_2008_cptu.yaml",),
        processed_dirs=("sheringham_shoal_2008_cptu",),
        interim_dirs=("sheringham_shoal_2008_cptu",),
        readiness_files=(
            "processed/sheringham_shoal_2008_cptu/cpt_readiness.json",
            "processed/sheringham_shoal_2008_cptu/liquefaction_readiness.json",
        ),
        capability_ids=("liquefaction",),
        source_note="Liquefaction physics is not computed; only evidence readiness exists.",
    ),
    Project(
        id=_BARROW,
        title="Barrow 2016",
        description="Barrow Offshore Wind Farm export-cable depth-of-burial survey (Deep BV, 2016).",
        evidence_kind=REAL_SOURCE,
        configs=("configs/barrow_2016.yaml", "configs/project_manifests/barrow_2016.yaml"),
        processed_dirs=("barrow_2016",),
        interim_dirs=(),
        readiness_files=(
            "processed/barrow_2016/readiness/burial_profile_readiness.json",
            "processed/barrow_2016/project/project_readiness.json",
        ),
        capability_ids=("burial_exposure",),
    ),
)

# --- liquefaction card (UI-001 Section 12) ---------------------------------------------------------

LIQUEFACTION_CARD = {
    "title": "Liquefaction",
    "maturity": FOUNDATION_READY,
    "available": (
        "Real Sheringham CPT/CPTU evidence",
        "138,514 canonical measurement rows",
        "100 parsed CPT tests",
        "qc",
        "fs",
        "u2",
        "canonical source identity",
        "CRS integrity",
        "schema + lineage integrity",
    ),
    "under_construction": (
        "CPT resistance normalization",
        "CRR",
        "CSR",
        "earthquake triggering FS",
        "magnitude scaling",
        "stress corrections",
    ),
    "not_implemented": (
        "LPI",
        "settlement",
        "lateral spreading",
        "wave-induced pore-pressure response",
    ),
}


# --- lookups -----------------------------------------------------------------------------------


def hazard_by_id(capability_id: str) -> Capability:
    for capability in HAZARDS:
        if capability.id == capability_id:
            return capability
    raise KeyError(capability_id)


def project_by_id(project_id: str) -> Project:
    for project in PROJECTS:
        if project.id == project_id:
            return project
    raise KeyError(project_id)


def maturity_counts() -> dict[str, int]:
    """Summary-strip counts computed from the hazard registry (never hard-coded)."""

    counter = Counter(c.maturity for c in HAZARDS)
    return {m: counter.get(m, 0) for m in MATURITY_VOCABULARY}


def all_registered_test_files() -> tuple[str, ...]:
    files: set[str] = set()
    for capability in HAZARDS:
        files.update(capability.tests)
    for supporting in SUPPORTING:
        files.update(supporting.tests)
    return tuple(sorted(files))


def capabilities_for_test_file(test_file: str) -> tuple[str, ...]:
    ids = [c.id for c in HAZARDS if test_file in c.tests]
    ids += [s.id for s in SUPPORTING if test_file in s.tests]
    return tuple(ids)


def capabilities_for_project(project_id: str) -> tuple[Capability, ...]:
    project = project_by_id(project_id)
    return tuple(hazard_by_id(cid) for cid in project.capability_ids)
