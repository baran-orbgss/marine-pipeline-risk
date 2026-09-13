"""Guided View story data (UI-002).

Pure presentation data: which stages exist for each project, in what order, with what
plain-language explanation, and which real local files to try in priority order. Nothing here
computes, thresholds, reinterprets or fabricates a scientific result -- every referenced file is
existence-checked through `output_inspector`/`map_view` at render time, never assumed present.

MAR ticket numbers and CLI commands are deliberately NOT stored on a `GuidedStage`. The
"Show technical details" expander in the UI pulls those from `capability_registry` (via
`capability_id`) only when rendering technical detail, so Guided View text can never leak a
ticket number by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ui import capability_registry as registry

__all__ = [
    "VISUAL_KINDS",
    "BADGE_AVAILABLE",
    "BADGE_FOUNDATION",
    "BADGE_PARTIAL",
    "BADGE_COMING_SOON",
    "BADGE_VOCABULARY",
    "VisualCandidate",
    "GuidedStage",
    "ProjectStory",
    "STORIES",
    "story_for_project",
    "stage_badge_maturity",
]

VISUAL_KINDS = ("png", "vector", "raster", "chart", "html", "readiness")

BADGE_AVAILABLE = "AVAILABLE"
BADGE_FOUNDATION = "FOUNDATION"
BADGE_PARTIAL = "PARTIAL"
BADGE_COMING_SOON = "COMING SOON"
BADGE_VOCABULARY = (BADGE_AVAILABLE, BADGE_FOUNDATION, BADGE_PARTIAL, BADGE_COMING_SOON)

_BADGE_BY_MATURITY = {
    registry.QUALIFIED_POC: BADGE_AVAILABLE,
    registry.FOUNDATION_READY: BADGE_FOUNDATION,
    registry.PARTIAL: BADGE_PARTIAL,
    registry.UNDER_CONSTRUCTION: BADGE_COMING_SOON,
}


def _validate_glob(glob: str) -> None:
    if glob.startswith(("/", "\\")) or ".." in glob.split("/"):
        raise ValueError(f"visual glob must be relative to data/ without '..': {glob}")


@dataclass(frozen=True)
class VisualCandidate:
    """One thing a guided stage can try to show. `kind` decides which fields apply:

    * "png" / "html": `glob` is a single glob relative to `data/`.
    * "vector": `layers` is an ordered list of (glob, layer_name_or_None, display_label) to
      compose into one scene.
    * "raster": `raster_glob` is a single glob to a GeoTIFF (rendered as a bounded preview).
    * "chart": `chart_id` names a chart `map_view` knows how to build (e.g. "cpt_profile").
    * "readiness": `glob` is a single glob to an engine-written readiness JSON.
    """

    kind: str
    label: str
    glob: str | None = None
    layers: tuple[tuple[str, str | None, str], ...] = ()
    raster_glob: str | None = None
    chart_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in VISUAL_KINDS:
            raise ValueError(f"unknown visual kind {self.kind!r}")
        if self.glob:
            _validate_glob(self.glob)
        if self.raster_glob:
            _validate_glob(self.raster_glob)
        for g, _layer, _display_label in self.layers:
            _validate_glob(g)


@dataclass(frozen=True)
class GuidedStage:
    key: str
    label: str
    what_you_are_looking_at: str
    what_it_means: str
    what_it_does_not_mean: str
    visuals: tuple[VisualCandidate, ...] = ()
    capability_id: str | None = None
    optional_layers: tuple[VisualCandidate, ...] = ()
    limitations: tuple[str, ...] = ()
    not_applicable_note: str | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.key.isidentifier():
            raise ValueError(f"stage key must be a plain snake_case identifier: {self.key!r}")
        for name, text in (
            ("what_you_are_looking_at", self.what_you_are_looking_at),
            ("what_it_means", self.what_it_means),
            ("what_it_does_not_mean", self.what_it_does_not_mean),
        ):
            if not text:
                raise ValueError(f"{self.key}: {name} is mandatory")
        if self.capability_id is not None:
            registry.hazard_by_id(self.capability_id)  # raises KeyError if unknown


@dataclass(frozen=True)
class ProjectStory:
    project_id: str
    stages: tuple[GuidedStage, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        registry.project_by_id(self.project_id)  # raises KeyError if unknown
        if not self.stages:
            raise ValueError(f"{self.project_id}: a story needs at least one stage")
        keys = [s.key for s in self.stages]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.project_id}: duplicate stage keys")


def stage_badge_maturity(stage: GuidedStage) -> str | None:
    """Compact guided-badge value, or None when a stage has no tied hazard capability (the
    caller then shows an evidence chip instead of a maturity badge)."""

    if stage.capability_id is None:
        return None
    return _BADGE_BY_MATURITY[registry.hazard_by_id(stage.capability_id).maturity]


# --- PL854 -----------------------------------------------------------------------------------

_PL854_MAPS = "processed/pl854/maps"

_PL854_STAGES = (
    GuidedStage(
        key="route_area",
        label="Route & Area",
        what_you_are_looking_at="Pipeline route and study area in the southern North Sea.",
        what_it_means=(
            "Shows the location of the pipeline and the area used for environmental and "
            "geohazard assessment."
        ),
        what_it_does_not_mean=(
            "This is not a risk map. It only shows the route and the area used for analysis."
        ),
        visuals=(
            VisualCandidate(
                kind="vector",
                label="Route & area",
                layers=(
                    ("processed/pl854/pipeline.gpkg", "pipeline", "Pipeline route"),
                    ("processed/pl854/aoi.gpkg", "study_aoi", "Study area (AOI)"),
                    ("processed/pl854/chainage_25m.gpkg", "chainage_points", "KP markers"),
                ),
            ),
        ),
    ),
    GuidedStage(
        key="bathymetry",
        label="Bathymetry",
        what_you_are_looking_at=(
            "Seabed depth across the study area, from a regional 100 m baseline survey."
        ),
        what_it_means="This is the seabed shape the engine used as terrain input for this route.",
        what_it_does_not_mean=(
            "This is a regional baseline, not a high-resolution route survey -- no PL854-"
            "specific high-resolution bathymetry is available locally."
        ),
        visuals=(
            VisualCandidate(
                kind="raster",
                label="Bathymetry (regional baseline)",
                raster_glob="processed/pl854/bathymetry/emodnet_baseline_lat_100m.tif",
            ),
        ),
    ),
    GuidedStage(
        key="currents_waves",
        label="Currents & Waves",
        what_you_are_looking_at=(
            "Modelled current speed and wave orbital velocity at the seabed along the route."
        ),
        what_it_means=(
            "These are the two physical forcings that drive sediment movement and bed stress "
            "at each route section."
        ),
        what_it_does_not_mean=(
            "These are model outputs at reference locations, not a live or measured metocean feed."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Currents",
                glob=f"{_PL854_MAPS}/pl854_reference_current_forcing.png",
            ),
            VisualCandidate(
                kind="png",
                label="Wave orbital motion",
                glob=f"{_PL854_MAPS}/pl854_wave_orbital_forcing.png",
            ),
        ),
        optional_layers=(
            VisualCandidate(
                kind="vector",
                label="Current reference segments",
                layers=(
                    ("processed/pl854/metocean/current_reference_segments.gpkg", None, "Currents"),
                ),
            ),
            VisualCandidate(
                kind="vector",
                label="Wave orbital reference segments",
                layers=(
                    (
                        "processed/pl854/metocean/wave_orbital_reference_segments.gpkg",
                        None,
                        "Waves",
                    ),
                ),
            ),
        ),
    ),
    GuidedStage(
        key="bed_shear",
        label="Bed Shear",
        what_you_are_looking_at=(
            "Combined wave-and-current bed shear stress along the route, at each section."
        ),
        what_it_means=(
            "This is how hard the water is pushing on the seabed once currents and waves are "
            "combined."
        ),
        what_it_does_not_mean=(
            "This is a physical stress value, not a sediment-movement prediction by itself."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Combined bed shear",
                glob=f"{_PL854_MAPS}/pl854_combined_bed_shear_sensitivity.png",
            ),
        ),
        optional_layers=(
            VisualCandidate(
                kind="vector",
                label="Combined bed shear segments",
                layers=(
                    (
                        "processed/pl854/metocean/combined_bed_shear_segments.gpkg",
                        None,
                        "Combined bed shear",
                    ),
                ),
            ),
        ),
    ),
    GuidedStage(
        key="sediment_mobility",
        label="Sediment Mobility",
        what_you_are_looking_at=(
            "Route sections where wave-current bed stress exceeds tested grain-size movement "
            "thresholds."
        ),
        what_it_means=(
            "The seabed here has greater capacity to mobilise a tested noncohesive grain-size "
            "scenario."
        ),
        what_it_does_not_mean=(
            "This is not a sediment transport rate and does not predict how many metres the "
            "seabed will move."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Sediment mobility capacity",
                glob=f"{_PL854_MAPS}/pl854_noncohesive_mobility_capacity.png",
            ),
            VisualCandidate(
                kind="png",
                label="Mobility capacity profile",
                glob=f"{_PL854_MAPS}/pl854_mobility_capacity_profile.png",
            ),
        ),
        capability_id="sediment_mobility",
        optional_layers=(
            VisualCandidate(
                kind="vector",
                label="Mobility capacity segments",
                layers=(
                    (
                        "processed/pl854/sediment/noncohesive_mobility_capacity_segments.gpkg",
                        "noncohesive_mobility_capacity_segments",
                        "Mobility capacity",
                    ),
                ),
            ),
        ),
        limitations=(
            "Sediment evidence (BGS surface samples and predictive sediment class) supports "
            "this screening as context; it is not itself a mobility result.",
        ),
    ),
    GuidedStage(
        key="scour",
        label="Scour",
        what_you_are_looking_at=(
            "Scenario-based scour-onset screening: how deep the pipe would need to be "
            "embedded to avoid onset, under a range of tested burial scenarios."
        ),
        what_it_means=(
            "This shows which sections carry higher scour-onset susceptibility under the "
            "tested embedment scenarios."
        ),
        what_it_does_not_mean=(
            "This is a tested-scenario screening envelope, not a site-specific scour-depth "
            "prediction."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Scour onset susceptibility",
                glob=f"{_PL854_MAPS}/pl854_scour_onset_susceptibility_screening.png",
            ),
            VisualCandidate(
                kind="png",
                label="Scour onset scenario envelope",
                glob=f"{_PL854_MAPS}/pl854_scour_onset_scenario_envelope.png",
            ),
        ),
        capability_id="scour",
        optional_layers=(
            VisualCandidate(
                kind="vector",
                label="Tested embedment scenario envelope",
                layers=(
                    (
                        "processed/pl854/scour/pipeline_scour_screening.gpkg",
                        "tested_embedment_scenario_envelope",
                        "Tested embedment scenarios",
                    ),
                ),
            ),
        ),
        limitations=(
            "A dedicated site-specific scour-susceptibility screening is registered but reports "
            "SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_PROFILE for every "
            "section on this route (no embedment profile) -- only the tested-scenario envelope "
            "above is shown.",
        ),
    ),
    GuidedStage(
        key="burial_exposure",
        label="Burial / Exposure",
        what_you_are_looking_at="Burial and exposure screening for this pipeline.",
        what_it_means=(
            "Burial/exposure assessment is implemented and proven on real data elsewhere "
            "(Barrow 2016)."
        ),
        what_it_does_not_mean=(
            "It has not been run for the PL854 route in this workbench -- no local output "
            "exists for this project."
        ),
        visuals=(),
        capability_id="burial_exposure",
        not_applicable_note=(
            "See the Barrow 2016 project for a real burial/exposure demonstration on this "
            "capability."
        ),
    ),
    GuidedStage(
        key="free_span",
        label="Free Span",
        what_you_are_looking_at=(
            "Observed and historical free-span events along the pipeline, from operator "
            "survey records."
        ),
        what_it_means=(
            "These are sections recorded as unsupported above the seabed in operator surveys."
        ),
        what_it_does_not_mean=(
            "These are observed/historical records, not a live structural-integrity assessment."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Observed free spans (2018)",
                glob=f"{_PL854_MAPS}/pl854_observed_freespans_2018.png",
            ),
            VisualCandidate(
                kind="png",
                label="Historical free spans (2012-2018)",
                glob=f"{_PL854_MAPS}/pl854_historical_freespans_2012_2018.png",
            ),
        ),
        capability_id="free_span",
        optional_layers=(
            VisualCandidate(
                kind="vector",
                label="Historical free-span evidence",
                layers=(
                    (
                        "processed/pl854/freespan_evidence/anglia_freespan_spatial_evidence.gpkg",
                        "historical_freespans",
                        "Historical free spans",
                    ),
                ),
            ),
        ),
    ),
    GuidedStage(
        key="transport_intensity",
        label="Transport Intensity",
        what_you_are_looking_at=(
            "A relative sediment transport intensity scenario matrix along the route."
        ),
        what_it_means=(
            "This compares how transport-active different sections are, across tested "
            "scenarios, relative to each other."
        ),
        what_it_does_not_mean=(
            "This is not an absolute transport rate and does not predict a volume of sediment "
            "moved."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Transport intensity scenario matrix",
                glob=f"{_PL854_MAPS}/noncohesive_transport_intensity_scenario_matrix.png",
            ),
        ),
        capability_id="transport_intensity",
        optional_layers=(
            VisualCandidate(
                kind="vector",
                label="Transport intensity segments",
                layers=(
                    (
                        "processed/pl854/sediment/noncohesive_transport_intensity_segments.gpkg",
                        None,
                        "Transport intensity",
                    ),
                ),
            ),
        ),
    ),
    GuidedStage(
        key="evidence_summary",
        label="Evidence Summary",
        what_you_are_looking_at=(
            "A combined engineering evidence atlas summarising every section of the route "
            "across the hazards screened above."
        ),
        what_it_means=(
            "This is the single-page summary an engineer would start from when reviewing PL854."
        ),
        what_it_does_not_mean=(
            "This is a summary of evidence gathered, not an overall pass/fail or risk verdict "
            "for the pipeline."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Engineering evidence atlas",
                glob=f"{_PL854_MAPS}/pl854_engineering_evidence_atlas.png",
            ),
            VisualCandidate(
                kind="png",
                label="Engineering evidence strip",
                glob=f"{_PL854_MAPS}/pl854_engineering_evidence_strip.png",
            ),
            VisualCandidate(
                kind="html",
                label="Engineering evidence report",
                glob="processed/pl854/report/pl854_engineering_evidence_report.html",
            ),
        ),
        optional_layers=(
            VisualCandidate(
                kind="vector",
                label="Engineering evidence atlas (map)",
                layers=(
                    (
                        "processed/pl854/evidence_atlas/pl854_engineering_evidence_atlas.gpkg",
                        None,
                        "Evidence atlas",
                    ),
                ),
            ),
        ),
    ),
)

# --- Sheringham Shoal 2020 ---------------------------------------------------------------------

_SS20_MAPS = "processed/sheringham_shoal_2020/maps"

_SHERINGHAM_2020_STAGES = (
    GuidedStage(
        key="survey_bathymetry",
        label="Survey Bathymetry",
        what_you_are_looking_at=(
            "Seabed elevation from the Fugro 2020 high-resolution multibeam survey, with a "
            "data-quality check against the source."
        ),
        what_it_means=(
            "This is the measured seabed shape the terrain and change analyses below are "
            "built from."
        ),
        what_it_does_not_mean=(
            "This is the surveyed surface at survey time only -- it is not a prediction of "
            "future change."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Bathymetry QA",
                glob=f"{_SS20_MAPS}/sheringham_shoal_2020_bathymetry_qa.png",
            ),
        ),
    ),
    GuidedStage(
        key="terrain",
        label="Terrain",
        what_you_are_looking_at=(
            "High-resolution seabed terrain, including sand waves, banks and troughs."
        ),
        what_it_means=(
            "Shows the shape of the seabed, including sand waves, banks and troughs; slope, "
            "ruggedness and relief are derived from it."
        ),
        what_it_does_not_mean=(
            "It is not a prediction of future change and does not show seabed change on its own."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Terrain atlas",
                glob=f"{_SS20_MAPS}/sheringham_shoal_2020_terrain_atlas.png",
            ),
        ),
        capability_id="terrain",
        optional_layers=(
            VisualCandidate(
                kind="raster",
                label="Canonical bed elevation (raw raster preview)",
                raster_glob="processed/sheringham_shoal_2020/terrain/canonical_bed_elevation.tif",
            ),
        ),
    ),
    GuidedStage(
        key="bedforms",
        label="Bedforms",
        what_you_are_looking_at=(
            "Sand-wave and bedform geometry measured from the terrain, matched across survey "
            "epochs where possible."
        ),
        what_it_means=(
            "These are real, measured bedform shapes and their observed crest displacement "
            "between 2018 and 2020."
        ),
        what_it_does_not_mean=(
            "This is an observed geometry comparison, not a sediment transport-rate model."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Bedform morphometry",
                glob=f"{_SS20_MAPS}/sheringham_shoal_2020_bedform_morphometry.png",
            ),
        ),
        capability_id="bedforms",
        optional_layers=(
            VisualCandidate(
                kind="png",
                label="Bedform diagnostics (non-canonical, secondary)",
                glob=f"{_SS20_MAPS}/sheringham_shoal_2020_bedform_diagnostics_noncanonical.png",
            ),
        ),
    ),
    GuidedStage(
        key="multi_epoch_change",
        label="Multi-epoch Change",
        what_you_are_looking_at="Seabed elevation difference between the 2018 and 2020 surveys.",
        what_it_means=(
            "Shows where the seabed measurably rose or fell between the two survey dates."
        ),
        what_it_does_not_mean=(
            "This is a two-epoch comparison, not a continuous or predictive erosion model."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Seabed change (2018-2020)",
                glob=f"{_SS20_MAPS}/sheringham_shoal_2018_2020_seabed_change.png",
            ),
        ),
        capability_id="erosion_deposition",
        optional_layers=(
            VisualCandidate(
                kind="png",
                label="Change QA (secondary)",
                glob=f"{_SS20_MAPS}/sheringham_shoal_2018_2020_change_qa.png",
            ),
            VisualCandidate(
                kind="raster",
                label="Delta bed elevation (raw raster preview)",
                raster_glob=(
                    "processed/sheringham_shoal_2020/change/"
                    "delta_bed_elevation_2020_minus_2018_m.tif"
                ),
            ),
        ),
    ),
    GuidedStage(
        key="slope_screening",
        label="Slope Screening",
        what_you_are_looking_at=(
            "Seabed slope screened against a strength-demand check at two analysis windows."
        ),
        what_it_means="This flags where slope and demand together warrant a closer look.",
        what_it_does_not_mean=(
            "This is a screening indicator, not a landslide probability or a geotechnical "
            "stability verdict."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Slope instability screening",
                glob=f"{_SS20_MAPS}/sheringham_shoal_2020_slope_instability_screening.png",
            ),
        ),
        capability_id="slope_instability",
        optional_layers=(
            VisualCandidate(
                kind="raster",
                label="Slope, 10 m window (raw raster preview)",
                raster_glob="processed/sheringham_shoal_2020/slope_stability/slope_10m_deg.tif",
            ),
        ),
    ),
)

# --- Sheringham Shoal 2008 CPTU ------------------------------------------------------------------

_CPTU_ROOT = "processed/sheringham_shoal_2008_cptu"

_CPTU_STAGES = (
    GuidedStage(
        key="investigation_area",
        label="Investigation Area",
        what_you_are_looking_at=(
            "The 100 CPT/CPTU test locations from the 2008 GEO geotechnical investigation at "
            "Sheringham Shoal."
        ),
        what_it_means=(
            "This is the real, measured extent of the geotechnical investigation used as "
            "evidence in the stages that follow."
        ),
        what_it_does_not_mean=(
            "This is the investigation's point coverage, not a declared survey boundary -- no "
            "separate boundary file exists locally."
        ),
        visuals=(
            VisualCandidate(
                kind="vector",
                label="Investigation extent",
                layers=((f"{_CPTU_ROOT}/cpt_locations.gpkg", "cpt_locations", "CPT locations"),),
            ),
        ),
    ),
    GuidedStage(
        key="cpt_locations",
        label="CPT Locations",
        what_you_are_looking_at=(
            "Every individual CPT/CPTU test location, with its identifier and canonical position."
        ),
        what_it_means=(
            "Pick any location (or its test ID) to inspect its own measured depth profile."
        ),
        what_it_does_not_mean=(
            "Position here is the canonical surveyed position, not the source's raw declared "
            "coordinate."
        ),
        visuals=(
            VisualCandidate(
                kind="vector",
                label="CPT locations",
                layers=((f"{_CPTU_ROOT}/cpt_locations.gpkg", "cpt_locations", "CPT locations"),),
            ),
        ),
    ),
    GuidedStage(
        key="cpt_evidence",
        label="CPT Evidence",
        what_you_are_looking_at=(
            "Readiness of the CPT/CPTU evidence itself, axis by axis (source package, digital "
            "profile, identity, depth reference, units, spatial reference)."
        ),
        what_it_means=(
            "This is how complete and trustworthy the underlying measured evidence is, as "
            "assessed by the engine."
        ),
        what_it_does_not_mean=(
            "This is evidence readiness, not a liquefaction susceptibility, hazard or risk "
            "statement."
        ),
        visuals=(
            VisualCandidate(
                kind="readiness",
                label="CPT evidence readiness",
                glob=f"{_CPTU_ROOT}/cpt_readiness.json",
            ),
        ),
    ),
    GuidedStage(
        key="profile_availability",
        label="Profile Availability",
        what_you_are_looking_at=(
            "The measured depth profile for one selected CPT test: cone resistance (qc), "
            "sleeve friction (fs) and pore pressure (u2)."
        ),
        what_it_means=(
            "These are the source's own measured channels, plotted against depth below seabed."
        ),
        what_it_does_not_mean=(
            "Corrected cone resistance (qt) is not available for this source, and no soil "
            "classification or liquefaction calculation is performed here."
        ),
        visuals=(VisualCandidate(kind="chart", label="CPT depth profile", chart_id="cpt_profile"),),
    ),
    GuidedStage(
        key="liquefaction_foundation",
        label="Liquefaction Foundation",
        what_you_are_looking_at=(
            "What the CPT evidence foundation for liquefaction assessment currently covers, "
            "and what it deliberately does not."
        ),
        what_it_means=(
            "The evidence foundation (real CPT data, canonical profile, identity and CRS "
            "integrity) is complete."
        ),
        what_it_does_not_mean=(
            "No CRR, CSR, factor of safety, probability, LPI, settlement or lateral-spreading "
            "result exists -- triggering physics is not implemented."
        ),
        visuals=(
            VisualCandidate(
                kind="readiness",
                label="Liquefaction readiness",
                glob=f"{_CPTU_ROOT}/liquefaction_readiness.json",
            ),
        ),
        capability_id="liquefaction",
    ),
)

# --- Barrow 2016 -------------------------------------------------------------------------------

_BARROW_MAPS = "processed/barrow_2016/maps"

_BARROW_STAGES = (
    GuidedStage(
        key="route_asset",
        label="Route / Asset",
        what_you_are_looking_at=(
            "The Barrow export-cable route used as the reference asset for burial/exposure "
            "evidence."
        ),
        what_it_means=(
            "This is the asset every burial measurement below is located against, by chainage."
        ),
        what_it_does_not_mean=(
            "This is the reference geometry only -- it carries no burial or exposure "
            "information by itself."
        ),
        visuals=(
            VisualCandidate(
                kind="vector",
                label="Asset route",
                layers=(
                    (
                        "processed/barrow_2016/burial/canonical_asset_route.gpkg",
                        "asset_route",
                        "Asset route",
                    ),
                ),
            ),
        ),
    ),
    GuidedStage(
        key="burial_profile",
        label="Burial Profile",
        what_you_are_looking_at=(
            "The source's measured burial/exposure value along the cable, by chainage."
        ),
        what_it_means=(
            "This is the 2016 depth-of-burial survey's own measured value at each recorded point."
        ),
        what_it_does_not_mean=(
            "The source does not state a vertical reference point or sign convention for this "
            "value -- it is shown exactly as measured, never relabelled as a burial depth."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Burial KP profile",
                glob=f"{_BARROW_MAPS}/barrow_2016_burial_kp_profile.png",
            ),
        ),
        capability_id="burial_exposure",
        optional_layers=(
            VisualCandidate(
                kind="chart", label="Burial profile (chainage detail)", chart_id="burial_profile"
            ),
        ),
    ),
    GuidedStage(
        key="observed_burial_state",
        label="Observed Burial / Exposure State",
        what_you_are_looking_at=(
            "Where the source itself flags a point as measured versus source-interpreted exposure."
        ),
        what_it_means=(
            "These are two distinct evidence roles the source keeps separate: a direct "
            "measurement, and the source's own interpretation."
        ),
        what_it_does_not_mean=(
            "This is not a current condition assessment -- it reflects the 2016 survey only."
        ),
        visuals=(
            VisualCandidate(
                kind="png",
                label="Observed burial state",
                glob=f"{_BARROW_MAPS}/barrow_2016_observed_burial_state.png",
            ),
        ),
        capability_id="burial_exposure",
        optional_layers=(
            VisualCandidate(
                kind="vector",
                label="Measured burial points",
                layers=(
                    (
                        "processed/barrow_2016/burial/burial_exposure_poc.gpkg",
                        "burial_measurements",
                        "Measured burial points",
                    ),
                ),
            ),
            VisualCandidate(
                kind="vector",
                label="Source-interpreted exposure",
                layers=(
                    (
                        "processed/barrow_2016/burial/burial_exposure_poc.gpkg",
                        "source_interpreted_exposure",
                        "Source-interpreted exposure",
                    ),
                ),
            ),
        ),
    ),
)

STORIES: tuple[ProjectStory, ...] = (
    ProjectStory(project_id="pl854", stages=_PL854_STAGES),
    ProjectStory(project_id="sheringham_shoal_2020", stages=_SHERINGHAM_2020_STAGES),
    ProjectStory(project_id="sheringham_shoal_2008_cptu", stages=_CPTU_STAGES),
    ProjectStory(project_id="barrow_2016", stages=_BARROW_STAGES),
)

_STORIES_BY_PROJECT = {story.project_id: story for story in STORIES}


def story_for_project(project_id: str) -> ProjectStory:
    try:
        return _STORIES_BY_PROJECT[project_id]
    except KeyError:
        raise KeyError(project_id) from None
