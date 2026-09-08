# Marine Module POC Architecture

Status: **partially implemented**. MAR-026 (see "MAR-026 update" notes throughout this
document, and the "Current implementation boundary" section at the end) added a first real,
generic, local-file operator-project registration and readiness layer
(`src/marine_engine.project`, CLI command `build-project-readiness`). It implements a
meaningful first slice of Sections 2-4 below for local files only -- it is explicitly NOT a
complete implementation of every stage described in this document, and several stages (5-9)
remain concept-only. See the "Current implementation boundary" section for the precise
line between what exists in code today and what is still future work.

The PL854 work completed so far (MAR-002 through MAR-018) demonstrates every
stage of this architecture end-to-end for exactly one fixed, public-data
project. Generalizing it to arbitrary operator-supplied projects was the
next milestone; MAR-026 is the first real step of that generalization,
described here at the logical/conceptual level for the parts still ahead.

## 1. Project

A project is the top-level unit of work: one pipeline route (or corridor),
one working CRS, one AOI, one set of study parameters. PL854 is the first
(and currently only) concrete project instance. A generic implementation
would let an operator define a new project without engineering involvement
in the core analytics code.

## 2. Data Ingestion

Operator-supplied files (MBES, sidescan sonar, sub-bottom/HR seismic
products, CPT, boreholes, grab samples, metocean, pipeline route/design
parameters, existing infrastructure) enter the system here. This stage is
concerned only with accepting files in the formats operators actually use
and recording their provenance (source, retrieval date, licence) -- it does
not interpret or validate their content.

**MAR-026 update:** implemented for **local files only** -- an operator
provides local project files plus a YAML manifest
(`marine_engine.project.manifest.ProjectManifest`); `build-project-readiness`
registers each declared asset (content-based SHA-256 identity, byte size,
resolved path, never mutating the source file) and records its provenance
exactly as declared, kept structurally separate from facts observed in the
file itself. Initial supported categories with a real ingestion path:
`PIPELINE_ROUTE` (GeoPackage/GeoJSON), `BATHYMETRY_RASTER` (GeoTIFF),
`BURIAL_PROFILE` (CSV/Parquet, explicit column mapping). Every other
category in the forward-compatible vocabulary (`METOCEAN`, `CPT`,
`BOREHOLE`, `SHALLOW_GAS_INTERPRETATION`, ...) can be *registered*
(identity + provenance recorded) but has no ingestion/interpretation logic
yet -- see "Current implementation boundary" below. No web UI, API server,
cloud upload, object storage, database, or authentication exists -- this is
a local, offline CLI layer only.

## 3. Data QA / Readiness

Before any analysis, ingested data must pass integrity and readiness
checks: CRS and vertical datum present and consistent, coordinate scale
plausible, nodata/sentinel values handled, survey epoch recorded, spatial
coverage relative to the route understood. This mirrors the data-integrity
discipline already used throughout the PL854 work (e.g. MAR-005/006/016),
generalized to arbitrary operator-supplied files rather than one fixed set
of public sources.

**MAR-026 update:** implemented for the three categories above.
`PIPELINE_ROUTE` gets a new generic readiness check set
(`marine_engine.project.route_adapter`: CRS presence, coordinate
plausibility, topological continuity -- a disconnected route is reported
`NOT_READY` rather than having connectivity invented, and canonical route
direction is always `SOURCE_GEOMETRY_ORDER`, never a guessed platform/
landfall endpoint). `BATHYMETRY_RASTER` and `BURIAL_PROFILE` reuse the
existing accepted `marine_engine.terrain.readiness.assess_bathymetry_readiness`
and `marine_engine.burial.readiness.assess_burial_profile_readiness`
unmodified, proven by real registration of the accepted Sheringham Shoal
2020 bathymetry and Barrow 2016 burial profile through this generic layer.
Status vocabulary (`READY`/`READY_WITH_LIMITATIONS`/`NOT_READY`,
`BLOCKING`/`LIMITATION`) is unchanged from the existing convention -- no
readiness score, percentage, or confidence index exists anywhere in this
layer, and there is no aggregate "project is ready for marine hazard
analysis" claim (see `project.registry.PROJECT_HAZARD_READINESS_DISCLAIMER`).

## 4. Canonical Project Model

A standardized, project-agnostic internal representation that every
downstream geohazard engine consumes: canonical route/chainage geometry,
a common working CRS, a shared section/support-grid concept, and typed
slots for each data category (measured / interpreted / derived, see below).
PL854's own canonical pipeline/AOI/chainage model (MAR-002/003/004) is the
existing single-project prototype of this idea.

**MAR-026 update:** a first project-agnostic slice exists --
`marine_engine.project.registry` builds a canonical per-asset inventory
(`project_asset_registry.parquet`) and, where a route asset resolves safely,
a canonical project route in the working CRS (`project_route.gpkg`). Every
registered asset carries an explicit, manifest-declared, never-inferred
evidence role -- `PROJECT_GEOMETRY` / `MEASURED` / `SOURCE_INTERPRETED` /
`DERIVED` (`marine_engine.project.categories`) -- which is exactly the axis
Sections 5-7 below describe; MAR-026 makes it a structural field rather
than only a narrative distinction. What is still missing from a full
canonical project model: cross-asset correlation (e.g. burial-profile
coverage relative to a linked route), a shared section/support-grid
concept, and typed ingestion for any category beyond the three implemented
so far.

## 5. Measured Data

Direct physical observations, supplied by the operator or their surveyors.
Examples: MBES bathymetry, CPT, boreholes, grab samples, current
observations, sidescan sonar, sub-bottom/HR seismic. The canonical project
model records what was measured, when, and by whom -- it does not alter or
reinterpret the measurement itself.

**MAR-026 update:** this is the `MEASURED` evidence role. MBES bathymetry
and burial-survey profiles are registered under it today with real
readiness adapters (Section 3); CPT, boreholes, grab samples, current
observations, sidescan sonar, and sub-bottom/HR seismic can be *registered*
under `MEASURED` but have no ingestion/readiness adapter yet.

## 6. Interpreted Data

Expert or contractor interpretation layered on top of measured data.
Examples: shallow gas polygons, faults, buried channels, boulders, seabed
feature interpretation. This category is explicitly allowed to remain
operator- or consultant-supplied; the Marine Module does not require every
interpretation to be reproduced in software.

**MAR-026 update:** this is the `SOURCE_INTERPRETED` evidence role
(renamed from "Interpreted Data" only for the manifest vocabulary, not for
this document's meaning). `SOURCE_INTERPRETED` data can never be
automatically re-labelled `MEASURED` -- the two are orthogonal, manifest-
declared fields, mirroring the MAR-025A measured-geometry-vs-source-
interpretation principle generalized project-wide. Shallow gas polygons,
faults, buried channels, and boulder catalogues can be registered under
this role today (identity + provenance only, `REGISTERED_READINESS_NOT_IMPLEMENTED`
-- see Section 2); no interpretation logic exists for them yet.

## 7. Derived Geohazard Layers

Software-computed analytics built from measured and/or interpreted data via
domain-specific geohazard engines. Examples already demonstrated for PL854:
regional terrain morphology (MAR-007), combined wave-current bed shear
(MAR-012), noncohesive sediment mobility capacity (MAR-013), scour-onset
screening (MAR-014). Examples not yet demonstrated: free-span
susceptibility, slope-instability screening, and a generalized route-
constraints layer. Each engine remains scientifically distinct -- this
architecture does not fuse engines into a single score.

**MAR-026 update:** this is the `DERIVED` evidence role. `DERIVED` output
can never be automatically re-labelled source evidence. MAR-026 itself
introduces no new derived geohazard layer -- see MAR-020 through MAR-025A
for the independent, unchanged engines already implemented (terrain,
seabed change, bedforms, scour, burial/exposure, free-span/support-loss).

## 8. Map View

Spatial evidence/hazard layers rendered together against the route and a
shared geographic extent. The MAR-018/019 engineering evidence atlas is the
first static demonstrator of this view for one fixed project.

## 9. KP / Route View

Aligned along-route engineering evidence, chainage/KP-referenced. The
MAR-018/019 evidence strip is the first static demonstrator of this view.

## 10. Reporting / GIS Export

Human-readable engineering reports (with provenance and limitations always
attached) and GIS export (GeoPackage layers openable in QGIS/ArcGIS). The
MAR-018 engineering evidence report and atlas GeoPackage are the first
static demonstrators of this stage.

**MAR-026 update:** unchanged -- Sections 8-10 remain concept-only. MAR-026
produces one project-scoped HTML report and one GeoPackage per project
(`project_readiness_report.html`, `project_route.gpkg`), but does not
implement a shared map view, a KP/route view, or a multi-project GIS export
convention; those remain future work.

---

## Current implementation boundary (as of MAR-026)

MAR-026 (`src/marine_engine/project/`, CLI command `build-project-readiness`)
implemented the first real slice of this architecture:

**Implemented:**
- A typed, validated operator project manifest (`project.manifest.ProjectManifest`,
  Pydantic, unknown keys rejected, duplicate asset IDs rejected, asset paths
  resolved relative to the manifest file).
- Content-based (SHA-256), never-mutated source-file registration
  (`project.identity`).
- The four-way orthogonal evidence-role vocabulary (`project.categories`),
  enforced structurally, never inferred from filename text.
- Real readiness adapters, each delegating to an existing accepted module
  rather than reimplementing its checks: `PIPELINE_ROUTE` (new generic
  route-readiness logic, `project.route_adapter`), `BATHYMETRY_RASTER`
  (reuses `terrain.readiness.assess_bathymetry_readiness`,
  `project.bathymetry_adapter`), `BURIAL_PROFILE` (reuses
  `burial.readiness.assess_burial_profile_readiness`,
  `project.burial_adapter`).
- Declared-vs-observed separation for every asset, with explicit conflict
  recording (never a silent reprojection or silently-picked value).
- A deterministic per-project output package (normalized manifest, asset
  registry parquet, structured readiness JSON, canonical route GeoPackage
  when applicable, HTML report) with no numeric readiness score and no
  aggregate "ready for marine hazard analysis" claim anywhere.
- Real demonstrations: the accepted Sheringham Shoal 2020 MAR-020
  bathymetry and the accepted Barrow 2016 MAR-024/MAR-024A burial profile,
  each registered through this generic layer with zero change to their
  real, already-accepted scientific conclusions.

**Explicitly NOT implemented by MAR-026 (future work):**
- A web UI, API server, cloud upload, object storage, database, or
  authentication -- this is a local-file, offline CLI layer only.
- Ingestion/readiness adapters for any category beyond `PIPELINE_ROUTE`/
  `BATHYMETRY_RASTER`/`BURIAL_PROFILE` (e.g. `METOCEAN`, `CPT`, `BOREHOLE`,
  `SHALLOW_GAS_INTERPRETATION`, ...) -- these register (identity +
  provenance only) but are explicitly `REGISTERED_READINESS_NOT_IMPLEMENTED`.
- Cross-asset correlation (e.g. computing a burial profile's real coverage
  fraction against a linked route asset).
- Any new geohazard physics, structural assessment, or risk/hazard score --
  MAR-026 is an integration/foundation ticket, not a new scientific engine.
- A shared map view, KP/route view, or multi-project GIS export convention
  (Sections 8-10 above).

MAR-026 does **not** mean all operator file formats are supported, and it
does **not** claim any registered project is ready for scour, free-span,
liquefaction, shallow gas, or any other specific marine geohazard analysis
-- only that its registered assets' structural registration and (for the
three implemented categories) data readiness have been assessed.
