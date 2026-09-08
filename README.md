# marine-engine

Research PoC for estimating how seabed morphodynamics affect a subsea
pipeline over its design life: erosion, deposition, sediment mobility,
burial/exposure, scour susceptibility, free-span susceptibility, and
lifetime risk scenarios.

**First study case:** PL854, Anglia A -> LOGGS pipeline corridor, Southern
North Sea (see [`configs/pl854.yaml`](configs/pl854.yaml)).

This is a personal research project, not a production engineering tool.

## Architecture

The scientific engine is a plain, importable Python package
(`marine_engine`), not notebooks or one-off scripts, so it can later be
wrapped by an API and consumed by a GIS/web application. Each stage of the
target workflow is its own subpackage:

```
open-data providers -> preprocessing -> seabed/morphology features
    -> metocean features -> sediment mobility -> erosion/deposition
    -> pipeline interaction -> lifetime risk scenarios -> validation
    -> export (GeoTIFF / GeoPackage / Parquet / JSON)
```

| Package        | Responsibility                                                        |
|-----------------|------------------------------------------------------------------------|
| `providers`     | Open-data provider clients (bathymetry, metocean, sediment, route, ...)|
| `preprocessing` | Cleaning, reprojection, resampling, harmonisation of raw provider data  |
| `morphology`    | Seabed morphology features (slope, roughness, bedforms, mobility)      |
| `sediment`      | Sediment mobility modelling (grain size, shear stress, transport)      |
| `metocean`      | Metocean features (waves, currents, tides) driving seabed mobility     |
| `pipeline`      | Pipeline interaction (burial/exposure, free-span, scour susceptibility)|
| `risk`          | Lifetime risk scenarios combining the above                            |
| `validation`    | Validation of model outputs against observed/survey data               |
| `export`        | Export of results to GeoTIFF, GeoPackage, Parquet, and JSON            |

`config.py` defines the schema (via pydantic) and loader for study-specific
YAML configuration files under `configs/`. `cli.py` is a thin argparse
entry point over that config system.

Beyond the NSTA provider (MAR-002), AOI/chainage preprocessing
(MAR-003/004), bathymetry source discovery (MAR-005), the canonical
EMODnet baseline DTM (MAR-006), CDI source-survey resolution (MAR-006B),
broad regional seabed morphology (MAR-007), the PL854 sediment evidence
base (MAR-008), and the PL854 metocean forcing evidence base (MAR-009),
the stage packages contain no algorithms yet — see "Status" below.

## Project layout

```
configs/            Study-specific YAML configs (e.g. pl854.yaml)
data/raw/            Unmodified downloaded datasets (gitignored, not committed)
data/interim/        Intermediate/derived data (gitignored, not committed)
data/processed/      Analysis-ready outputs (gitignored, not committed)
src/marine_engine/   The engine package (see table above)
tests/               pytest suite
```

## Getting started

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                # create .venv and install dependencies
uv run pytest                          # run the test suite
uv run ruff check .                    # lint
uv run marine-engine version
uv run marine-engine validate-config configs/pl854.yaml
uv run marine-engine ingest-pipeline configs/pl854.yaml
uv run marine-engine build-aoi configs/pl854.yaml
uv run marine-engine build-chainage configs/pl854.yaml
uv run marine-engine discover-bathymetry configs/pl854.yaml
uv run marine-engine fetch-bathymetry configs/pl854.yaml
uv run marine-engine build-bathymetry configs/pl854.yaml
uv run marine-engine resolve-bathymetry-sources configs/pl854.yaml
uv run marine-engine build-regional-morphology configs/pl854.yaml
uv run marine-engine build-sediment-evidence configs/pl854.yaml
uv run marine-engine build-metocean-evidence configs/pl854.yaml
uv run marine-engine build-engineering-evidence-atlas configs/pl854.yaml
uv run marine-engine build-marine-poc-review-package configs/pl854.yaml
uv run marine-engine build-highres-terrain-poc configs/sheringham_shoal_2020.yaml
uv run marine-engine build-seabed-change-poc configs/sheringham_shoal_2020.yaml
uv run marine-engine build-bedform-morphodynamics-poc configs/sheringham_shoal_2020.yaml
```

`ingest-pipeline`, `discover-bathymetry`, `fetch-bathymetry`,
`build-bathymetry`, `resolve-bathymetry-sources`, `build-regional-morphology`,
`build-sediment-evidence`, and `build-metocean-evidence` require network
access to public services (NSTA, MEDIN, BGS GeoNetwork/ArcGIS REST,
EMODnet, SeaDataNet CDI, Copernicus Marine). Their live-source smoke tests
are excluded from the default test run; opt in with `uv run pytest -m live`.

`build-metocean-evidence` additionally requires a (free) Copernicus Marine
account: catalogue metadata is public, but real data acquisition needs
`uv run copernicusmarine login` (or the `COPERNICUSMARINE_SERVICE_USERNAME`
/ `COPERNICUSMARINE_SERVICE_PASSWORD` environment variables) run once,
outside of any AI assistant, before this command's real acquisition steps
can proceed -- it stops with a clear `CopernicusAuthenticationRequiredError`
otherwise, never an interactive credential prompt.

## Status

- `MAR-001`: project scaffold — structure, config system, CLI, and test
  infrastructure. No scientific algorithms, no dataset downloads, no ML, no
  web application.
- `MAR-002`: first real provider (`providers/nsta.py`) ingests the PL854
  pipeline geometry from the authoritative NSTA UKCS offshore infrastructure
  dataset and normalizes it to `data/processed/pl854/pipeline.gpkg`.
- `MAR-003`: AOI preprocessing (`preprocessing/aoi.py`) buffers the canonical
  pipeline by the configured `area_of_interest.corridor_buffer_m` (5000 m)
  into the study's spatial extent, `data/processed/pl854/aoi.gpkg`.
- `MAR-004`: chainage/KP preprocessing (`preprocessing/chainage.py`)
  generates the linear-reference point system at the configured
  `pipeline.chainage_interval_m` (25 m) plus the exact route terminus, into
  `data/processed/pl854/chainage_25m.gpkg`. Chainage direction is recorded
  honestly as `source_geometry_start` — semantic endpoint identity
  (Anglia A vs LOGGS) remains unresolved by design.
- `MAR-005`: bathymetry source discovery (`providers/bathymetry/`) queries
  the approved UKHO (via MEDIN), BGS, and EMODnet sources, spatially
  verifies each against the real canonical pipeline/AOI/chainage, ranks
  candidates, and acquires the mandatory EMODnet 2024 baseline into
  `data/interim/pl854/bathymetry_inventory.{parquet,gpkg}` and
  `data/raw/bathymetry/`. As of this run, no automatically-downloadable
  dataset intersects the pipeline itself (the historical BGS surveys are
  metadata/bbox-only and restricted-access) — the primary analysis
  candidate is honestly reported as unresolved rather than defaulting to
  EMODnet.
- `MAR-006`: canonical EMODnet baseline DTM (`preprocessing/bathymetry.py`)
  transforms the raw EMODnet 2024 raster into a reproducible baseline:
  empirically-verified sign convention (observed `negative_elevation` for
  this raster, converted to positive-down `depth_lat_m`), reprojected to
  `EPSG:32631` at a 100 m analysis grid using bilinear resampling, clipped
  to the real AOI polygon (not just its bounding box), and written to
  `data/processed/pl854/bathymetry/emodnet_baseline_lat_100m.tif` with a
  JSON provenance sidecar. Source-reference and quality-index attribution
  (`providers/bathymetry/emodnet.py`'s WFS functions, queried server-side
  via `CQL_FILTER`) is sampled onto all 941 chainage stations into
  `data/processed/pl854/bathymetry/chainage_bathymetry.parquet`. Depth
  processing and source/quality attribution are kept deliberately
  separable — a WFS attribution outage is recorded as
  `source_attribution_status = unavailable` but never fails the DTM build.
  An official Mean Sea Level product was confirmed reproducibly acquirable
  (tile D4, 2024 release, via `emodnet:download_tiles`) but is not
  downloaded or merged with the LAT baseline in this ticket. Still no
  morphology (slope/curvature/roughness/BPI), erosion/deposition,
  scour/free-span, or risk calculations — those are later tickets.

  **Scientific limitations of this baseline** (apply to every downstream
  use of `emodnet_baseline_lat_100m.tif` / `chainage_bathymetry.parquet`
  until a higher-resolution survey supersedes it):
  1. EMODnet 2024 is approximately 115 m-class regional bathymetry.
  2. The 100 m projected grid is an analysis grid, not true 100 m
     measurement resolution.
  3. Appropriate for regional seabed context and broad morphology only.
  4. NOT sufficient by itself for pipeline-scale local scour, metre-scale
     sand-wave geometry, or free-span detection.
  5. High-resolution MBES can replace/augment this baseline later without
     changing the canonical pipeline/chainage architecture.

  Still no sediment/metocean providers, erosion/deposition or risk science,
  no ML, no web app — those arrive in later tickets.
- `MAR-006B`: PL854 EMODnet CDI source-survey resolution
  (`providers/bathymetry/cdi.py`, `preprocessing/source_resolution.py`)
  follows each of the three real 2024-release `source_references` ids that
  cross the pipeline (`110153`, `121953`, `121954`) through their official
  SeaDataNet CDI `metadata_url` to real survey provenance: `110153` is a
  1992 single-beam-echosounder survey ("Haddock Bank", UKHO cruise HI560,
  covering the first ~1% of the route); `121953`/`121954` are 1991 surveys
  (UKHO cruise HI524-HI525-HI531, instrument unstated) covering the
  remaining ~99%. All three are 32-33 years old at the 2024 release and
  require registration plus owner negotiation (via OceanWise/UKHO) to
  request the original data -- none is directly downloadable, and **none of
  the three states a numeric spatial resolution**, so whether the original
  source data is actually finer than the ~115 m EMODnet composite remains
  unverified (see `MAR-006C` below) -- they are requestable/negotiable, not
  confirmed higher-resolution products. Output:
  `data/interim/pl854/emodnet_cdi_sources.parquet`. The CDI report host
  fronts every request with a client-side proof-of-work challenge that a
  plain HTTP client cannot pass, so live automated resolution falls back to
  a manually browser-verified snapshot for these three known ids and
  reports that honestly rather than either fabricating data or crashing.

  **`EMODnet DTM 2024` is a product release/version, not a seabed survey
  date -- do not read "2024" as when this bathymetry was measured.** This
  ticket's audit fix makes that explicit in the schema itself:
  `SurveyRecord` now has a separate `product_release_year` field: the
  EMODnet composite sets `product_release_year=2024` and leaves
  `acquisition_year=None`, while each CDI-resolved underlying survey
  carries its own real `acquisition_year` (1991/1992 here) and
  `survey_age_at_product_release_year`.

  Still no morphology, sediment/metocean, erosion/deposition,
  scour/free-span, or risk science, no ML, no web app, no LAT/MSL
  conversion, and no SeaDataNet data request was submitted.
- `MAR-006C`: provenance-semantics correction. MAR-006B's
  `classify_recovery_potential()` had conflated "a request path exists"
  with "higher resolution is confirmed" -- a source requestable via owner
  negotiation was reported as `HIGH_RES_SOURCE_REQUESTABLE` even though
  none of PL854's three CDI records state a numeric resolution (QI
  instrument class, e.g. QI_Vertical=4 suggesting MBES, is not proof of
  exported/grid resolution on its own). Fixed: `recovery_potential` now
  requires CDI itself to state a real numeric resolution finer than
  EMODnet's ~115 m baseline before returning a `HIGH_RES_*` value; for the
  current three PL854 records it correctly reports
  `SOURCE_RESOLUTION_UNKNOWN`, while `access_class`
  (`OWNER_PERMISSION_REQUIRED` for all three) still separately and
  accurately conveys that a real request path exists. Also moved this
  ticket's own output from `data/processed/pl854/bathymetry/` to
  `data/interim/pl854/` -- it is provenance-resolution metadata, not an
  analysis-ready product; the canonical DTM and chainage-bathymetry outputs
  are unaffected.
- `MAR-006D`: a further recovery-potential fix. `SOURCE_RESOLUTION_UNKNOWN`
  vs `HIGH_RES_SOURCE_*` had been checking both `horizontal_resolution_note`
  and `vertical_resolution_note`, but vertical resolution/accuracy (how
  precisely depth is measured) does not establish horizontal spatial/grid
  resolution (how densely the seabed is sampled) -- a sub-metre vertical
  figure could have incorrectly triggered a `HIGH_RES_SOURCE_*` result with
  no horizontal resolution stated at all. Fixed to consider only
  `horizontal_resolution_note`; `vertical_resolution_note` remains
  preserved as metadata everywhere else. PL854's three records are
  unaffected (`SOURCE_RESOLUTION_UNKNOWN`, as before).
- `MAR-007`: broad (500/1000/2000 m-scale) regional seabed morphology
  context (`morphology/regional.py`) from the EMODnet baseline --
  local-plane-fit slope, Topographic Position Index, local relief, and
  terrain variability, computed on a 2.2 km analysis halo beyond the
  canonical AOI (to avoid edge bias) and then clipped back to it. Outputs:
  `data/processed/pl854/morphology/{slope_500m_deg,slope_1000m_deg,
  tpi_1000m_m,tpi_2000m_m,local_relief_1000m_m,local_relief_2000m_m,
  terrain_std_1000m_m,terrain_std_2000m_m}.tif`,
  `chainage_regional_morphology.parquet` (941 stations), and
  `morphology_metadata.json`. No curvature, rugosity, TRI, or aspect; no
  sand-wave crest/trough/wavelength detection; no scour, free-span, or risk
  scoring -- none of that is supported by this baseline (see below).

  **These are broad regional morphology derivatives, not present-day local
  pipeline survey products.** The source bathymetry underlying 100% of
  PL854 is from CDI surveys acquired in 1991 (~99% of the route) and 1992
  (~1%) -- confirmed from the actual joined MAR-006B/C provenance, not
  assumed. EMODnet 2024 is the DTM *product release* year, never the
  acquisition year. These broad, kilometre-scale features are appropriate
  for regional gradient and bank/flank/channel-scale context only; they
  must NOT be read as current sand-wave crests/troughs, local pipeline
  scour, metre-scale roughness, embedment, or free-span condition -- no
  such mapping has been performed. EMODnet's official per-cell QA
  attributes (min/max/std depth, sample count, interpolation flag) were
  checked live and found unavailable as a small/queryable coverage for
  this release (only a whole-tile, non-AOI-clipped "SD" archive exists);
  the corresponding chainage fields are recorded as null rather than
  fabricated.
- `MAR-008`: PL854 seabed sediment/substrate evidence base
  (`providers/sediment/bgs.py`, `sediment/{evidence,grain_size}.py`) from
  three separate, never-blended BGS evidence tiers, each queried spatially
  against the real AOI polygon (not a bounding box or text match):

  1. **Observed** -- BGS "Offshore samples: particle size analysis": 27
     real point samples intersect the PL854 AOI (all confirmed
     `SURFACE_GRAB`; sample years 1979-2009). Distance to the pipeline
     ranges 141-4986 m (median ~2928 m) -- proximity is always reported
     alongside the observation, never treated as proof the sample
     represents seabed conditions at the pipe. `surface_evidence_class`
     (e.g. `SURFACE_GRAB`) is a vertical-position/sampling-relationship
     classification at collection time only -- it does NOT establish that
     an observation represents present-day seabed conditions merely
     because it was taken at the seabed surface; a 1979 grab is still
     `SURFACE_GRAB`, with its real age carried separately and explicitly in
     `sample_date`/`sample_year`/`sample_age_years_at_run` (`MAR-008A`).
  2. **Mapped** -- BGS Seabed Sediments 250k (1:250,000 regional geological
     mapping, never site-specific ground truth): 8 polygons intersect the
     AOI, covering all 941 chainage stations (`mapped_250k_*` fields).
  3. **Predictive** -- BGS Predictive Seabed Sediments UK (Distributional
     Random Forest, ~38,000 training observations, covariates including
     bathymetry/morphometry/currents/tides): `evidence_role =
     SECONDARY_MODEL_COMPARISON` always, with an explicit
     `circularity_warning` on every predictive-field record -- never
     treated as ground truth, never blended with the observed or mapped
     tiers, never used to fill a missing observed value.

  Grain-size percentiles (D10/D50/D90) are derived only from a PSA
  record's own internally consistent, non-overlapping phi bins (never from
  Folk class, GSM percentages, or the predictive product); of the 27
  observed samples, 5 have a valid whole-sample D10/D50/D90 (D50 range
  0.21-0.38 mm, pure/near-pure sand), 6 are `AMBIGUOUS_BIN_SCHEME`, and 16
  are `INSUFFICIENT_BINS` -- most of those because their phi-bin breakdown
  covers only the sand fraction while gravel is materially present (a real
  correctness guard, not conservatism for its own sake: computing a
  whole-sample D50 from a partial-fraction breakdown would misrepresent
  it). No sediment mobility, Shields parameter, critical shear stress,
  bedload/suspended transport, erosion/deposition, or cohesive/noncohesive
  classification is computed anywhere in this ticket.

  **D50 spatial support assessment: `VERY_SPARSE`** (descriptive only, an
  explicit project heuristic for planning purposes -- never a
  physical/statistical threshold). Only 34.1% of chainage stations have a
  surface PSA sample within 1000 m, and only 5 samples yield a usable D50;
  whether to build a continuous pipeline D50 field in a later ticket is
  left to the external scientific reviewer.
- `MAR-008A`: external review found that `derive_grain_percentiles`
  (`sediment/grain_size.py`) validated a percent-unit phi-bin total against
  the sample's `WEIGHT` only for mass-unit (`grams`) bins -- a percent-unit
  distribution's own populated-bin total was never checked against 100%,
  so a materially incomplete distribution (e.g. bins summing to 80%) could
  have been silently renormalized into a false whole-sample D10/D50/D90.
  Fixed: a new `PHI_PERCENT_TOTAL_TOLERANCE_PCT` (2 percentage points, an
  explicit project data-QA heuristic, never a physical threshold) gates
  percent-unit bins -- a total outside `100% +/- 2pp` now returns
  `INVALID_TOTAL` with null D10/D50/D90, while the original
  (non-renormalized) total remains recorded in
  `phi_total_before_normalization`. This check is additional to, not a
  replacement for, the existing gravel/sand/mud whole-sample coverage guard
  and the mass-bin/`WEIGHT` check, both of which are unchanged and continue
  to dominate the partial-fraction case (a sand-only distribution that
  itself sums to ~100% is still correctly rejected as `INSUFFICIENT_BINS`
  before this new check would ever run).

  Independently re-running `build-sediment-evidence` against the real
  PL854 data confirms the previous 5 valid-D50 records (`65218674,
  65222864, 65235166, 65243220, 65247338`) are unaffected and byte-for-byte
  identical -- all 5 use `PHI_UNITS=grams`, not `percent`, so this fix
  never touches them. Of PL854's 27 real PSA records, 6 use
  `PHI_UNITS=percent`; all 6 already failed as `AMBIGUOUS_BIN_SCHEME` for
  an unrelated, pre-existing reason (non-uniform phi-bin spacing) before
  ever reaching the new total check, so the new validation had zero
  observable effect on this specific dataset -- a genuinely verified
  outcome, not one forced to preserve the prior count. D50 spatial support
  assessment is unchanged: `VERY_SPARSE`. This ticket also corrected
  wording in `sediment/evidence.py` and this README that could be read as
  "surface evidence = present-day seabed sediment" -- `surface_evidence_class`
  is a vertical-position/sampling-relationship classification at collection
  time only (see the `MAR-008` bullet above); no age cutoff was introduced,
  and real surface/subsurface classifications were not changed.
- `MAR-009`: PL854 metocean forcing evidence base
  (`providers/metocean/{copernicus,acquisition}.py`,
  `metocean/{current,wave,evidence}.py`) from three separate, never-blended
  Copernicus Marine products, each mapped from the 941 dense chainage
  stations onto a much smaller set of real model grid cells ("support
  nodes") via nearest-wet-cell assignment -- never 941 fabricated
  independent time series, never bilinear interpolation of data or masks:

  1. **Primary current** -- `NWSHELF_ANALYSISFORECAST_PHY_004_013`
     (`cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i`, confirmed live): ~1.5 km
     3D hourly instantaneous current. Per support node/hour, the **deepest
     valid standard level** with finite `uo`/`vo` is selected and stored as
     `deepest_valid_standard_level_current` -- explicitly **NOT** the
     model's native terrain-following bottom cell, and never called
     "bottom current"/"seabed current" anywhere in the codebase (naming
     enforced by a schema-regression test). `height_above_model_bed_m` is
     carried alongside and flagged if negative.
  2. **Long-term surface current context** --
     `NWSHELF_MULTIYEAR_PHY_004_009`
     (`cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i`, confirmed live, 1993
     onward): `LONG_TERM_SURFACE_CURRENT_CONTEXT` role only, hourly
     instantaneous 2D surface current -- never the daily 3D mean on the
     same product (`cmems_mod_nws_phy-uv_my_7km-3D_P1D-m`, a 25-hour
     tide-removing average, explicitly forbidden as a hard rule), never
     used to fill a missing primary-current value, never downscaled.
  3. **Wave climate** -- `NWSHELF_REANALYSIS_WAV_004_015`
     (`MetO-NWS-WAV-RAN`, confirmed live, 1980 onward): 3-hourly `VHM0`/
     `VTPK`/`VTM02`/`VTM10`/`VMDR` (+ Stokes drift where available). `VMDR`
     is preserved as the raw wave FROM-direction
     (`wave_mean_direction_from_deg`); a TO-direction is only ever derived
     for convenience, never replacing the original.

  Current direction (`current_direction_to_deg`) and wave direction
  (`wave_mean_direction_from_deg`) use opposite conventions by design (TO
  vs FROM) and are never confused or arithmetic-averaged -- directional
  summaries use proper circular statistics. Model bathymetry (`deptho` on
  each product's own static dataset) and the canonical MAR-006
  `depth_lat_m` (LAT datum) are carried side by side, never subtracted or
  compared as an "error"
  (`canonical_model_bathymetry_vertical_datums_not_harmonised = true` in
  the metadata). Historical forcing evidence is cut off at least 48 hours
  behind the live analysis/forecast boundary, computed dynamically from
  the actual dataset time coordinates each run, never hard-coded.
  Acquisition uses the official Copernicus Marine Toolbox
  (`copernicusmarine`) with resumable, idempotent monthly/yearly chunks; a
  short-window surface-current-context ratio is reported as a descriptive
  diagnostic only, never a scale factor or bias correction.

  Real execution against PL854 (2026-09-04): all three dataset ids were
  confirmed live against the current Copernicus Marine catalogue
  successfully. Real data acquisition then stopped cleanly at
  `CopernicusAuthenticationRequiredError` -- no Copernicus Marine
  credentials are configured in this environment -- printing the exact
  `copernicusmarine login` steps an operator needs to run, per this
  ticket's explicit requirement to never attempt an interactive credential
  prompt or ask for a password/token in chat. The full pipeline (support-
  node mapping, deepest-valid-level selection, statistics, chainage
  assembly, metadata) is implemented and verified with 88 new offline
  tests plus opt-in live catalogue-reachability smoke tests; it has not
  yet processed real Copernicus current/wave data end-to-end pending that
  one-time authentication step. No bed shear stress, Shields parameter,
  sediment mobility, wave orbital velocity, erosion/deposition, scour,
  free-span, fatigue, or risk scoring is computed anywhere in this ticket.
  No further ticket has started.
- `MAR-009A`: the real MAR-009 acquisition subsequently completed and
  external review of its actual output found a genuine vertical-eligibility
  integrity failure: `select_deepest_valid_standard_level`
  (`metocean/current.py`) checked only that `uo`/`vo` were finite, never
  the Copernicus model's own bathymetry. On **100% of the 260,764 real
  primary-current hourly rows across all 14 route-used support nodes**,
  standard levels well below that cell's own model bathymetry (up to 75 m
  deep against a bathymetry of ~23-30 m) still carried finite `uo`/`vo` --
  direct inspection showed an identically-repeated "held/padded" fill
  pattern below where the real profile stops varying, never a genuine
  measurement. Fixed: a depth candidate is now eligible only when `uo`/`vo`
  are finite AND `depth_m <= model_bathymetry_m + tolerance` AND, where the
  static 3D `mask`'s own depth coordinate is confirmed exactly aligned to
  the dynamic dataset's (`check_depth_coordinate_alignment` -- true for
  this real product pair), that mask cell is wet; neither condition alone
  would have caught every contaminated case at the one real node directly
  inspected, so both are required and share one numerical tolerance with
  `height_above_model_bed_m`'s own validity check so the two can never
  disagree.

  A second, independent real bug (this ticket's Section 7): long-term
  surface current and wave normalization were called against every wet grid
  cell in the request bounding box, not just the cells actually assigned to
  a PL854 chainage station -- the real MAR-009 report showed 330 wave and
  18 long-term-current time-series node ids against only 14/4 actually
  route-used (the support-node *tables* were already correctly filtered;
  only the time-series normalization calls were not). Fixed:
  `_cmd_build_metocean_evidence` (`cli.py`) now builds
  `used_primary_nodes`/`used_long_term_nodes`/`used_wave_nodes` and
  normalizes only those. A third, defensive fix (Section 6): static and
  dynamic dataset grid indices are no longer assumed identical -- each
  node's canonical lon/lat is re-resolved against the dynamic dataset's own
  coordinate arrays (`reconcile_node_grid_indices`, refused beyond 10% of
  that axis's median grid spacing, never a guessed index) before every
  sample. For the real PL854 primary-current product pair, static and
  dynamic datasets happen to share identical coordinate arrays, so this
  path was not itself the cause of the observed contamination, but is
  exercised by dedicated index-shift, reversed-ordering, and
  unreconcilable-coordinate tests.

  Independently re-running `build-metocean-evidence` against the
  already-downloaded raw Copernicus chunks (idempotent manifest -- zero
  re-download confirmed across two full reruns, byte-identical raw file
  mtimes and an unchanged 111-entry manifest) and reopening all 9 canonical
  outputs directly confirms **ALL 260,764 canonical primary-current rows
  are now within the Copernicus model water column**
  (`height_above_model_bed_m` min=0.243 m, median=3.195 m, p95/max=4.868 m;
  zero violations beyond the 1e-6 m numerical tolerance). Current speed
  statistics changed materially now that the contaminating deep levels are
  excluded (mean 0.245 -> 0.338 m/s, p95 0.671 -> 0.698, p99 0.770 -> 0.814,
  max 0.954 -> 0.983 m/s; the corrected values are canonical, the old
  values are not preserved by design) and the time-series node counts
  collapsed to the correct route-used set (wave 330 -> 14, long-term
  current 18 -> 4). Station-to-node distance diagnostics (min/median/p95/
  max, never a confidence score) are now reported for all three products.
  Model bathymetry and the canonical MAR-006 LAT bathymetry remain
  deliberately unharmonised and unsubtracted
  (`canonical_model_bathymetry_vertical_datums_not_harmonised = true`); new
  metadata additionally records
  `dynamic_grid_coordinate_reconciliation_method`,
  `static_dynamic_coordinate_match_status` (all three products' used nodes
  fully reconciled), `primary_current_vertical_eligibility_rule`,
  `static_depth_mask_used = true` (the real static/dynamic depth
  coordinates align exactly for this product), and a below-model-bed
  finite-candidate diagnostic summary (QA only, 1,266,568 excluded
  candidates across all 260,764 timestamps -- never entering canonical
  statistics). 25 new offline tests were added across `test_current.py` and
  `test_metocean_evidence.py`; the full offline suite (498 tests) and
  repo-wide `ruff format`/`ruff check` pass clean. No bed shear stress,
  Shields parameter, sediment mobility, erosion/deposition, scour,
  free-span, fatigue, or risk scoring is computed anywhere in this ticket.
  Per this ticket's explicit instruction, external scientific review of the
  corrected height-above-bed, model bathymetry, current statistics, and
  support-node distances is required before MAR-010 (near-bed hydrodynamic
  formulation) begins -- no further ticket has started.
- `MAR-009B`: the real re-run confirmed a second, independent integrity
  finding: the Copernicus Marine Toolbox's `subset()` treats `end_datetime`
  as INCLUSIVE, so two adjacent monthly/yearly acquisition chunks each
  return their shared boundary instant, and
  `xr.open_mfdataset(..., combine="by_coords")` does not itself deduplicate
  -- the real primary-current record carried **18,626 raw hourly
  timestamps per node against the physically correct 18,600** (26
  duplicated internal monthly-chunk boundaries), pushing completeness to a
  physically-impossible ~100.1%; the same class of duplication was
  confirmed in the yearly long-term-current (33 duplicate boundaries) and
  wave (46 duplicate boundaries) acquisitions. Fixed at the CHUNK ASSEMBLY
  boundary, not inside any statistics function:
  `deduplicate_time_coordinate` (`providers/metocean/acquisition.py`)
  detects every duplicated timestamp, requires every data variable to
  agree at every duplicate occurrence (NaN treated as equal to NaN) before
  ever collapsing it to one canonical row, and raises
  `DuplicateTimestampConflictError` -- never silently picking a side -- the
  moment any duplicate's data disagrees; it never rewrites or deletes the
  raw NetCDF chunk files. Every `normalize_*` function now always receives
  an already-unique, already-monotonic time coordinate.
  `validate_temporal_integrity` (`metocean/evidence.py`) defensively
  re-confirms uniqueness/monotonicity/no-duplicate-`(node, time)`-rows on
  the normalized output itself, and a new `_completeness_pct` helper
  refuses (`TemporalCompletenessError`) to ever report completeness above
  100% for any of the three products -- a hard failure, never a silent
  clamp. `compute_long_term_surface_current_statistics` gained a
  `completeness_pct` field for the first time (Section 7 of the ticket).

  Independently re-running `build-metocean-evidence` against the
  already-downloaded raw chunks (zero re-download confirmed: byte-identical
  raw file mtimes and an unchanged 111-entry manifest) and reopening all 9
  canonical outputs directly confirms **primary current: exactly 260,400
  canonical rows (18,600/node x 14 nodes), 100.0% completeness at every
  node, zero duplicate `(node, time)` rows, ALL CANONICAL METOCEAN TIME
  COORDINATES ARE UNIQUE AND MONOTONIC** -- matching the ticket's
  independently-derived expected count exactly. Long-term surface current
  (1,174,464 rows, 4 nodes) and wave (1,895,264 rows, 14 nodes) show the
  same zero-duplicate, 100.0%-completeness result. Removing exact
  duplicate rows left the aggregate current-speed/Hs/Tp statistics
  materially unchanged (duplicates carried the same values as their
  originals, so only the row/completeness accounting was wrong, not the
  distribution) -- the old (MAR-009A-era) on-disk data itself still
  exceeded 100% completeness, so old-vs-new comparison values for the
  absolute statistics are reported as `n/a` (unavailable) rather than
  computed from data the new strict check correctly refuses to trust; the
  two node-count comparisons (wave 14, long-term current 4) are unchanged,
  as expected. 19 new offline tests were added across
  `test_metocean_acquisition.py` and `test_metocean_evidence.py`, including
  realistic monthly-current, yearly-long-term-current, and yearly-wave
  boundary-overlap cases and an explicit NaN-consistency case; the full
  offline suite (517 tests) and repo-wide `ruff format`/`ruff check` pass
  clean. No bed shear stress, Shields parameter, sediment mobility,
  erosion/deposition, scour, free-span, fatigue, or risk scoring is
  computed anywhere in this ticket -- no further ticket has started.
- `MAR-010`: current-only near-bed normalization
  (`metocean/{current_normalization,current_map}.py`) -- the first step
  beyond forcing evidence, and the first static map in this project.
  Normalizes the MAR-009B corrected primary-current reference sample
  (`deepest_valid_standard_level_current`, 0.243-4.868 m above the
  Copernicus model bed) to a standard 1.0 m above that SAME model bed,
  using the ticket's fixed logarithmic velocity-profile ratio
  `S(z_t,z_r,z0) = [ln(z_t+z0)-ln(z0)]/[ln(z_r+z0)-ln(z0)]` --
  `uo_1m/vo_1m = S * uo_ref/vo_ref` preserves direction exactly (`S` is a
  positive scalar shared by both components). Canonical role name
  `CURRENT_ONLY_LOG_PROFILE_SENSITIVITY` throughout -- never "bed
  current"/"seabed current"/"combined near-bed current" (this is
  current-only; wave-current bottom-boundary-layer interaction is
  explicitly deferred, `current_wave_interaction_applied = false`).
  Roughness is run as five FIXED sensitivity scenarios (SILT 5e-6 m
  through GRAVEL 3e-4 m, consistent with long-standing DNV F105/F109
  roughness classes without claiming certified compliance) -- never a
  canonical PL854 seabed roughness choice, never a BGS-Folk mapping, never
  a D50-derived field, never averaged into a "best estimate"; the
  sensitivity envelope (min/max across the five) is itself the output.
  Every row also carries `z_r_over_h_model` and an explicitly-named
  `log_profile_vertical_domain_status` against a conservative 0.30
  project screening heuristic (never a universal physical threshold) --
  rows outside it get null normalized values, never a silent
  extrapolation. `NormalizationCompletenessError` mirrors MAR-009B's own
  completeness invariant for this derived product.

  Contiguous current-support map sections (`current_reference_segments.gpkg`)
  dissolve runs of chainage stations sharing one real support node using
  `shapely.ops.substring` on the true canonical route geometry -- never a
  straight chord between chainage points, never 941 independently-coloured
  25 m cells (honest ~1.5 km model spatial support). The required static
  map (`maps/pl854_reference_current_forcing.png`, matplotlib + rasterio,
  `Agg` backend, a newly-declared project dependency) colours the route by
  the assumption-minimal `current_reference_speed_p95_m_s` only -- never
  one arbitrary roughness scenario's 1 m value, never a risk judgement --
  with a muted EMODnet bathymetry background, KP labels, scale bar, north
  arrow, and the top-3 native-p95 sections called out; endpoints stay
  "Source geometry start"/"Source geometry terminus" per the project's
  established direction-honesty stance.

  Real execution against PL854 confirms the vertical screen independently
  (never hard-coded): **0 of 260,400 canonical rows fall outside the 0.30
  screen** (z_r/h_model max = 0.163), producing exactly 1,302,000 hourly
  sensitivity rows (260,400 x 5) and a 70-row (14 nodes x 5 scenarios)
  stats table, 14 contiguous map sections (one per real route-used node,
  tiling the full 23,480.67 m route with zero gaps), and a rendered PNG.
  40 new offline tests were added across `test_current_normalization.py`,
  `test_current_map.py`, and `test_cli.py` (including a full synthetic
  end-to-end CLI fixture); the full offline suite (557 tests) and
  repo-wide `ruff format`/`ruff check` pass clean. No bed shear stress,
  Shields parameter, sediment mobility, scour, free-span, erosion/
  deposition, or risk scoring is computed anywhere in this ticket -- no
  further ticket has started.
- `MAR-011`: wave-only spectral near-bed orbital velocity
  (`metocean/{wave_orbital,wave_orbital_map}.py`) -- converts the MAR-009B
  canonical wave evidence into a near-bed orbital-velocity forcing product
  using the FIXED Soulsby & Smallman irregular-wave spectral approximation
  (`Tn = sqrt(h/g)`, `t = Tn/Tz`, `A = [6500 + (0.56 + 15.54*t)^6]^(1/6)`,
  `Urms = 0.25*Hs / [Tn*(1+A*t^2)^3]`). Canonical role name
  `WAVE_ONLY_SPECTRAL_NEAR_BED_ORBITAL_VELOCITY` throughout -- never "bed
  current"/"combined wave-current velocity"/"bed shear stress". `Tz` is
  always Copernicus `VTM02` (`tm02_s`) -- changing observed `VTPK`
  (`tp_s`, preserved as a diagnostic) or `VTM10` (`tm10_s`, preserved as
  context) alone never changes the canonical Urms, confirmed by dedicated
  regression tests. Water depth `h` comes from the WAVE product's own
  static `deptho` at the same real wave support node -- never the
  canonical MAR-006 LAT depth, never a current-product bathymetry
  substitute, and no current data are read anywhere in this ticket. Every
  row also carries an explicitly-named `orbital_velocity_method_status`
  against a 0.30->0.54 method-accuracy calibration domain (never called a
  universal physical threshold) -- rows outside it keep their raw
  Hs/Tm02/Tp/depth values but get null canonical Urms/equivalent
  amplitude, never a silent out-of-domain extrapolation.
  `hs_over_model_depth` is reported purely as a non-breaking-assumption QA
  diagnostic (min/median/p95/p99/max) and never gates/rejects a row. A
  real edge case worth naming: an exactly-zero depth yields a
  mathematically finite (not NaN) `Tn`/`t` via plain propagation, which
  would otherwise slip through as spuriously "within domain" -- caught by
  an explicit `is_depth_and_period_valid` check before classification, not
  relied-upon incidental NaN propagation alone.

  Contiguous wave-support map sections
  (`wave_orbital_reference_segments.gpkg`) mirror MAR-010's honest-
  spatial-support approach independently (a self-contained module, so
  MAR-010's already-shipped map is never put at risk by this ticket's
  changes) -- true route geometry via `shapely.ops.substring`, never 941
  independently-coloured 25 m cells. The required static map
  (`maps/pl854_wave_orbital_forcing.png`) colours the route by the
  assumption-minimal `orbital_rms_p95_m_s` only -- never Hs/Tp/equivalent
  amplitude/direction/risk -- and applies presentation fixes from the
  MAR-010 map review: a landscape canvas sized from the TRUE displayed
  content (route + background raster) rather than the route alone, at
  most 3 hotspot labels stacked to avoid collisions, and simplified
  km-precision hotspot KP labels (`KP 6.14-8.16`) rather than the
  survey-grade `+metres` form still used for the canonical `kp_start`/
  `kp_end` segment attributes themselves.

  Real execution against PL854 confirms a genuine, expected finding
  (never hidden): **275,434 of 1,895,264 canonical rows (14.5%) fall
  outside the 0.54 calibration domain** (Tn/Tz median=0.428, max=1.301) --
  physically explained by short-period local wind-sea conditions (real
  Tm02 median 3.93 s) pushing `t` above the method's own accuracy domain;
  every one of those rows independently verified to retain its raw Hs
  while its canonical Urms is null, and every within-domain row (with
  valid Hs) has a real Urms (mean=0.029, p95=0.132, p99=0.238,
  max=0.717 m/s). Zero duplicate `(wave_node_id, time_utc)` rows, 14
  route-used wave nodes, 14 contiguous map sections tiling the full
  23,480.67 m route exactly as MAR-010's current sections did. 45 new
  offline tests were added across `test_wave_orbital.py`,
  `test_wave_orbital_map.py`, and `test_cli.py`; the full offline suite
  (602 tests) and repo-wide `ruff format`/`ruff check` pass clean. No bed
  shear stress, friction factor, Shields parameter, sediment mobility,
  erosion/deposition, scour, free-span, or risk scoring is computed
  anywhere in this ticket -- no further ticket has started.
- `MAR-011A`: corrects a real scientific error in MAR-011 -- `Tn/Tz <= 0.54`
  was incorrectly treated as a hard validity boundary, silently nulling
  the canonical Urms for 275,434 of 1,895,264 real PL854 rows (14.5%)
  solely because `t > 0.54`. Soulsby (2006), HR Wallingford Report TR155
  Section 3.1, actually states only that the approximation fits the exact
  computed spectral value to BETTER THAN 1% for `0 <= Tn/Tz <= 0.54` and
  that "orbital velocities are very small" above it -- an ACCURACY
  QUALIFICATION, never "the method is invalid here". Fixed:
  `wave_orbital_velocity_rms_near_bed_m_s` is now computed for every
  physically valid row (finite Hs>=0, finite Tz>0, finite depth>0)
  regardless of `t`; only genuinely invalid Hs/Tz/depth still nulls it.
  The status column is renamed `soulsby_smallman_accuracy_status`
  (`WITHIN_.../OUTSIDE_REPORTED_BETTER_THAN_1PCT_ACCURACY_RANGE`) to make
  the semantics explicit. A real edge case caught in review: an exactly-
  zero depth yields a mathematically finite (not NaN) `t` via plain
  propagation, which would otherwise misclassify as "within range" --
  caught by an explicit depth/period validity check before classification,
  independent of the incidental NaN-propagation path. Data completeness is
  now cleanly separated from method-accuracy coverage
  (`input_valid_count`/`input_data_completeness_pct` vs
  `within_reported_1pct_accuracy_count`/`outside_reported_1pct_accuracy_count`)
  -- completeness is reduced only by genuinely invalid/missing data, never
  by `t > 0.54`; a conditional `orbital_rms_p95_within_reported_1pct_accuracy_range_m_s`
  secondary QA statistic is retained under its own explicit name but never
  substituted for the canonical full-record percentiles. The map/CLI's
  temporal-integrity check (duplicate/non-unique/non-monotonic
  `(wave_node_id, time_utc)`) is upgraded from a warning to a hard failure.

  Real re-execution against PL854 (no network) confirms the canonical p95
  shifted down modestly and consistently once the wrongly-nulled ~14.5% of
  (correctly small, per TR155) rows are included: route-wide Urms p95
  0.132 -> 0.128 m/s, p99 0.238 -> 0.233 m/s, max unchanged at 0.717 m/s
  (the largest sea states were already within-range); per node the p95
  shift ranges from -3.2% to -9.7%. Input data completeness is now 100% at
  every node (was previously conflated with the accuracy-range split).
  8 new/rewritten regression tests confirm: `t > 0.54` with valid inputs
  still yields a finite Urms; genuinely invalid depth/Tz still yields none;
  completeness is unaffected by `t > 0.54`; the canonical and conditional
  percentiles are independently and correctly scoped; and duplicate/
  non-monotonic rows now fail the CLI hard. Full offline suite (609 tests)
  and repo-wide `ruff format`/`ruff check` pass clean. No wave-current
  interaction, bed shear stress, friction factor, Shields parameter,
  sediment mobility, or risk scoring is computed anywhere in this ticket --
  no further ticket has started.
- `MAR-012`: combines the accepted MAR-010 current-only 1 m normalization and
  MAR-011A spectral wave orbital velocity into a contemporaneous wave-current
  bed-shear-stress sensitivity product (`metocean/{combined_bed_shear,
  combined_bed_shear_map}.py`), using the SOULSBY ALGEBRAIC interaction
  approximation only -- never a full Grant-Madsen iterative BBL solution,
  never an iterated apparent wave-enhanced roughness, never a wave-dispersion
  correction (`interaction_model = SOULSBY_ALGEBRAIC_WAVE_CURRENT_BED_SHEAR`,
  `full_grant_madsen_bbl_applied = false`). Current-only stress inverts
  MAR-010's own log-profile at the SAME z0 (`u_star_c = kappa*U_1m /
  ln((1+z0)/z0)`, `tau_c = rho*u_star_c^2`); wave-only stress uses the
  MAR-011A equivalent amplitude (`sqrt(2)*Urms`, never raw Urms) and
  representative period (`1.28*Tz`, never observed VTPK) through competing
  smooth/laminar (`Rw<=5e5`) and rough friction-factor branches, with the
  canonical factor always the MAX of the two (never their average); Soulsby's
  algebraic mean/max combined-stress formulas fold in the current/wave axis
  angle (minimal 0..180 difference, folded around 90 to exploit the wave
  axis's own 180-degree symmetry). All five MAR-010 roughness scenarios are
  reused exactly (never re-chosen, never averaged), and the SAME scenario z0
  is used for current and wave stress within a row. A real edge case handled
  explicitly rather than left to bare IEEE-754 arithmetic: phi is
  deliberately null for zero current (avoiding the spurious `atan2(0,0)==0`
  direction), so `compute_soulsby_max_combined_stress_pa` explicitly branches
  on `tau_c==0`/`tau_w==0` rather than trusting `0 * NaN` to resolve to `0`.

  Two previously-independent real-world grids are reconciled by COORDINATE,
  never assumed identical by node-id string equality (Section 19's own
  explicit "this must be VERIFIED" instruction): `build_hydro_pairs`
  projects both products' stored lon/lat into the working CRS and matches
  nearest cells within a tolerance derived from the current grid's own
  median nearest-neighbour spacing, hard-failing
  (`UnreconciledHydroNodeError`) rather than silently pairing an
  out-of-tolerance cell. The combined 3-hourly series
  (`combined_bed_shear_3hourly.parquet`) is built from an EXACT-timestamp
  inner join between the hourly current and 3-hourly wave series (after
  spatial pairing) -- never combining independent percentile statistics,
  never interpolating. Long-term wave-only bed-shear context is computed
  separately from the FULL 1980-2026 wave record
  (`wave_only_bed_shear_long_term_stats.parquet`), with an
  overlap-vs-full-record p95 ratio reported strictly as REPRESENTATIVENESS
  CONTEXT, never a confidence score. The required map
  (`maps/pl854_combined_bed_shear_sensitivity.png`) colours the route by
  `tau_max_p95_sensitivity_max_pa` -- deliberately the UPPER BOUND across the
  five roughness scenarios, never a best estimate -- with hotspot labels
  showing the FULL p95 envelope (e.g. `p95 0.47-1.04 Pa`), never just the
  upper value.

  Real execution against PL854 (no network) VERIFIED, not assumed, that the
  current and wave products share the identical AMM15 support grid: all 14
  current/wave node pairs reconciled with coordinate separation exactly
  0.000000 m. The current-wave overlap runs 2024-07-20 03:00 to 2026-04-30
  21:00 UTC (5,199 expected/matched 3-hour timestamps, 100% completeness),
  producing 363,930 combined rows (5,199 timestamps x 14 hydro pairs x 5
  scenarios) and 14 contiguous map sections tiling the full 23,480.67 m
  route exactly as MAR-010/011's own sections did. A genuine, unforced
  finding: within this specific ~21-month overlap window, the wave Reynolds
  number never exceeded the 5e5 laminar/smooth-turbulent transition (max
  observed ~305,246, hand-verified against the raw amplitude/period columns)
  -- every row falls in `LAMINAR_BRANCH`, while the ROUGH friction branch
  still independently controls ~20% of rows (73,027 / 363,930) since the two
  branches are compared independently of that regime label. Stresses
  increase monotonically with roughness across all five scenarios for both
  components (GRAVEL always largest, SILT/FINE_SAND always smallest;
  confirmed as the most/least frequent p95-minimising/maximising scenario
  across all 14 segments) -- a physically-expected, non-random pattern. The
  overlap-period wave-only tau p95 (0.904 Pa) runs modestly above the
  full-record p95 (0.818 Pa, ratio 1.148), reported as context only, never
  substituted into the combined statistics. 62 new offline tests were added
  across `test_combined_bed_shear.py`, `test_combined_bed_shear_map.py`, and
  `test_cli.py`; the full offline suite (671 tests) and repo-wide `ruff
  format`/`ruff check` pass clean. MAR-012 computes hydrodynamic bed shear
  stress only -- no Shields parameter, critical shear stress, sediment
  mobility, erosion/deposition, scour, free-span, or risk scoring is
  computed anywhere in this ticket -- no further ticket has started.
- `MAR-013`: the first sediment-RESPONSE product (`sediment/
  {noncohesive_mobility,noncohesive_mobility_map}.py`) -- but since PL854
  has no defensible continuous D50 field, this never claims "the actual
  seabed sediment is mobile here" along the route. Instead it calculates
  `NONCOHESIVE_SEDIMENT_MOBILITY_CAPACITY`: for nine FIXED hypothetical
  noncohesive grain-size TEST scenarios (0.063-16 mm,
  `TESTED_NONCOHESIVE_GRAIN_SIZE_SCENARIOS_NOT_SITE_SPECIFIC_D50`), it
  computes grain-consistent skin friction and the Soulsby-Whitehouse
  critical Shields stress, then reports which TESTED grain sizes the real
  hydrodynamic forcing is capable of mobilising -- a forcing-CAPACITY
  product, never a continuous site-specific sediment-truth map. Critically,
  this never divides MAR-012's `tau_max_p95_sensitivity_max_pa` (an
  independent roughness-sensitivity envelope) by a D50-derived threshold:
  for each candidate D50, `z0_skin_m = d50_m/12` drives BOTH a fresh
  grain-related current/wave skin-friction recalculation (reusing MAR-012's
  own validated friction-branch and Soulsby combined-stress functions
  directly, per the ticket's own explicit reuse instruction, generalised
  from MAR-010's fixed 1 m reference height to the raw MAR-009B sample's
  own `height_above_model_bed_m`) AND the matching critical stress
  (`D* = d50*[g*(s-1)/nu^2]^(1/3)`; `theta_cr = 0.30/(1+1.2*D*) +
  0.055*[1-exp(-0.020*D*)]`; `tau_cr = theta_cr*(rho_sediment-rho_water)*g*d50`)
  -- the SAME D50 drives both sides of the comparison. `mobility_ratio =
  tau_max_grain_skin/tau_cr`; `>= 1` is threshold-crossing initiation-of-
  motion POTENTIAL only, never erosion/transport/scour/risk. Reuses
  MAR-012's verified coordinate-based hydro-pair reconciliation and
  exact-timestamp join directly rather than re-deriving them. The discrete
  `largest_tested_d50_with_p95_mobility_ratio_ge_1_mm` capacity field is
  never reported above the largest tested scenario (a passing 16 mm instead
  flags `CAPACITY_EXCEEDS_TESTED_GRAIN_SIZE_RANGE`), and `mobility_ratio_p95`
  monotonicity across the nine scenarios is VERIFIED per hydro pair, never
  assumed. BGS sediment evidence is used strictly: `mapped_250k_folk_class`
  is regional context only, never converted to a numeric D50; the five
  valid observed PSA D50 points are point context only
  (`POINT_OBSERVATION_NOT_INTERPOLATED_TO_PIPELINE`, count always derived
  from real data, never hard-coded); the BGS predictive product never
  enters the physics. The primary map
  (`maps/pl854_noncohesive_mobility_capacity.png`) colours the route with a
  DISCRETE (never continuously interpolated) scale matching the nine
  tested scenarios, overlaying the valid PSA points as visually distinct
  markers that never influence route colour; a secondary chainage profile
  PNG (`maps/pl854_mobility_capacity_profile.png`) shows the same stepped
  capacity on a log D50 axis alongside the same PSA points at their own
  true D50 and chainage.

  Real execution against PL854 (no network) independently confirmed every
  hand-computed threshold term (z0_skin, D*, theta_cr, tau_cr) bit-for-bit
  across all nine scenarios, and reproduced the textbook Soulsby-Whitehouse
  Shields-curve shape: theta_cr genuinely DIPS to a minimum (0.0302 at
  D*=20.3, the 1 mm scenario) before rising again at coarser grain sizes
  (0.0557 at D*=325, 16 mm) -- a real, unforced confirmation that the
  formula was implemented correctly, not a monotonic curve fitted to
  expectations. tau_cr itself still increases monotonically overall (0.120
  Pa at 0.063 mm to 14.18 Pa at 16 mm). Route-wide mobility ratio and
  threshold exceedance both decrease monotonically with grain size across
  all 14 real hydro pairs (p95 exceedance 3.50/59.8% at 0.063 mm down to
  0.114/0.0% at 16 mm) with ZERO monotonicity violations detected route-
  wide -- verified, not assumed. p95 capacity is 1 mm for 12/14 segments
  and 0.5 mm for 2/14, producing 655,074 combined 3-hourly rows (5,199
  timestamps x 14 hydro pairs x 9 grain scenarios) and 126 stats rows (14 x
  9) at 100% completeness. The five real valid observed PSA D50s (0.21-0.38
  mm, all Folk class "S") sit comfortably below the computed capacity
  ceiling -- a reassuring but never conflated cross-check between the
  capacity model and real point observations. 53 new offline tests were
  added across `test_noncohesive_mobility.py`,
  `test_noncohesive_mobility_map.py`, and `test_cli.py`; the full offline
  suite (724 tests) and repo-wide `ruff format`/`ruff check` pass clean. No
  Shields threshold above this, transport rate, erosion, deposition, scour,
  free-span, fatigue, or pipeline risk is computed anywhere in this ticket
  -- no further ticket has started.
- `MAR-014`: pipeline-specific scour-ONSET screening (`scour/
  {scour_onset,scour_onset_map,pipeline_condition}.py`) using Marini et
  al. (2024)'s combined wave-current generalisation of the Sumer et al.
  (2001) tunnel-scour-onset criterion -- never predicted scour depth,
  erosion depth, burial, exposure, free-span geometry, failure
  probability, or pipeline risk. PL854's real diameter (0.3048 m, 12 inch,
  a fixed constant -- never inferred from geometry/imagery) is OUTSIDE the
  source experimental envelope (D=0.05-0.10 m); every output states
  `PIPE_DIAMETER_OUTSIDE_SOURCE_EXPERIMENT_ENVELOPE` and
  `RESEARCH_SCREENING_EXTRAPOLATION_NOT_CALIBRATED_PL854_PREDICTION` --
  this is never presented as a validated field prediction. Three fixed D50
  screening scenarios (0.160/0.250/0.480 mm -- the Marini/Zang COMBINED-flow
  calibration envelope only, never all nine MAR-013 scenarios), three
  porosity scenarios (0.35/0.40/0.45), and five tested embedment ratios
  (e/D = 0/0.03/0.06/0.10/0.15) are FIXED sensitivity dimensions; the
  published onset equation is evaluated ONLY at these five explicit
  embedment scenarios -- never solved for an unconstrained continuous
  critical embedment, and a persisting onset at e/D=0.15 is flagged
  `ONSET_PERSISTS_AT_MAX_TESTED_EMBEDMENT` rather than extrapolated.

  Since the Marini/Zang laboratory experiments are effectively
  2D/codirectional, PL854's oblique current/wave directions are handled by
  an explicit, disclosed `PIPELINE_NORMAL_2D_SCREENING_PROJECTION` --
  current and wave orbital amplitude are projected onto the LOCAL route
  tangent (a numerical tangent from the true curved geometry at each
  hydro-pair section's own chainage midpoint, never a whole-route chord),
  reusing MAR-012's angle-folding function directly since a pipe tangent,
  like the wave axis, is an undirected line with 180-degree symmetry.
  `oblique_flow_extension_directly_validated_by_source_experiments = false`
  throughout. Current is reconstructed at each tested embedment's own
  pipe-top height (`z_top = D*(1-e/D)`) using MAR-013's `z0_skin = d50/12`
  roughness convention and MAR-012's log-profile inversion, generalised to
  an arbitrary target height rather than MAR-010's fixed 1 m. The Marini
  wave Shields parameter, KC (Eq. 30, with explicit `COMBINED`/`WAVE_ONLY`/
  `CURRENT_ONLY`/`CALM` branches -- current-only uses the original Sumer et
  al. constants a=0.025/b=0.5 directly rather than forcing the generalised
  equations through a genuine `theta=0`/`KC=infinity` indeterminate form),
  and the combined-velocity alpha/beta polynomial are all computed
  independently of MAR-012/013's own stress/mobility definitions, per the
  ticket's explicit "use the model's own formulation" instruction. A real
  edge case caught in review: the pipeline-normal projection's `magnitude *
  sin(angle)` silently produced `NaN` (not the physically-correct `0`) for
  a genuinely zero-magnitude flow whose direction is null by the same
  zero-speed convention MAR-012/013 established -- fixed with an explicit
  zero-magnitude branch, mirroring this project's now-familiar `0 * NaN !=
  0` discipline; a second, related edge case (`Omega_forcing/Omega_threshold`
  giving `NaN` for the CALM branch, since `a`/`b` are deliberately never
  computed there) is guarded the same way.

  The compact 3-hourly output stores one row per `hydro_pair_id x time_utc
  x tested_d50_mm x porosity_scenario` (never a 5x embedment row fan-out)
  with five `onset_margin_eD_*` columns; embedment monotonicity
  (`Omega_forcing/Omega_threshold` must never increase with embedment) is
  verified empirically per row and hard-fails the run on any genuine
  violation beyond floating-point tolerance. Route sections reuse
  MAR-012/013's hydro-pair segmentation exactly; the required map colours
  each section by `p95_required_embedment_upper_class` using a DISCRETE
  six-class scale (0/0.03D/0.06D/0.10D/0.15D/>0.15D, with PL854 mm
  equivalents in the legend) plus a compact, clearly-separated 2018
  official-survey context box -- MAR-007's regional morphology
  (1991-1992 acquisition) is attached as `LEGACY_REGIONAL_CONTEXT_ONLY`
  and never enters the onset equation. The required official 2018
  PL854/PL855 condition benchmark (Ithaca Energy Anglia Decommissioning
  Environmental Appraisal, page 29, Tables 3.4-3.5) is recorded exactly,
  including its own internal prose-vs-table free-span-length inconsistency
  (10 m narrative vs. 23.2 m tabled) preserved verbatim rather than
  silently "corrected" -- and is never spatially distributed to PL854
  segments (`spatial_kp_locations_available_as_machine_readable_data =
  false` -- later refined by MAR-014A once official 2018 free-span KP
  positions were separately recovered; see below).

  Real execution against PL854 (no network) hand-verified every threshold
  term and reproduced the expected physical behaviour: at zero embedment,
  33.2-41.0% of real combined current-wave states (across the 9 D50 x
  porosity scenarios) already reach the onset criterion; this collapses to
  0.0-0.3% at just e/D=0.03 and to ~0% by e/D=0.06 -- so ALL 14 real hydro
  pairs, across all 9 sensitivity scenarios, resolve to the SAME p95
  required embedment class (0.03D, 9.1 mm) with zero monotonicity
  violations across 655,074 real combined rows. PL854's real diameter
  falls outside the source envelope as expected, but projected real
  current states fall inside the source Uc range 71.8% of the time and
  inside the source KC range 21.1% of the time (Uw only 10.3% -- an honest
  applicability finding, never used to discard results). 77 new offline
  tests were added across `test_scour_onset.py`, `test_scour_onset_map.py`,
  `test_pipeline_condition.py`, and `test_cli.py`; the full offline suite
  (801 tests) and repo-wide `ruff format`/`ruff check` pass clean. No
  equilibrium scour depth, scour propagation, sediment transport rate,
  exposure prediction, free-span prediction, fatigue, or risk scoring is
  computed anywhere in this ticket -- no further ticket has started.

- `MAR-014A`: recovers EXPLICIT spatially-resolved free-span survey evidence
  (`scour/freespan_evidence.py`, `scour/freespan_evidence_map.py`,
  `validation/freespan_model_context.py`) from a second official source --
  Ithaca Energy (UK) Limited's "Pipelines and Umbilical Comparative
  Assessment" (April 2020), Appendix B Table B.1 -- which tabulates KP/
  Easting/Northing/length/height for 2012, 2014, and 2018, correcting
  MAR-014's prior assumption that free-span locations aren't
  machine-readable. Table B.1's own scope is the PIGGYBACKED PL854/PL855
  corridor, never PL854 alone: every one of the 17 recovered events (2012:
  2/23.20 m, 2014: 7/68.23 m, 2018: 8/97.42 m -- all three sums verified
  exactly against the source's own rounded totals) carries
  `asset_scope=PL854_PL855_PIGGYBACK_CORRIDOR` and
  `individual_line_attribution=UNRESOLVED`, and is stored as a small
  TRACKED CSV resource (`src/marine_engine/resources/
  anglia_table_b1_freespans.csv`), never scraped at runtime, guarded by its
  own checksum verification against the source's rounded per-year
  statements.

  The source CRS is NOT stated, so it was inferred empirically rather than
  assumed: two candidates (EPSG:23031 ED50/UTM31N vs. EPSG:32631
  WGS84/UTM31N) were transformed onto the TRUE canonical PL854 route and
  scored against a formal five-criterion acceptance guard (median/max
  point-to-route distance, materially-smaller-than-alternative, a coherent
  KP-vs-chainage linear fit, consistent orientation) that hard-fails
  (`CRSReconciliationError`) rather than silently choosing a CRS. Real
  execution decisively accepted EPSG:23031 (median distance 1.03 m vs.
  172.81 m for the naive EPSG:32631 alternative, R²=0.999994) and confirmed
  the survey KP direction is REVERSED relative to canonical chainage --
  derived from the fit's own sign, never hard-coded in advance. Every event
  is projected independently onto the real route (never inheriting the
  source's own Start as the canonical route start), with canonical
  chainage min/max derived per event and event geometry built as a true
  `shapely.ops.substring` of the curved route, never a straight chord. Two
  2014 events (near source KP 0) landed exactly at the canonical route's
  own terminus -- a genuine, disclosed edge case where the physical survey
  extends slightly beyond the digitized PL854 geometry -- surfaced via the
  same `endpoint_*_route_distance_m` fields rather than hidden.

  A data-derived (never hard-coded) 2014-partial-coverage check found that
  Table B.1 lists no 2014 event near two zones (~KP 11.85-11.91 and ~KP
  22.89-22.90) where 2012 and/or 2018 both report one -- surfaced as an
  explicit warning on the historical map and in the reconciliation
  metadata, always phrased as a coverage observation, never as proof a
  free span was absent; no automatic cross-survey event matching
  (persistent/migrated/disappeared/newly-formed) is ever performed. The
  8-event 2018 subset is joined (Section 17, `validation/
  freespan_model_context.py`, matching that package's own pre-existing
  "validation of model outputs against observed data" docstring) against
  MAR-012/013/014's own already-computed segment outputs (bed-shear
  sensitivity, largest passing D50, required embedment class -- the latter
  honestly shown as spatially uniform, 0.03D route-wide) purely for
  side-by-side human review -- explicitly NO score, probability, rank, or
  accuracy metric anywhere in that output. Segment-level 2018 event counts
  use genuine chainage-interval overlap (never midpoint-only): 3 of PL854's
  14 hydro-pair sections contain at least one 2018 event. MAR-014's 2018
  condition benchmark metadata was refreshed in place (aggregate numbers
  never touched) to replace the blanket
  `spatial_kp_locations_available_as_machine_readable_data=false` with six
  precise flags distinguishing "free-span positions now tabulated" from
  "exposed-section positions still unavailable" and "individual PL854-vs-
  PL855 attribution still unresolved".

  Three new maps render self-contained (never importing MAR-012/013/014's
  own map modules): the 8-event 2018 evidence map, a 17-event historical
  map (2012/2014/2018, colour-by-year, with the 2014-coverage-gap warning
  in its footer), and a three-panel model-context profile showing
  MAR-012/013/014's outputs honestly juxtaposed against the 8 event
  positions. 44 new offline tests were added across `test_resources.py`,
  `test_freespan_evidence.py`, `test_freespan_evidence_map.py`,
  `test_freespan_model_context.py`, `test_cli.py`, and
  `test_pipeline_condition.py`; the full offline suite (845 tests) and
  repo-wide `ruff format`/`ruff check` pass clean. No exposure probability,
  free-span probability, scour depth, VIV, fatigue, or arbitrary risk score
  is computed anywhere in this ticket -- no further ticket has started.

- `MAR-014B`: repairs a real evidence gap MAR-014A left behind -- the
  tracked Table B.1 CSV's `source_comment`/`source_page` columns were
  blank, losing the source's own per-row remarks about which spans it
  calls "the same span" across surveys, which changed length, and which
  areas simply were not surveyed earlier. `source_page=44` (never
  zero-based PDF numbering) is now set on all 17 rows, and 8 rows carry
  their own preserved comment text (`scour/pipeline_condition.py` and the
  CSV itself are unchanged in shape; only these two columns are filled
  in). A brand-new tracked resource
  (`resources/anglia_table_b1_freespan_relationships.csv`, loaded via
  `load_anglia_table_b1_freespan_relationships`) separately encodes 14
  machine-readable relationships: 8 event-level rows -- 3 same-span pairs
  (2014-05<->2018-06 +8.14 m in 2018, 2014-06<->2018-07 +0.8 m in 2018,
  both length deltas verified exactly against the tracked lengths; a third,
  2012-02<->2014, whose SPECIFIC 2014 counterpart is genuinely ambiguous
  from this transcription and is therefore recorded with a null
  `event_id_b` rather than a guessed one) and 3
  not-surveyed-in-2014 statements (2018-01/02/03) -- plus 6 pure
  group/narrative statements (e.g. "the 2014 survey mapped only Anglia
  West/Anglia A NUI/LOGGS, not the full route") that Table B.1's layout
  does not tie to specific event IDs, and which are therefore NEVER forced
  into a fabricated pair. `relationship_type` and `attribution_method` are
  both hard allow-listed at load time (`DIRECT_TABLE_COMMENT` /
  `DIRECT_DOCUMENT_NARRATIVE` only -- `SPATIAL_NEAREST_NEIGHBOUR` and
  `INFERRED_MATCH` are deliberately never valid values, so a future edit
  that tried to sneak in a spatially-inferred relationship would hard-fail
  the loader, not silently pass).

  A new module (`scour/freespan_temporal_provenance.py`) joins these
  source-stated relationships against MAR-014A's own canonical chainage
  (context only, never used to invent an unstated pairing) and writes
  `pipeline_condition/anglia_freespan_temporal_relationship_evidence.parquet`
  (14 rows, role `SOURCE_STATED_HISTORICAL_FREESPAN_EVOLUTION_EVIDENCE`,
  no score/probability/prediction/rank field anywhere) plus explicit
  survey coverage semantics: 2014 is
  `PARTIAL_ROUTE_COVERAGE_SOURCE_STATED` (citing the source's own
  Anglia-West/A-NUI/LOGGS-only statement), while 2018 is recorded only as
  `PRE_DECOMMISSIONING_SURVEY` -- deliberately never promoted to a
  route-wide "no freespan here" negative-coverage label
  (`2018_full_route_negative_label_assumption_applied = false`) without a
  future source explicitly stating that stronger claim. A new, genuinely
  optional map (`maps/pl854_source_stated_freespan_evolution.png`) draws a
  link ONLY between unambiguous same-span/length-change event pairs and a
  marker for single-event statements, with every pure narrative statement
  shown in a text box instead of an invented geometry. The existing
  model-context profile (Section 11 presentation fix) now also draws each
  2018 event's REAL observed span width (~0.2-23 m, never enlarged to stay
  visible against the 23.5 km route) and fixes a label-crowding bug where
  events anchored near chainage 0 could stagger into negative offsets and
  overlap the y-axis itself -- fan-out direction is now chosen per cluster
  to always point away from whichever route end is nearer. 17 new offline
  tests were added across `test_resources.py`,
  `test_freespan_temporal_provenance.py`, `test_freespan_evidence_map.py`,
  and an extended `test_cli.py` end-to-end check; the full offline suite
  (862 tests) and repo-wide `ruff format`/`ruff check` pass clean. No
  susceptibility score, probability, prediction model, or automatic
  cross-survey event matching was created anywhere in this ticket -- no
  further ticket has started.

- `MAR-014C`: reconciles PL854/PL855 against a THIRD official source -- NSTA's
  own Pipeline Freespans registry (`providers/nsta_freespan.py`,
  `scour/nsta_freespan_reconciliation{,_map}.py`) -- and repairs one small
  MAR-014B gap (the six `DIRECT_DOCUMENT_NARRATIVE` relationship records,
  `REL-09`..`REL-14`, now correctly cite page 15/"Section 3 summary
  description", never Table B.1's own page 44). The ticket's own supplied
  URL (`data.nstauthority.co.uk`) does not resolve -- confirmed genuine
  NXDOMAIN against a public resolver, not a sandbox restriction, since
  `services-eu1.arcgis.com` (the domain `nsta.py` already uses) and the
  bare `nstauthority.co.uk` both resolve fine. The real services were found
  the same way `nsta.py`'s own were: the public ArcGIS Online item-search
  API (`owner:NSTA_GIS`), landing on "UKCS offshore infrastructure pipeline
  freespans (WGS84)" and its "removed" counterpart, both at layer id 1
  (matching `nsta.py`'s own convention -- never the ticket-assumed
  `FeatureServer/3`/`9`, which belonged to the non-existent host).

  The one live acquisition this ticket performs (`ingest-nsta-freespan-
  registry`, `NSTAPIPNO IN ('PL854','PL855')` against both layers,
  `outFields=*`) returned **zero features in both the current and removed
  layers** (953 + 25 = 978 total freespan records; 222 distinct NSTAPIPNO
  values across both). Per the ticket's own explicit instruction, this was
  investigated rather than silently broadened to fuzzy `PIPE_NAME`
  matching: every distinct `NSTAPIPNO` value, plus the older `LEGACY_ID`/
  `LEG_P_ID` identifier fields (still formal fields, never fuzzy text), was
  checked for "854"/"855" and found none; a diagnostic (never
  match-affecting) `PIPE_NAME` search for "ANGLIA"/"LOGGS" found only
  unrelated pipelines sharing the LOGGS terminal (PL2643, PL454). PL854/
  PL855 are therefore, as of this real acquisition, genuinely absent from
  NSTA's line-specific freespan registry -- a real, reportable outcome
  preserved as-is, never fabricated or routed around.

  `build-freespan-registry-reconciliation` runs fully offline from that
  cached snapshot: it never snaps an NSTA feature onto the route before
  measuring its real separation, reconciles chainage the same coordinate-
  projection way as MAR-014A, and would classify any future PL854/PL855
  NSTA record via multiple independent diagnostics (interval overlap,
  midpoint separation, length/height/survey agreement) -- a
  `STRONG_CROSS_SOURCE_MATCH` always requires clear spatial correspondence
  AND at least one independent attribute agreement, never spatial
  proximity alone; multiple equally-plausible candidates resolve to
  `AMBIGUOUS_MULTIPLE_NSTA_CANDIDATES`, never an arbitrary winner; a
  coincident PL854/PL855 pair is labelled
  `PIGGYBACK_COINCIDENT_FREESPAN_RECORDS` rather than silently keeping one
  copy. With the real (empty) snapshot, all 8 2018 Table B.1 events
  resolve to `NO_NSTA_CROSS_SOURCE_MATCH`, and `individual_line_attribution`
  stays `UNRESOLVED` exactly as MAR-014A left it -- the original Ithaca
  corridor evidence is never mutated, only ever supplemented in a separate
  derived file. Two new maps
  (`pl854_nsta_table_b1_freespan_reconciliation.png`,
  `pl854_2018_freespan_attribution_crosswalk.png`) render correctly even
  with zero NSTA records, honestly showing the real outcome rather than an
  empty/broken plot. Per the ticket's own explicit scope limit, no model
  accuracy, ROC/AUC, susceptibility score, or probability is computed
  anywhere -- the current hydrodynamic model (2024-2026 forcing) is not
  contemporaneous with the 2018 survey. 29 new offline tests were added
  across `test_nsta_freespan_provider.py`, `test_nsta_freespan_reconciliation.py`,
  `test_nsta_freespan_reconciliation_map.py`, and an extended `test_cli.py`,
  plus 3 new `live`-marked tests (`test_nsta_freespan_live.py`, excluded by
  default, documenting the real zero-result acquisition -- 32 new tests in
  total); the full offline suite (893 tests) and repo-wide `ruff format`/
  `ruff check` pass clean.

- `MAR-015`: a positive-only evidence AUDIT, not model validation --
  (`validation/freespan_context_audit{,_map}.py`) asks whether
  MAR-007/008/010/011A/012/013/014's independently-built context variables
  show any obvious spatial distinction at the 8 official 2018 corridor
  freespan events, and what evidence/resolution gaps block a defensible
  predictive model. The 8 events stay `CORRIDOR_LEVEL_OBSERVED_FREESPAN_
  EVENTS` (never rewritten to a PL854-only label), and a route section with
  no tabulated event is `NO_TABULATED_2018_EVENT_IN_THIS_SUPPORT_SECTION`
  (never NEGATIVE/SAFE/STABLE) -- no formal negative-label dataset is ever
  created. Confirmed on the real PL854 route: the 8 events occupy only
  **3 of the 14** independent hydrodynamic support sections (5/2/1 events
  each), so every descriptive comparison here has an effective sample size
  of 3, not 8 -- reported honestly rather than inflated. Every feature is
  audited independently (never fused into a combined index): count/min/
  median/max and a raw range-overlap flag only, no p-value, odds ratio, or
  effect size anywhere. The real result is a genuinely mixed, honestly
  reported one -- current/wave/combined-shear p95 and MAR-013 mobility
  capacity show **no** spatial discrimination (event sections' ranges fully
  overlap the route background, e.g. current p95 event range 0.43-0.67 m/s
  vs. background 0.38-0.70 m/s), and MAR-014's embedment class is
  `SPATIALLY_UNIFORM_AT_CURRENT_SUPPORT` (1 route-wide value, 0.03D, so it
  cannot discriminate anything by construction) -- but MAR-007's local
  relief and slope (a legacy, 1991-1992-sourced, 100 m-grid morphology
  feature that PREDATES the 2018 survey by ~26 years) show events confined
  to the 77-92 percentile (top quartile) of both. This is reported as a
  literal, descriptive observation, explicitly NOT causal validation: the
  one variable with an apparent spatial coincidence is also the one
  variable that is temporally mismatched and coarsest relative to the
  ~0.2-23 m observed span scale, while every 2024-2026 hydrodynamic-forcing
  feature (the temporally closest to present, still 6+ years post-survey)
  shows nothing. `compute_resolution_gap_statements` spells out the honest
  reading per event (e.g. a ~1500 m current/wave model cell is 150x larger
  than a 10 m observed span -- forcing CONTEXT at that location, never
  10 m hydrodynamic resolution). Five outputs are written under
  `validation/` and `maps/` (per-event and per-section audit parquets, an
  evidence-readiness JSON, a metadata JSON carrying explicit
  `negative_labels_created=false`/`model_validation_performed=false`/
  `classifier_fitted=false`/`score_created=false` flags, and three PNGs --
  a route map with true-length freespan intervals and an explicit
  scale-comparison box, a 4-panel small-multiple with no fitted trend, and
  a neutral feature-evidence table with no traffic-light colouring) via
  `uv run marine-engine audit-freespan-context configs/pl854.yaml` (fully
  offline). 24 new tests across `test_freespan_context_audit{,_map}.py`
  cover the positive-only/no-negative-label discipline, event-independence
  counting, temporal/support-resolution honesty, the dual-mode (continuous
  + categorical) audit of MAR-013's empirically 2-valued mobility feature,
  range-overlap correctness, source-stated-only temporal matching, true-
  width rendering, and the absence of any score/probability/accuracy term
  anywhere in the output schemas; the full offline suite (917 tests) and
  repo-wide `ruff format`/`ruff check` pass clean.

- `MAR-016`: a DATA-DISCOVERY / ACCESS-RECOVERY audit, never a new model --
  asks whether MAR-015's demonstrated gap (pipeline-scale/current-era
  seabed morphology, since MAR-007 is 1991-1992-sourced) can actually be
  closed with verified open data. A new provider
  (`providers/bgs_offshore_surveys.py`) queries BGS's real, public OGC API
  (`ogcapi.bgs.ac.uk`, collection `offshore-oil-gas-site-surveys` -- its
  own description states "BGS do not hold the data") spatially against the
  real PL854 AOI, real bbox + CQL2 `filter` queries, never the interactive
  viewer. Real result: **14 candidates** found (9 route-intersecting, 2
  AOI-only, 3 nearby-only), and all 5 of the ticket's named candidates
  confirmed live, including a genuine demonstration of Section 3's own
  warning -- `bgs_ref_no=GB02SS0003` ("Anglia A") and `decc_ref_no=GS_807`
  ("Bedevere rig site survey") both cite block 48/18, yet GS_807's real
  footprint sits ~12.9 km from the route (confirmed geometrically, never
  assumed from the shared block number). Several real footprints are
  near-perfect rectangles (`footprint_rectangularity` >= 0.96) consistent
  with a licensed block extent rather than an as-run survey track --
  flagged explicitly and never allowed to justify
  `CAN_ADDRESS_PIPELINE_SCALE_MORPHOLOGY_GAP` on its own; the real best
  nominal route-coverage candidates (GB02SS0003/GB03SS0002, 42.1%) are
  block-shaped and metadata-only, so neither qualifies. **Every one of the
  14 real candidates is `METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED`** (BGS's
  own collection statement) -- independently cross-validated against
  MAR-005's own, separate, earlier BGS check (a completely different
  GeoNetwork CSW protocol, `metadata.bgs.ac.uk`), which agrees exactly on
  dates/access/restriction for the two candidates both tickets checked. A
  dedicated Fugro-2018 dossier
  (`anglia_fugro_2018_recovery_dossier.json`) traces the official
  2018 pre-decommissioning survey to two real, cited gov.uk decommissioning
  filings (Ithaca Energy's programme + Hartley Anderson's April-2020
  Environmental Appraisal, which verbatim confirms 20-28 m depth, ~5 m
  sandwaves, 8 freespans/97 m, 19 exposed sections/519 m, and a poorly-
  sorted-coarse-sand sample) -- but the survey itself has **no discoverable
  catalogue record anywhere searched** (BGS, data.gov.uk/National Data
  Library, MEDIN, Marine Data Exchange); only the descriptive EA PDF is
  public, no MBES/XYZ/GeoTIFF grid was found, so its dossier status is
  `REPORT_EVIDENCE_ONLY`. Three real, verified-open Southern North Sea
  analog datasets (Cefas ECREC, JNCC/Cefas Inner Dowsing-Race Bank-North
  Ridge cSAC, Sheringham Shoal OWF) are recorded separately as
  `METHOD_DEVELOPMENT_ANALOG_ONLY` -- never PL854 evidence. Six outputs are
  written under `seabed_data/` and `maps/` (a 14-row survey-inventory
  parquet, an intentionally-empty file-inventory parquet since no
  downloadable file was ever found, the Fugro dossier, an access-gap JSON
  answering all 6 Section 19 questions literally, a coverage map colouring
  footprints ONLY by real route/AOI overlap class, and a timeline making
  the 1991-92/2018/2024-2026 temporal mismatch visually obvious) via
  `uv run marine-engine inventory-highres-seabed-data configs/pl854.yaml`
  (the one live step; every classification/rendering afterward is pure
  offline computation on the acquired data). 16 new tests across
  `test_highres_seabed_survey_inventory{,_map}.py` cover acquisition-year/
  update-date separation, the block-number-is-not-overlap discipline, real-
  geometry-vs-bounding-box intersection, metadata-only access never
  upgrading to open, equipment-parsed-never-implies-downloadable-data, and
  the absence of any score/probability/susceptibility term anywhere; the
  full offline suite (933 tests) and repo-wide `ruff format`/`ruff check`
  pass clean.

- `MAR-017`: a reusable HIGH-RESOLUTION SAND-WAVE MORPHOMETRY ENGINE, built
  and validated on a real, open Southern North Sea analog -- never PL854
  evidence, never a freespan model. Since MAR-016 found no verified open
  PL854-specific bathymetric grid, the engine is developed instead on
  JNCC/Cefas's real "Processed bathymetry from Haisborough, Hammond and
  Winterton cSAC" (CEND 11/11, RV Cefas Endeavour, 2011-06-11 to
  2011-06-21, UK Open Government Licence) -- `HHW-Bathy.zip`, confirmed
  127,960,429 bytes, downloaded and checksummed once and cached
  thereafter. Opening the real archive shows genuine ESRI Arc/INFO Binary
  Grid format (13 separate `hdr.adf`-rooted grids, never assumed GeoTIFF)
  at a real 1 m native resolution, EPSG:32631 -- read directly from inside
  the ZIP via GDAL's `/vsizip/` filesystem, never fully extracted (~7.1 GB
  uncompressed) to disk. The primary grid (`asciito_hhw_2` of the four
  "official processed bathymetry" candidates) is selected at runtime by
  evaluating each one's own best-achievable tile validity -- never
  hard-coded. Real result, reported honestly rather than forced: the
  ticket's 2000 m/1000 m canonical/floor tile sizes found **zero** tiles
  clearing 90% valid data anywhere (best achieved 49%/56%); the cascade
  had to continue to **250 m** (94.9% best) before finding 7 valid tiles --
  a genuine, demonstrated property of this dataset's real swath-line
  coverage pattern (not a full-corridor mosaic), not a bug, and explicitly
  flagged as below the ticket's own stated floor. A real bug WAS found and
  fixed during development: incompletely-filled nodata gaps were leaking
  the source's float32 nodata sentinel into the 2D FFT (a colourbar
  literally scaled to `1e37`), corrupting several tiles' spectral
  diagnostics -- fixed by always seeding gap-fill at the tile's own valid
  mean (never the raw sentinel) with a hard-clamp safety net, after which
  every diagnostic became physically sensible. Across the 7 valid tiles,
  3 were selected (by directional concentration, then peak-to-median
  power ratio -- a transparent ranking convenience, never a score) for 3
  cross-crest transects each; 3 of the 9 were honestly excluded
  (insufficient density along the transect itself, recorded in the table
  as `EXCLUDED_INSUFFICIENT_VALID_DATA`, never silently dropped) and the
  remaining 6 detected **11 real individual bedforms** (wavelength
  21-147 m, height 0.19-6.25 m, asymmetry -0.56 to +0.49) -- every one
  correctly flagged `FILTER_SCALE_SENSITIVE` under the 20/30/40 m
  comparison, an honest consequence of tiles this small relative to the
  detected bedform scale. Ten outputs are written under
  `processed/analogs/hhw_cend1111/` (never a PL854 layer -- every
  canonical row/JSON carries `pl854_evidence=false` and
  `scientific_role=HIGH_RESOLUTION_SANDBED_MORPHOMETRY_METHOD_DEVELOPMENT_
  ANALOG`): a 127-row source-file inventory, the tile/transect/individual-
  bedform parquets, a crest/trough point GeoPackage, a pipeline-transfer
  contract naming exactly what a future PL854 survey must provide, and
  four PNGs (a pre-morphology QA overview, the primary method figure
  showing native bathymetry / filtered surface / crest orientation+
  transects / an annotated profile, a bedform-statistics figure, and a
  dominant-wavelength map) via
  `uv run marine-engine build-analog-sandwave-morphometry configs/pl854.yaml`
  (one live download, cached thereafter). The reusable engine itself
  (`morphology/sandwave_morphometry{,_map}.py`) never references an HHW-
  specific identifier or coordinate -- verified by source inspection --
  and operates only on an arbitrary raster + geometry + metre-scale
  parameters, so it can later accept a real PL854 survey unchanged. 29 new
  tests across `test_sandwave_morphometry.py` (synthetic sinusoidal DEMs:
  wavelength/height/orientation recovery, positive-down sign handling,
  trend-removal wavelength preservation, ripple suppression, zero-phase
  crest positions, hand-exact asymmetry, nodata-tile exclusion) and
  `test_hhw_cend1111_analog.py` (analog-only flags, no HHW identifier in
  the reusable engine, no forbidden score/risk/migration-rate term
  anywhere) cover the required list; the full offline suite (962 tests)
  and repo-wide `ruff format`/`ruff check` pass clean. No further ticket
  has started.

- `MAR-017A`: a SUPPORT-SCALE INTEGRITY REPAIR of MAR-017, triggered by an
  external review that found the original HHW run had silently cascaded
  its CANONICAL 2D tile search below the ticket's own stated >=1000 m
  floor (down to 250 m), let those 250 m tiles populate the canonical
  validation outputs despite failing the required `tile_size_m /
  dominant_wavelength_m >= 3.0` eligibility test, and never gated a
  detected 21.1 m trough-to-trough feature out of the canonical 30 m
  bedform output (the Butterworth cutoff attenuates, it does not
  mathematically zero sub-cutoff content). All three are now fixed by
  splitting one conflated pipeline into two, permanently separate ones:
  `find_valid_tiles` (CANONICAL) tries only 2000 m then 1000 m and
  legitimately returns empty rather than cascading further; the OLD
  cascade-to-100 m logic now lives only in the new, explicitly-labelled
  `find_exploratory_small_support_tiles`
  (`EXPLORATORY_SMALL_SUPPORT_DIAGNOSTIC`, every row stamped
  `canonical_validation_eligible=false`/
  `reason=BELOW_MINIMUM_SPATIAL_SUPPORT`, never written into a canonical
  parquet). Canonical tile selection (`select_canonical_eligible_tiles`)
  now enforces the >=3-wavelengths condition with **zero fallback** --
  0/1/2 eligible tiles are selected as 0/1/2, never forced to 3 from an
  ineligible pool (the OLD `select_top_tiles`'s exact bug). A new explicit
  post-detection gate (`apply_canonical_wavelength_gate`) rejects any
  bedform with `wavelength_m < 30` from canonical output and reports the
  rejected count for QA, applied identically in both places a canonical
  bedform list is built (the persisted table and the method-figure
  profile) so the two can never drift apart. Re-run on the real HHW data
  with the repaired engine: canonical tile search found **zero** tiles at
  both 2000 m (best achieved 49.4%) and 1000 m (best achieved 55.9%) --
  correctly empty `tile_spectral_morphometry.parquet` /
  `transect_morphometry.parquet` / `individual_bedforms.parquet`, no stale
  `detected_profile_extrema.gpkg`; the separate exploratory search found 7
  tiles at 250 m (94.9% best), yielding 9 transects (3 honestly excluded
  for insufficient along-transect density, as before) and **10** canonical-
  gated bedforms (down from the old, ungated 11) -- the wavelength gate
  correctly rejected exactly 1 sub-30 m feature (the real, previously-
  leaking 21.1 m one), with every remaining bedform confirmed >=30 m
  (minimum 35.28 m) by independent re-verification. `pipeline_transfer_
  contract.json` now separates two questions that must never collapse
  into one: "is the generic engine implemented" (**YES**, unchanged by
  real-data outcome) vs. "has the full canonical workflow been validated
  on HHW" (**NO**, a real, honestly-reported property of this dataset's
  coverage, not a defect of the engine) -- both derived from the actual
  run's tile/bedform counts, never hard-coded. A new `analog_validation_
  gap.json` records the maximum valid fraction achieved at all four tried
  tile sizes, why HHW fails canonical support, and the next analog
  candidate identified for a future ticket (JNCC/Cefas's "Inner Dowsing,
  Race Bank and North Ridge cSAC" processed bathymetry, same CEND 11/11
  survey programme, not yet downloaded or processed). The exploratory
  method/statistics/dominant-scale figures now carry unmistakable red
  subtitles/banners identifying them as below the canonical floor (never
  presented as accepted validation), including a figure-layout fix so the
  banner text never collides with the plot title. 12 new MAR-017A-lettered
  tests (Section 15's A-L, split across both test files by which module
  they exercise) plus 3 pre-existing tests updated for the new function
  signatures/behaviour; the full offline suite (975 tests, 25 live/network
  tests correctly deselected) and repo-wide `ruff format`/`ruff check`
  pass clean. No further ticket has started.

- `MAR-017B`: a SECOND OPEN-ANALOG canonical real-data validation attempt
  for the same reusable sand-wave morphometry engine, run against JNCC/
  Cefas's real "Processed bathymetry from Inner Dowsing, Race Bank and
  North Ridge cSAC" (IDRBNR CEND 11/11, same survey programme as HHW --
  its own internal lineage path confirms 2011-06, RV Cefas Endeavour --
  but a genuinely different geographic product) -- `IDRBNR-Bathy.zip`,
  confirmed 218,399,605 bytes, downloaded and checksummed once and cached
  thereafter. First, a real, useful presentation-layer bug fix: the
  background-display path's `np.ma.masked_where` left a source's raw
  float32 nodata sentinel (~-3.4e38) sitting in the masked array's own
  `.data` buffer, which matplotlib's colour normalization still touched
  and overflowed on -- fixed by replacing invalid cells with `nan` before
  masking (`_safe_masked_array`), never by suppressing the warning, with
  a regression test reproducing the exact scenario. This archive's real
  layout is genuinely different from HHW's (never assumed): 21 raster
  candidates -- 18 small ESRI Arc/INFO Binary Grid directories PLUS,
  unlike HHW, 3 large standalone ESRI ASCII Grid (`.asc`) files -- and a
  real, previously-undocumented data-integrity finding: two of the 18
  grids (`asciito_idrb3`/`idrb4`) declare CRS EPSG:4326 (degrees) while
  their actual coordinate values are obviously UTM-scale metres, an
  internally inconsistent source CRS declaration detected generically (by
  the CRS's own `is_geographic` flag, never by hard-coding those two
  grids' names) and excluded from candidacy with an explicit reason,
  never silently used. A new canonical-support PREFLIGHT
  (`run_canonical_support_preflight`) runs every one of the 19 remaining
  credible candidates through the engine's own 2000 m/1000 m-only tile
  search BEFORE any morphometry begins -- real result, reported honestly:
  every candidate fails the required 90% valid-fraction bar (best
  achieved 49.2% at 2000 m, 70.2% at 1000 m, by the largest standalone
  `.asc` file) -- closer than HHW's 55.9% at 1000 m, but still a genuine
  failure, so the ticket's hard early-stop rule correctly fires
  (`INSUFFICIENT_CONTINUOUS_SPATIAL_SUPPORT`) before any tile is selected
  or any spectral processing begins, writing only the source inventory,
  the 21-row preflight table, a support-audit figure (background coverage
  + a prominent "NO QUALIFYING TILE" banner, real percentages included),
  and validation metadata with empty (correctly-schema'd) canonical
  parquets -- exactly the ticket's own definition of a complete,
  successful negative result. Two new, generically-reusable engine
  additions exist for the canonical-pass case this real dataset didn't
  reach: `select_spatially_independent_eligible_tiles` (strict
  >=3-wavelengths eligibility PLUS a greedy non-overlap check, at most 5
  tiles, so detailed validation can never report many overlapping tiles
  as independent samples) and an explicit A-D validation-acceptance
  waterfall (`derive_canonical_real_validation_status`) requiring >=1
  canonical tile, >=1 tile meeting the wavelength condition, >=1
  successful transect, and >=3 retained bedforms -- both fully
  synthetically tested even though this real run never exercises them. A
  new cross-analog comparison table
  (`analogs/sandwave_morphometry_analog_validation_summary.parquet`)
  places HHW and IDRBNR side by side on method-validation SUPPORT only
  (never morphology values), built by re-reading HHW's already-persisted
  MAR-017A canonical (not exploratory) outputs rather than re-running its
  pipeline. 35 new tests across `test_sandwave_morphometry.py`,
  `test_sandwave_morphometry_map.py` (new), and
  `test_idr_bnr_cend1111_analog.py` (new) cover the required list (never
  below 1000 m, early-stop correctness, strict eligibility with no
  fallback, the 30 m gate, the nodata fix, the bedform-count floor,
  status derivation from real criteria, analog-only flags, HHW's
  exploratory rows never entering this cross-analog summary, and no
  forbidden risk/score term anywhere); the full offline suite (1010
  tests, 25 live/network tests correctly deselected) and repo-wide `ruff
  format`/`ruff check` pass clean. No further ticket has started.

- `MAR-017C`: the FINAL open-analog canonical real-data validation
  attempt for the sand-wave morphometry engine, against The Crown
  Estate's Marine Data Exchange record `TCE-439` -- "2014, ADUS
  DeepOcean, Greater Gabbard, Bathymetry Survey" (confirmed live, and
  deliberately distinguished from a fabricated `TCE-999999` id returning
  HTTP 404). Real acquisition mechanics turned out very different from
  the ticket's own estimate: the site's only download button bundles
  everything into ONE Azure-blob-hosted ZIP, confirmed via HEAD request
  at 8,474,296,318 bytes (~8.47 GB) -- not the ~770 MB two-file estimate.
  Rather than a full download, this module reads ONLY the remote ZIP's
  central directory via HTTP range requests (a few KB) to enumerate
  every real entry, then extracts ONLY the specific entries actually
  needed via one targeted range request per entry -- confirmed working
  by extracting the exact 501,963,720-byte `GEOTIFF.zip` entry this way.
  That extraction then revealed Section 2's own escape hatch was needed
  for real: `GEOTIFF.zip`'s 1,192 entries are 3-band uint8 RGB colour
  renders (verified via `rasterio` `ColorInterp`), not analytical
  elevation data -- a uint8 image cannot carry metre-scale depth at any
  useful precision. The ASCII package is used instead, and specifically
  its TWO properly-gridded sub-products -- `Foundations/GRIDDED
  0.25x0.25` (144 per-turbine-foundation grids) and `Corridors/GRIDDED
  0.5x0.5` (152 inter-array-cable-corridor grids), both in a non-standard
  header-less "easting northing depth" text format requiring a custom
  parser -- extracted via the same range-request technique: 328 real
  entries (144 + 152 + 32 `Concrete mattressing` rock-protection files),
  9,695,822,453 bytes. The corridor sub-product was itself a real, honest
  correction mid-ticket: this module's OWN first archive-inspection pass
  missed it entirely (assumed corridors were only ever the irregular
  point-cloud `ALL ASCII` variant), caught only when the initial 144-
  candidate preflight's per-format counts didn't reconcile against the
  full inventory -- "never assume" turned out to apply to this module's
  own reconnaissance, not only to the archive. Real result, computed
  across every one of the FULL 296 candidates (never sampled or
  assumed): the 144 foundation grids are individually too small in at
  least one dimension to even attempt a 1000 m tile (largest: `IGSUB` at
  777 m x 509 m), while several of the 152 corridor grids -- the long
  export-cable routes to the substation, e.g. `IGB04-IGSUB` at 5036 m x
  2283 m -- ARE large enough in both dimensions to attempt one, but their
  real valid-data density is still far short of 90% (best achieved across
  all 296: 53.7% at 1000 m, by `IGSUB-IGH07`) -- a corridor survey
  follows a narrow cable route, not a full-width swath, so most of its
  own bounding rectangle is genuinely unsurveyed (`INSUFFICIENT_
  CONTINUOUS_SPATIAL_SUPPORT`, the hard early-stop firing correctly
  before any spectral processing). A genuinely new problem for this
  analog (an operating wind farm, not an open-seabed survey): an
  infrastructure-context inventory (144 turbine/substation foundation
  positions -- the two real substations, `GASUB`/`IGSUB`, detected
  generically via a "SUB" substring check, never hard-coded -- plus 32
  rock/concrete-protection footprints) was derived entirely from the
  bathymetry package's own file structure, and 148 of 152 named
  inter-array cable corridors were resolved to real line segments between
  their two named foundations' centroids -- all without any second
  GIS-package download. A natural-seabed eligibility gate
  (`assess_natural_seabed_eligibility`) and an extended A-G validation
  waterfall (adding the natural-seabed criterion between spectral
  eligibility and transect success) exist and are fully synthetically
  tested even though this real, confirmed-empty canonical tile set never
  reaches them. Also fixed in passing: three shared figure-rendering
  functions (`render_method_figure`, `render_bedform_distribution_
  figure`, `render_canonical_support_audit_map`) had a latent bug --
  their footer disclaimer hard-coded "HHW CEND 11/11" regardless of which
  analog actually called them, dead code only because MAR-017B's own run
  never reached a figure-rendering branch; now a required `dataset_label`
  parameter, with all three prior call sites updated. A second, real
  layout bug found via visual inspection of the actual rendered support-
  audit figure: without an explicit view limit, matplotlib auto-scaled
  the axes to fit every far-flung turbine marker across the whole wind
  farm, shrinking the one background candidate's own real coverage down
  to an imperceptible sliver -- fixed by fixing the view to the
  background's own extent (confirmed by re-rendering: the real corridor
  survey's narrow, branching swath pattern is now clearly, unmistakably
  visible). The cross-analog summary
  (`sandwave_morphometry_analog_validation_summary.parquet`) now carries
  all three analogs plus a new `natural_seabed_eligible_count` column --
  ranked by how close each came to passing, real coverage is IDRBNR
  (70.2% at 1000 m) > HHW (55.9%) > Greater Gabbard (53.7%), all still
  well short of 90% -- and a new
  `sandwave_morphometry_engine_validation_status.json` records the final
  MAR-017-family decision: `NO_FURTHER_OPEN_ANALOG_SEARCH_PLANNED` --
  three independent official open analogs all failed the canonical
  support/eligibility protocol for three structurally different,
  honestly-documented reasons; analog hunting is closed for good, and the
  reusable engine remains implemented and synthetically verified but NOT
  canonically real-data validated. 25 new tests in
  `test_greater_gabbard_2014_analog.py` cover the required list
  (preflight-before-processing, no sub-1000 m fallback, both candidate
  classes covered, anthropogenic-tile rejection, eligibility independent
  of spectral ranking, missing-infrastructure honesty, strict wavelength
  eligibility, the 30 m gate, the natural-seabed validation floor, the
  bedform-count floor, analog-only flags, cross-analog schema purity,
  and the final closed-search decision); the full offline suite (1035
  tests, 25 live/network tests correctly deselected) and repo-wide `ruff
  format`/`ruff check` pass clean.

- `MAR-018`: the PL854 Engineering Evidence Atlas -- a map-first GIS/
  report packaging milestone, deliberately introducing NO new scientific
  model and NO fused score: every value in the new `evidence_atlas`
  package is read from an already-accepted MAR-007/010/011A/012/013/014/
  014A/014B/014C/015/016 output and, at most, renamed/joined/relabelled.
  The canonical `pl854_section_evidence.parquet` (14 rows, one per real
  hydro-pair support section) is built by merging six already-aligned
  segment tables (current/wave/combined-shear/mobility/scour/freespan-
  counts) purely on `segment_id`/`hydro_pair_id`, with every merge passed
  `validate="one_to_one"` so a silent schema drift in any upstream ticket
  would fail loudly here rather than silently corrupt the atlas -- the
  shared 14-section grid this depends on (first asserted by MAR-015) is
  thereby verified again, not just assumed. Real reconnaissance overturned
  several of the ticket's own assumptions before any code was written:
  its "MAR-016" label for regional morphology was actually MAR-007
  (MAR-016 is the separate high-res survey-access audit); its guessed
  column names `combined_tau_max_p95_lower_pa`/`upper_pa` do not exist
  anywhere -- the real min/max-across-roughness-scenarios columns
  (`tau_max_p95_sensitivity_min_pa`/`max_pa`) live only in the segments
  GeoPackage, never in the stats parquet; MAR-013's "mobility capacity"
  is a plain float from a fixed 9-point D50 ladder, never a range-string
  class like "0.5-1.0 mm"; and every regional-morphology column carries
  a unit suffix the ticket's own names omitted (`slope_500m_deg`, not
  `slope_500m`). The `highres_survey_inventory` GIS layer needed real
  footprint polygons that the accepted MAR-016 table itself does not
  carry (kept geometry-free on purpose); rather than fabricate one, it is
  rebuilt OFFLINE from the same already-cached raw BGS GeoJSON responses
  MAR-016 itself fetched (zero network, zero new science), reusing MAR-
  016's own `build_survey_footprints_gdf` helper unchanged. The atlas
  GeoPackage (7 layers: `pipeline_route`, `engineering_support_sections`,
  `observed_freespans_2018`, `historical_freespans_2012_2018`,
  `observed_psa_d50_points`, `highres_survey_inventory`,
  `chainage_reference_points`) is confirmed EPSG:32631 throughout with no
  null geometry, and the 8 real 2018 freespan events keep their true,
  variable-length (0.25-26.2 m) route-substring geometry, never widened.
  Real facts the atlas surfaces: 8 official 2018 corridor freespans
  (97.42 m/23.16 m/0.41 m) occupy only 3 of 14 hydrodynamic support
  sections; 19 exposed sections/519 m remain aggregate-only (no spatial
  locations exist); noncohesive p95 mobility capacity is 0.5-1.0 mm
  across the route; MAR-014's p95 screening class is spatially uniform
  at 0.03xD; and all 14 real MAR-016 high-resolution survey candidates
  remain `METADATA_ONLY_CUSTODIAN_REQUEST_REQUIRED` -- no verified open
  PL854-specific high-resolution bathymetric grid exists. Three real,
  unticketed bugs were found by actually looking at the rendered output
  rather than trusting the code: the primary 4-panel atlas figure's
  first 2x2 layout attempt (guessed height ratios) left large blank
  margins in every panel -- `ax.set_aspect("equal")` always shrinks a
  mismatched axes box down to the data's own aspect ratio rather than
  the reverse, confirmed by directly introspecting `ax.get_position()`
  rather than continuing to guess, then fixed by deriving each panel's
  height from its width and the real content aspect ratio (~2.6:1, route
  + background raster, padded); the section-summary table had no
  explicit column widths, so long text overflowed visibly into
  neighbouring cells -- fixed with explicit `colWidths` and a `bbox` that
  fills the axes exactly; and the provenance timeline's auto-generated
  date-range label rounded fractional years naively (`.0f` formatting
  turned a 2024.55-2026.67 span into the misleading "2025-2027") --
  fixed by requiring the caller to pass an exact, pre-formatted label
  rather than reconstructing one from rounded floats. 21 new tests in
  `test_evidence_atlas.py` cover the required list (derived, not
  hard-coded, section count; 2018 events remain corridor-level/
  unresolved-by-line and are never enlarged in GIS geometry; exposure
  stays non-spatial; evidence types never mix within one scientific
  role; no fused score/probability/risk field anywhere; MAR-012/013
  values pass through unmodified; 0.03xD stays a screening class, never
  a design requirement; the 1991-1992 morphology label and non-25m
  hydrodynamic support scale are preserved; the GeoPackage opens with
  every expected EPSG:32631 layer; the HTML report builds with the
  Python `socket` module itself blocked; MAR-017 analog data never
  reaches a section-evidence column or the `core` module's own source);
  the full offline suite (1056 tests, 25 live/network tests correctly
  deselected) and repo-wide `ruff format`/`ruff check` pass clean.

- `MAR-019`: the OrbGSS Marine POC v0.1 external-reviewer package -- a
  product-framing/presentation milestone introducing NO new geohazard
  physics, NO risk/susceptibility score, and NO scientific recomputation:
  the new `build-marine-poc-review-package` command REREADS MAR-018's
  already-persisted `pl854_section_evidence.parquet` and atlas GeoPackage
  layers directly (never calling `evidence_atlas.core`'s build functions),
  so the canonical scientific values are provably frozen. Repairs three
  real, externally-flagged presentation problems in the MAR-018 atlas: (1)
  a naive 2x2 panel layout with guessed height ratios left large blank
  margins -- root-caused by direct `ax.get_position()` introspection
  (`ax.set_aspect("equal")` always shrinks a mismatched box down to the
  data's own aspect ratio, never the reverse) rather than continued
  guessing, then fixed by deriving each panel's height from its width and
  the real content aspect ratio and enlarging the whole figure
  (16in -> 21in wide, ~30% larger boxes/typography); (2) Panel D's
  high-resolution-survey hatch visually dominated the route -- fixed by
  switching from a dense cross-hatch to a sparse, low-alpha single-
  direction hatch and redrawing the route/support-section structure on
  top, at higher contrast; (3) the title asserted an unverified physical
  flow direction (`Anglia A -> LOGGS`) -- replaced with the non-directional
  `Anglia A-LOGGS corridor` everywhere. The evidence strip gained KP-
  primary tick labels (chainage metres demoted to a light secondary axis),
  an independent visibility marker above each true-width 2018 freespan
  interval (the true width itself is never altered -- confirmed by a test
  that asserts the source GeoDataFrame's geometry and chainage columns are
  bit-identical before/after rendering), a compact categorical scour-onset
  screening band (replacing a full-height line plot of what is, in the
  real data, a spatially uniform value) explicitly labelled "screening",
  never "required"/"design" burial, and Folk-class legend descriptions
  sourced verbatim from BGS's own existing free-text field (`S` -- sand,
  `gS` -- gravelly sand, `(g)S` -- slightly gravelly sand) rather than an
  invented mapping. The engineering report gained a corrected Section 9
  (no longer conflates freespan events with the separate, aggregate-only
  exposure evidence) and a new "About the OrbGSS Marine POC" section. Two
  new artifacts frame the product itself for an external georisk reviewer:
  a 3-5-minute POC overview (measured/interpreted/derived data model,
  a non-committal future hazard-map concept table using only
  "demonstrated"/"screening prototype"/"planned" statuses, explicit
  product negatives -- OrbGSS is not a survey/geophysical/drilling
  contractor) and an external review guide organized around engineering
  relevance, data input, scientific interpretation, GIS usability,
  reporting, workflow, and misinterpretation risk, plus a data-
  authorization/privacy caveat stating PL854 public data are sufficient
  for this first review round. An offline `docs/marine_module_poc_
  architecture.md` records the future generic (Project / Ingestion / QA /
  Canonical Model / Measured / Interpreted / Derived / Map / KP / Report)
  architecture as a concept only -- explicitly the next milestone
  boundary, explicitly not implemented here. 37 new tests (21 in
  `test_evidence_atlas.py` for the atlas/report changes, 16 in
  `test_marine_poc.py` for the new POC package) cover the required list;
  the full offline suite (1072 tests, 25 live/network tests correctly
  deselected) and repo-wide `ruff format`/`ruff check` pass clean.

- `MAR-020`: the first non-PL854 OrbGSS Marine Module benchmark --
  operator-supplied high-resolution bathymetry to generic terrain
  analytics, deliberately on a DIFFERENT real project (`sheringham_shoal_
  2020`, benchmarked against the real 2020 Fugro/Equinor Sheringham Shoal
  Seabed Monitoring Survey, The Crown Estate Marine Data Exchange series
  TCE-1986) to prove the engine generalizes beyond PL854, not just a
  second PL854 feature. The one needed file
  (`G201193_20210127_SS_MBES_1m_LAT.tif`, 162,924,933 bytes,
  SHA256 `c5e3ee92...`) was range-fetched out of a ~1.01 GB combined ZIP
  (reusing MAR-017C's `RemoteZipReader` HTTP-Range pattern) after an
  explicit user confirmation, never the whole bundle. Direct `rasterio`
  inspection (never the filename) confirmed single-band float32
  EPSG:32631 at exactly 1 m, and -- contradicting the ticket's assumed
  default case -- that the raw values were ALREADY elevation-style
  (-24.397 to -3.189 m), so the canonical `bed_elevation_m` conversion
  applies zero sign flip, with both conventions preserved in the record.
  A brand-new, from-scratch generic windowed-moment terrain engine (8
  layers: slope/aspect/profile curvature/plan curvature/local relief/
  terrain std/ruggedness, over the canonical bathymetry) hit a genuine
  `MemoryError: std::bad_alloc` on the real 252,386,550-pixel raster --
  root-caused to FFT convolution over a circular footprint padding to
  large complex-valued intermediate arrays -- and was redesigned around
  an exact (not approximated) square window computed as two separable 1D
  `scipy.ndimage.correlate1d` passes per moment, memory-bounded regardless
  of raster size; re-validated against 12 synthetic cases (planar slope/
  aspect exact recovery, cardinal aspects, paraboloid curvature sign,
  nodata non-contamination, physical-window invariance across pixel
  sizes, and a checkerboard ruggedness case matching an exact closed-form
  derived for the square window to 1e-9), then confirmed end-to-end on
  the real raster with no crash (slope/aspect ~140-190s, curvature
  ~130-140s, relief/std/ruggedness ~25-60s each at two physical scales).
  Aspect and curvature (Zevenbergen & Thorne 1987) are greenfield -- no
  prior implementation existed anywhere in the codebase. Three further
  real bugs were found and fixed from actually running the full CLI
  against the real raster, not just the synthetic suite: the acquisition
  function still issued a live HEAD request on every cache hit (fixed
  with a JSON sidecar, verified via a real second invocation with
  `already_cached=True` and zero network I/O); the readiness JSON's
  `crs_present` check carried a hard-coded failure message even when it
  passed, and two other checks leaked a `numpy.bool_` into the JSON as
  the string `"True"` instead of a proper boolean (both confirmed against
  the real output and fixed); and the terrain atlas's fixed 2-column grid
  left each panel mostly blank for the real survey's narrow ~1:2.7
  portrait swath, fixed by deriving panel geometry from the content's own
  aspect ratio (mirroring the MAR-018/019 atlas layout fix). No route/KP
  view is fabricated in the absence of a supplied route
  (`NOT_APPLICABLE_NO_AUTHORITATIVE_ROUTE_SUPPLIED`, verified by a
  source-inspection test). No risk, susceptibility, freespan, or scour
  score exists anywhere in this module -- terrain analytics only. 35 new
  tests (`test_terrain_poc.py`) cover the required list, including a
  fix for one genuinely flaky test (a `socket.socket` monkeypatch that
  raced real DNS resolution, replaced with a deterministic
  `requests.Session.head` patch, confirmed clean across 6 consecutive
  full-suite runs); the full offline suite (1107 tests, 25 live/network
  tests correctly deselected) and repo-wide `ruff format`/`ruff check`
  pass clean.

- `MAR-021`: the first multi-epoch OrbGSS Marine Module benchmark -- a
  generic DEM-of-Difference / observed seabed-change engine
  (`marine_engine.change`: epoch_compatibility, alignment, common_support,
  dod, uncertainty, comparator, maps, report, contract; zero PL854/
  Sheringham hard-coding, source-inspection tested), benchmarked against
  the real 2018 vs 2020 Sheringham Shoal surveys -- observed change only,
  no future erosion/deposition prediction, no sediment-transport model,
  no risk/susceptibility/scour/freespan score, no ML. Unlike 2020's ready-
  made GeoTIFF, the 2018 package's own file browser offers only raw
  Applanix/Qinsy acquisition folders plus a 3-part ASCII XYZ export inside
  a genuinely enormous 357.74 GB combined bundle (vs 2020's already-cached
  1.01 GB) -- range-fetched only the 3 XYZ parts (~2.19 GB decompressed,
  ~287 MB compressed) via the existing `RemoteZipReader`, then rasterized
  by DIRECT grid placement (rows already fall on half-integer cell-centre
  coordinates; the source's own `Range and Interpolation.txt` states "No
  interpolation was undertaken", matched by never interpolating here
  either) -- the rasterized min/max (-24.500/-3.690 m) reproduced the
  source-documented range exactly. Two further real files were acquired
  from the ALREADY-known 2020 bundle: `MBESDIFF_20v18.tif` (an independent
  source-produced comparison product, Section 3, never used to compute
  this project's own DoD) and a candidate uncertainty grid (`MBESHSD`),
  which direct rasterio inspection showed to be a classified `uint8`
  (values 0-254, nodata=255, `RepresentationType=THEMATIC`) with no scale/
  offset/unit metadata -- correctly never used quantitatively, labelled
  `UNVERIFIED_CLASSIFIED_GRID` rather than assumed to be metres. The
  mandatory hard vertical-datum gate is satisfied by REAL source evidence,
  not a shared filename token: the 2020 Comparison Report (fetched and
  searched directly by extracting `word/document.xml` from its `.docx`
  sibling -- 13.7 MB compressed vs. the 87 MB PDF's own text-plus-figures
  bulk) explicitly states "All soundings shall be reduced to Lowest
  Astronomical Tide (LAT)" and that "The 2013, 2014, 2015, 2018 and 2020
  surveys also utilised the ... (VORF) geoid model" for LAT referencing;
  the same report supplied genuine per-epoch precision evidence (+/-0.2 m
  nominal MBES vertical accuracy; <=0.15 m observed repeatability at a
  stable 100 m^2 datum square across winter surveys including 2018/2020;
  Fugro's own ~0.3 m analyst significance threshold) -- preserved as
  measurement-accuracy context (see MAR-021A below for how this project's
  handling of it was corrected). The two independently-built grids (2020's
  official GeoTIFF; this project's own from-scratch rasterization of
  2018's raw XYZ) landed on an exact whole-pixel offset
  (`INTEGER_PIXEL_OFFSET_ALIGNMENT`, row=-10, col=3) -- pure cropping, zero
  interpolation -- with 75,918,562 common valid cells (99.3%/99.2% of
  2018/2020). Real DoD: min=-5.464, median=-0.041, p05=-0.184, p95=0.112,
  max=5.846 m over an annualized 2.09-year interval (approximate --
  survey dates are source-stated only at month granularity). Comparing
  against the official `20v18` product surfaced a striking, honestly-
  earned finding: neither the GeoTIFF's own sidecars nor the Comparison
  Report's prose states its numeric sign convention (genuinely searched,
  not assumed), so this project's comparator reports BOTH interpretations
  transparently rather than picking the better-fitting one (Section 15 is
  explicit that fit quality must never be used to resolve an undocumented
  sign) -- and under the as-is interpretation the residual NMAD is
  5.66e-7 m (float32-precision noise; i.e. functionally identical),
  while the sign-flipped interpretation gives a real NMAD of 0.163 m,
  meaning this project's independently-reprocessed DoD reproduces Fugro's
  own official comparison product to within floating-point rounding --
  strong, transparent evidence, formally still reported as
  `SIGN_UNRESOLVED_FROM_DOCUMENTATION` since no explicit statement was
  ever found. Three real bugs were found and fixed from actually running
  the full CLI against real data, not just the synthetic suite: two
  module-level evidence constants were referenced in the CLI but never
  defined on the provider module (`AttributeError`, fixed); the DoD's
  common-support crop was correctly applied to both epochs but NOT to the
  independently-loaded (full, uncropped) `20v18` comparator array before
  comparison, raising a real shape-mismatch (`(25600,9855)` vs
  `(25610,9855)`) -- fixed with a new `crop_array_to_aligned_window`
  helper, verified by a test that checks spatial correctness (the same
  real-world coordinate maps to the same value before/after cropping), not
  just matching shapes; and the primary 4-panel change map's fixed
  roughly-square layout left most of each panel blank for the real
  survey's tall/narrow (~2.6:1) footprint -- fixed by deriving panel
  geometry from the content's own aspect ratio and switching to a single
  row of 4 panels (mirroring the MAR-018/019/020 atlas layout fix
  precedent), with a follow-up pass shortening panel titles and widening
  inter-panel spacing once the narrower panels caused title collisions.
  No route/KP view is fabricated in the absence of a supplied route
  (`NOT_APPLICABLE_NO_AUTHORITATIVE_ROUTE_SUPPLIED`, verified by a
  source-inspection test). 33 new tests (`test_seabed_change_poc.py`)
  cover the required list; the full offline suite (1140 tests, 25 live/
  network tests correctly deselected) and repo-wide `ruff format`/
  `ruff check` pass clean.

- `MAR-021A`: a scientific-honesty correction to MAR-021's uncertainty
  handling -- semantics only, no change to the canonical rasters,
  alignment, common support, DoD raster, comparator processing, or
  annualized raster (all confirmed byte-identical/statistically identical
  before and after this fix). MAR-021 had computed
  `sqrt(0.2^2 + 0.2^2) = 0.283 m` from the source's own "typically less
  than +/-0.2 m" nominal MBES accuracy statement and reported the result
  as a "defensible uncertainty threshold" -- that overstated the source's
  own claim: nowhere does the source establish that +/-0.2 m is a 1-sigma
  (or any other named confidence-level) standard uncertainty, so treating
  it as an addable-in-quadrature sigma was not defensible. The threshold
  status is now always `GENERIC_DOD_UNCERTAINTY_THRESHOLD_NOT_DEMONSTRATED`
  (`change.uncertainty.derive_change_threshold` never returns a
  demonstrated generic threshold for nominal-accuracy inputs, regardless
  of their values); the same RSS arithmetic is still computed and exposed
  as `nominal_accuracy_rss_reference_m`, but explicitly labelled as NOT a
  propagated 1-sigma DoD uncertainty and NOT a canonical significance
  threshold. Fugro's own "changes <0.3 m were not considered significant"
  analyst criterion is now a clearly separate, source-specific evidence
  item (`SOURCE_SPECIFIC_ANALYST_SIGNIFICANCE_THRESHOLD`) that can never
  be repackaged as a generic OrbGSS formula. The engineering report's
  Uncertainty section now structurally separates three categories that
  MAR-021 had blended into one list: Measurement Accuracy Evidence,
  Source-Specific Analyst Threshold, and Generic Propagated Uncertainty.
  `seabed_change_poc_validation.json`'s question E
  (`question_e_defensible_uncertainty_threshold`) is now hardcoded `NO`
  with an explicit reason, rather than an indirect null-check that
  happened to evaluate the same way. 5 new tests
  (`test_seabed_change_poc.py`, S-U) cover: nominal accuracy can never be
  silently treated as sigma regardless of its magnitude; the source-
  specific analyst threshold is structurally never fed into the RSS
  arithmetic; question E is hardcoded `NO`; no thresholded change
  classification is ever created; and the DoD-computation code block is
  structurally independent of the uncertainty/threshold machinery. The
  full offline suite (1145 tests, 25 live/network tests correctly
  deselected) and repo-wide `ruff format`/`ruff check` pass clean.

- `MAR-022`: the third non-PL854 OrbGSS Marine Module benchmark --
  `SANDBED_BEDFORM_MORPHOLOGY_AND_OBSERVED_CHANGE`, a new generic
  `marine_engine.bedforms` package (interpretation classification, natural-
  vs-anthropogenic tile context, array-based transect/bedform extraction,
  independent multi-epoch crest matching) built entirely on TOP of the
  already-accepted MAR-017 morphometry engine (reused unchanged) and the
  already-accepted MAR-020/021 canonical bathymetry pipeline (re-derived
  from the same cached sources, never reacquired; the MAR-021 DoD GeoTIFF
  itself is loaded read-only, never recomputed). One new real, live
  acquisition: the TCE-1986 series' separate "Interpretation Shapefiles"
  package (confirmed via the same button-click URL-intercept technique as
  every other bundle in this project) -- a genuinely tiny (121,996-byte)
  5-shapefile set, inspected directly (never assumed from filenames) to
  find 146 "Jackup location" points, 24 named cable exposures, a dedicated
  10-feature "SandWaveCrests_PollardBank" layer, a 359-feature mixed
  `LinearFeatures` layer (270 more sand-wave-crest lines plus fishing
  gear/rock dump/rope/cable-exposure/one unclassified "Unknown_Linear_
  Feature"), and a 46-feature polygon layer of wrecks/trenching/possible
  debris -- 280 real features classified `NATURAL_BEDFORM_INTERPRETATION`,
  303 `ANTHROPOGENIC_DISTURBANCE_CONTEXT`, 2 left honestly
  `UNCLASSIFIED_INTERPRETATION_FEATURE` (an unmapped descriptor is never
  assumed either way). Canonical spatial support on this project-grade 1 m
  MBES turned out to be abundant, not marginal like MAR-017's prior open
  analogs: 48 qualifying 2000 m tiles and 277 qualifying 1000 m tiles at
  up to 100% valid fraction. A genuinely interesting, honestly-reported
  finding: the strict `tile_size/wavelength >= 3` canonical-tile-ranking
  rule found zero eligible 2000 m tiles, because the tile-level 2D spectral
  diagnostic's "dominant wavelength" repeatedly locked onto the tile size
  itself (2000 m) -- residual long-wavelength content a first-order planar
  detrend cannot fully remove over such a large real window -- and because
  this project's real anthropogenic infrastructure (cable routes, rock
  dump) is itself strongly linear, the fallback top-5-by-directional-
  concentration ranking empirically favoured disturbed tiles over natural
  ones every time (`natural_bedform_validation_status` came back
  `ANTHROPOGENIC_DISTURBANCE_PRESENT` for all 5 selected tiles, 0 natural-
  eligible). This is reported transparently (a dedicated validation-JSON
  reason string and a report caveat/limitation) rather than re-tuned until
  a nicer-looking tile appeared. None of this blocked real morphometry:
  the (unaffected, per-transect) detection recovered 276/287 real
  canonical (>=30 m) bedforms for 2018/2020 respectively (plus 28/26 small
  sub-30 m bedforms preserved for QA only, per Section 9's scale
  separation), and independent multi-epoch crest matching -- spatial
  proximity + orientation compatibility + wavelength-scale consistency +
  displacement along a reproducible local cross-crest normal axis, global
  ambiguity-aware assignment so one epoch-2020 crest can never be claimed
  by two different 2018 crests, zero DoD involvement in any matching
  decision -- produced 209 canonical matches (117 `MATCHED_HIGH_SUPPORT`,
  92 `MATCHED_WITH_LIMITATIONS`, plus 4 genuinely tied pairs correctly
  left `AMBIGUOUS_NO_CANONICAL_MATCH` rather than forced) out of 16,421
  candidate pairs considered, with real observed displacement statistics
  (absolute displacement median 6.0 m, p95 31-38 m; apparent rate median
  0.48 m/yr) explicitly labelled `OBSERVED_APPARENT_CREST_DISPLACEMENT_
  RATE` and never compared against this site's own external-only 2013-2014
  ~10 m historical migration context. Comparing the independently-detected
  2020 crests against Fugro's own mapped sand-wave-crest interpretation
  (never used to seed detection) gave a real median nearest-distance of
  ~494 m -- consistent with the 5 analyzed tiles all sitting in
  infrastructure-disturbed ground rather than where Fugro's own natural-
  bedform interpretation was concentrated, not a bug. Two real bugs were
  found and fixed from actually running the CLI against real data: the
  source-interpretation feature inventory mixed strings and numbers in one
  parquet column across different real attribute tables (a genuine
  `pyarrow.lib.ArrowTypeError`, fixed by stringifying every inventoried
  value); and the real `LinearFeatures` shapefile has at least one row
  with a missing/`None` geometry, which crashed the naive per-row azimuth
  computation used for the source comparator (fixed to skip it, covered by
  a regression test). Two further map-layout defects -- a title/subtitle
  collision on the primary morphometry figure, and the multi-epoch change
  map rendering into little more than half its own canvas for this
  benchmark's tall/narrow footprint -- were caught by actually looking at
  the rendered PNGs and fixed the same way as every prior MAR-018/019/020/
  021 instance of this defect class (explicit title `y=`/`va="top"`
  placement; deriving figure size from the real content aspect ratio).
  25 new tests (`test_bedform_morphodynamics_poc.py`) cover the required
  list; the full offline suite (1170 tests, 25 live/network tests
  correctly deselected) and repo-wide `ruff format`/`ruff check` pass
  clean.

- `MAR-022A`: a scientific-integrity correction to MAR-022's canonical
  natural-bedform selection -- MAR-022's own real result (48 qualifying
  2000 m tiles, but the tile-ranking metric never natural-context-gated,
  so its top-5 fallback landed on 5 anthropogenically disturbed tiles
  every time) was provisionally accepted but explicitly NOT accepted as a
  canonical natural-bedform validation or a defensible displacement
  claim. The canonical selection hierarchy is now strict and ordered:
  assess `NATURAL_SEABED_ELIGIBLE` vs `ANTHROPOGENIC_DISTURBANCE_PRESENT`
  vs `INFRASTRUCTURE_CONTEXT_INSUFFICIENT` for EVERY qualifying support
  tile BEFORE any spectral ranking (never after), try 2000 m first, only
  fall through to 1000 m if 2000 m yields zero natural+spectrally-
  eligible tiles, and select via `select_spatially_independent_eligible_
  tiles` only -- the non-independent `rank_and_select_top_tiles` fallback
  is now reserved exclusively for the explicitly-noncanonical diagnostic
  path. A new `canonical_band_dominant_wavelength_m` diagnostic
  (`sandwave_morphometry.analyze_tile_dual_band`, a pure additive extension of
  the MAR-017 engine -- `max_wavelength_m` defaults to `None`, preserving
  every existing caller's behaviour exactly, covered by 4 new engine-level
  regression tests) restricts the existing tile-level 2D spectral
  diagnostic to `30 m <= wavelength <= tile_size_m/3`, directly enforcing
  the already-accepted >=3-wavelengths-across-tile rule rather than
  trusting the unbounded-above "global" peak, which a real run confirmed
  DOES repeatedly lock onto the tile size itself. On the real data: the
  2000 m pass still found 0 natural-eligible tiles (0/48 -- Sheringham's
  real anthropogenic infrastructure turned out to be distributed widely
  enough that no full 2000 m canonical square avoided it), but the 1000 m
  pass found 37 natural-eligible AND canonical-band-spectrally-eligible
  tiles out of 277 real qualifying candidates, from which 5 genuinely
  spatially-independent tiles were selected -- a real, positive canonical
  result MAR-022 itself never actually tested. Multi-epoch crest matching
  now runs three independently-parameterised, pre-registered tolerance
  sets (`CONSERVATIVE`/`NOMINAL`/`PERMISSIVE` -- search radius,
  along-crest offset, orientation and wavelength-ratio tolerance,
  ambiguity margin, all bundled in one `MatchingTolerances` dataclass so
  no single value can be varied in isolation) and labels every NOMINAL
  canonical match `STABLE_ACROSS_TESTED_TOLERANCES` / `TOLERANCE_
  SENSITIVE` / `AMBIGUOUS` depending on whether the SAME counterpart also
  resolves under both alternate tolerance sets -- of 63 real nominal
  canonical matches, 45 proved stable, and ONLY those 45 feed the primary
  observed-displacement statistics (median absolute displacement 4.0 m,
  p95 17.4 m, max 29.0 m -- materially smaller and more defensible than
  MAR-022's own disqualified 6.0/31.0/54.0 m). Outputs are now split
  unambiguously: `canonical_natural_tile_spectral_morphometry_2018/2020.
  parquet` and `canonical_natural_bedform_observations_2018/2020.parquet`
  may only originate from tiles passing every canonical gate; MAR-022's
  original disturbed-tile finding is preserved, never deleted, as
  `noncanonical_disturbed_tile_diagnostics.parquet` (325 rows spanning
  both support scales) plus separate noncanonical bedform/matching
  parquet files and two visually-subordinate, explicitly-`NONCANONICAL`-
  titled diagnostic figures. Every individual bedform/crest record is now
  labelled `TRANSECT_DERIVED_BEDFORM_OBSERVATION`/`TRANSECT_DERIVED_
  CREST_OBSERVATION` -- an observation count, never implied to be a count
  of unique physical crest lines. The source-interpretation comparator now
  reports median/p95 nearest distance and the fraction of canonical
  detections within 25/50/100 m rather than a single bare distance
  number, and the real result (median 714.7 m, 14.5% within 100 m)
  correctly earns `POOR_SPATIAL_CORRESPONDENCE_WITH_SOURCE_INTERPRETATION`
  -- reported as the real, valid negative result it is, never described
  as agreement. Validation question causality is now enforced structurally
  (`_derive_bedform_validation_questions`, a pure function so it is
  directly unit-testable): E (defensible displacement) can be YES only if
  A, C, and D are ALL YES and >=1 stable match exists, F can be YES only
  if E is YES, both asserted, never merely hoped for. A real bug was found
  and fixed from actually running the CLI: canonical tiles' transect
  orientation was initially still driven by each epoch's GLOBAL (potentially
  tile-scale-locked) spectral diagnostic rather than its own independent
  canonical-band-restricted one, silently undermining the very fix this
  ticket exists to make -- corrected so every canonical tile's transects
  use its own per-epoch band-restricted crest azimuth, diagnostic/legacy
  tiles unchanged. 17 new tests (13 in `test_bedform_morphodynamics_poc.py`
  covering Section 16's required list A-L, plus 4 new
  `test_sandwave_morphometry.py` engine-level regression tests) cover the
  required list; the full offline suite (1187 tests, 25
  live/network tests correctly deselected) and repo-wide `ruff format`/
  `ruff check` pass clean. Final real answers: `IS CANONICAL NATURAL
  SANDBED MORPHOMETRY DEMONSTRATED ON REAL PROJECT-GRADE MBES?` YES;
  `IS DEFENSIBLE OBSERVED 2018-2020 CREST DISPLACEMENT DEMONSTRATED?` YES.
- **MAR-023 (generic linear-asset scour susceptibility screening POC,
  `src/marine_engine/scour/susceptibility.py`/`susceptibility_map.py`/
  `observed_evidence.py`/`observed_evidence_map.py`/`contract.py`/
  `poc_report.py`, `providers/bathymetry/sheringham_shoal_2024.py`,
  `build-scour-susceptibility-poc`).** Two scientifically separate tracks,
  never conflated: Track A screens the operator's ACTUAL pipeline embedment
  against the accepted MAR-014 Marini et al. (2024) engine's own tested
  critical-embedment screening class -- reused completely unchanged, never
  re-derived -- via `embedment_protection_margin_p95_e_over_D` (actual
  minus critical p95) and a descriptive `scour_onset_screening_exceedance_
  fraction` (explicitly NOT a probability, NOT a failure probability, NOT a
  return-period metric); Track B ingests real observed scour evidence from
  a brand-new 2024 Crown Estate Marine Data Exchange survey (TCE-3974, "2024
  XOCEAN Sheringham Shoal Seabed Monitoring Survey") and is never used to
  validate Track A's pipeline physics, since the survey covers monopiles,
  cables, and protection, not a pipeline. PL854 genuinely has no continuous
  observed embedment profile, so its site-specific table (14 sections) is
  honestly `SITE_SPECIFIC_SCOUR_SUSCEPTIBILITY_NOT_AVAILABLE_NO_EMBEDMENT_
  PROFILE` throughout -- Track A is instead demonstrated via a PL854
  tested-embedment SCENARIO envelope (70 section x scenario rows, the five
  already-tested ratios treated in turn as a hypothetical actual value,
  never a real profile): at zero embedment the real forcing record exceeds
  the tested critical class on 20.9-50.8% of valid timesteps per section
  (median 43.8%), collapsing to 0-1.5% (median ~0.2%) at the 0.03D
  scenario, consistent with a route-wide `critical_embedment_ratio_p95` of
  exactly 0.03D throughout -- a real, internally-consistent, honestly
  reported result. Real inspection of the 2024 interpretation package (a
  single 746-feature `Targets` shapefile, 140,928 bytes, downloaded live
  via the same button-click-intercept technique established in MAR-022,
  then cached) found NO literal "scour" feature class despite the survey's
  own stated integrity-relevant purpose -- classified honestly into
  `SEABED_OBJECT_CONTEXT` (340), `SOURCE_INTERPRETED_EXPOSURE_EVIDENCE`
  (152, kept separate from scour evidence since the source never attributes
  causality), `ASSET_INFRASTRUCTURE_CONTEXT` (151), and
  `ANTHROPOGENIC_DISTURBANCE_CONTEXT` (103), with source-stated asset
  association preserved verbatim (390/346/9/1 across two IAC/WTG corridors,
  the export cable route, and one unassociated feature), yielding
  `OBSERVED_SCOUR_MORPHOMETRY_NOT_AVAILABLE_FROM_SOURCE_PACKAGE` rather
  than measuring a different feature class and calling it scour morphometry.
  The cached MAR-020/021 2020 bathymetry raster is reused unchanged as
  background map context (confirmed via a direct bounding-box check that
  all 746 real 2024 targets fall inside it) rather than re-acquiring a
  second, comparably large MBES surface -- zero new bathymetry bytes
  downloaded, satisfying the ticket's "minimum acquisition" requirement
  exactly. A real, subtle bug was found via visual inspection of the
  rendered scenario-envelope figure (the zero-embedment scenario's line was
  invisible, hidden under an incorrectly identical 0.03D line): `actual_
  embedment_m = scenario_ratio * diameter_m` followed by `actual_embedment_
  ratio = actual_embedment_m / diameter_m` does not always round-trip to
  the exact same float (confirmed: `0.03 * 0.3048 / 0.3048 ==
  0.029999999999999995`), which without a tolerance made the 0.03D
  scenario spuriously "exceed" real timesteps whose required embedment was
  the identical tested class -- fixed with a `1e-9` tolerance mirroring
  `scour_onset.py`'s own established `compute_embedment_monotonicity_
  violations` convention, with two new regression tests reproducing the
  exact real failure mode. A second real design issue was caught before
  the CLI ever ran: `bedforms.interpretation.classify_features` validates
  its caller-supplied category mapping against a frozenset hardcoded to
  the THREE bedform-specific category strings, so it could never actually
  accept this module's six scour-specific categories -- fixed by giving
  `observed_evidence.py` its own small, structurally-identical local
  `classify_scour_features`/`extract_scour_category` (still reusing the
  genuinely dataset-agnostic `build_feature_inventory` directly). Two
  figure title/subtitle collisions were found and fixed following the same
  established MAR-018/019/020/022 pattern (pinned suptitle y, `va="top"`
  anchored subtitles, widened `tight_layout` headroom). 31 new tests cover
  actual-vs-critical separation, negative/positive margin classification,
  valid-timestep-only exceedance computation, the missing-profile block,
  PL854 scenario-only structural checks, Marini domain-envelope
  preservation, absence of any scour-depth/risk-score field, Track A/B
  non-coupling, asset-physics-mismatch disclosure, a synthetic operator
  route producing a real rendered map, absence of hard-coded PL854/
  Sheringham coordinates in the generic engine, and an offline-cache-hit
  acquisition test that raises if `requests.get` is ever called on a cache
  hit; the full offline suite (1218 tests, 25 live/network tests correctly
  deselected) and repo-wide `ruff format`/`ruff check` pass clean. Final
  real answers: `IS GENERIC OPERATOR-SUPPLIED PIPELINE SCOUR-ONSET
  SUSCEPTIBILITY SCREENING DEMONSTRATED?` YES; `IS A SITE-SPECIFIC PL854
  SCOUR SUSCEPTIBILITY MAP DEFENSIBLE WITH THE CURRENT DATA?` NO (expected
  -- no actual embedment profile exists).
- **MAR-023A (observed scour evidence validation semantics repair, Track A
  untouched).** A precise correction, not a rewrite: MAR-023's validation
  Question F ("was real source-interpreted SCOUR evidence ingested from an
  operator survey?") was wrongly derived from `not evidence_gdf.empty` --
  true for the real Sheringham 2024 run's 746 classified features even
  though every single one of them is exposure/infrastructure/disturbance/
  seabed-object context, never a literal scour observation. Fixed to
  derive strictly from `explicit_scour_feature_count > 0`
  (`observed_evidence.summarize_observed_evidence` already computed this
  count correctly; only the CLI's validation call was wrong), so the real
  run now correctly answers Question F **NO**, with reason
  `NO_EXPLICIT_SOURCE_INTERPRETED_SCOUR_FEATURE_CLASS_PRESENT`. Two
  previously-conflated facts are now recorded as fully independent
  booleans everywhere the ticket required (validation JSON, HTML report,
  CLI final report): `operator_interpretation_package_ingested` (YES --
  real data genuinely arrived and was classified) and
  `explicit_source_interpreted_scour_evidence_present` (NO). The GIS
  geopackage's primary layer (`observed_scour_evidence.gpkg`, filename
  kept for compatibility) is renamed from the misleading
  `observed_scour_evidence` to `source_interpreted_integrity_context`;
  code to write a separate `explicit_observed_scour_evidence` layer exists
  for when a source genuinely does contain literal scour features, but is
  correctly absent for the real Sheringham run (zero such features). The
  evidence map's title changed to "Sheringham Shoal 2024 -- Source-
  Interpreted Scour / Integrity Context" with a prominent red banner --
  "NO EXPLICIT SOURCE-INTERPRETED SCOUR FEATURES WERE PRESENT" -- rendered
  with fixed figure headroom (`fig.subplots_adjust(top=0.83)`) so the
  banner can never collide with the title/subtitle regardless of whether
  it renders, avoiding a repeat of the title-collision bug class already
  fixed once in this same ticket. `susceptibility.py`/`susceptibility_map.py`
  (Track A: MAR-014 physics, the PL854 scenario envelope, critical-
  embedment percentiles, the actual-vs-critical margin, exceedance
  fractions, the floating-point tolerance fix) were not modified at all --
  confirmed both by `git status` and by a test asserting the module source
  carries no MAR-023A reference -- and a real offline rerun reproduced
  bit-identical PL854 numbers (0D exceedance min/median/max
  0.20907866897480284/0.43758415079823043/0.5079823042892864, unchanged to
  the last digit) against the already-cached MDE package
  (`already_cached=True`, zero new network access). 8 new tests cover: a
  non-empty package with zero explicit scour features producing Question F
  = NO with the correct reason; an explicit scour count > 0 producing
  Question F = YES; operator-ingestion status remaining YES independently
  of explicit-scour presence; exposure never counting as scour; the GIS
  layer/map title no longer implying every row is a scour observation; and
  an exact numeric golden-value regression proving Track A's scenario
  envelope arithmetic is untouched. Full offline suite: 1226 tests, 25
  live/network tests correctly deselected; repo-wide `ruff format`/
  `ruff check` pass clean.
- **MAR-024 (generic linear-asset burial/exposure state POC,
  `src/marine_engine/burial/` -- `semantics.py`/`readiness.py`/`route.py`/
  `profile.py`/`margin.py`/`exposure_screening.py`/`maps.py`/`contract.py`/
  `report.py`, `providers/barrow_2016.py`, `configs/barrow_2016.yaml`,
  `build-burial-exposure-poc`).** The first POC built around a REAL
  measured depth-of-burial profile and the first study that is neither
  PL854 nor Sheringham Shoal: the 2016 Deep BV Barrow Offshore Wind Farm
  export cable geophysical depth-of-burial survey (The Crown Estate Marine
  Data Exchange, TCE-48). Three concepts kept separate throughout:
  observed/measured burial state, source-interpreted exposure evidence,
  and future exposure susceptibility -- the last is only ever produced
  given a defensible, explicitly-typed seabed-lowering input
  (`OBSERVED_MULTI_EPOCH_SEABED_LOWERING` or
  `OPERATOR_DEFINED_LOWERING_SCENARIO`; a bare float is never accepted).
  Minimal real acquisition: only the Depth of Burial Listing (a real
  20,946-record Excel workbook, 2,756,040 bytes -- the package's own
  "Listing" folder name turned out to mean `.xlsx`, not plain text,
  confirmed by inspection rather than assumed, requiring a new `openpyxl`
  dependency) and the Route Position List Files (a real single-LineString
  26,233.4 m route plus 5,252 real per-station KP/Easting/Northing points,
  216,571 bytes) -- neither Multibeam Bathymetry, Side-scan Sonar,
  Sub-bottom Profiler, Cable Tracker Survey, nor Seabed Features were
  downloaded, since the DoB listing itself already carries a real,
  explicit exposure flag. The real source's own MEDIN lineage statement
  says "depth of burial (DoB or 'z')" as a general survey-campaign
  objective, but before trusting that label this ticket's Section 4 hard
  gate demanded real evidence: the real `Z` column ranges -28.56 to 2.48 m
  (mean -14.91 m) across all 20,946 records -- far too large in magnitude
  for a literal cable-burial depth -- and, decisively, the 1,288 real rows
  the source itself flags `"Exposure"` in a `Storage Db` column show a
  statistically SIMILAR `Z` distribution (mean -13.67 m) to the
  non-exposure rows (mean -14.99 m) rather than clustering near 0 m as a
  true burial-depth convention would require. This is real, direct,
  falsifying evidence against reading `Z` as burial depth, so the ticket's
  own escape hatch is used honestly: `SOURCE_BURIAL_REFERENCE_UNRESOLVED`,
  recorded with its full statistical justification in
  `source_burial_semantics.json`, never silently relabelled as top-of-
  cable burial. Because the reference is unresolved, current burial state
  for all 19,658 non-exposure-flagged records is honestly
  `MEASURED_REFERENCE_REQUIRES_REVIEW` (never inferred as buried/exposed
  from the number's sign) -- the remaining 1,288 records are
  `SOURCE_INTERPRETED_EXPOSED`, reachable ONLY via the source's own
  explicit `"Exposure"` flag, structurally never from `Z`'s value.
  Readiness is real and honest too: `READY_WITH_LIMITATIONS`, correctly
  flagging a real KP-unit mismatch between the two source packages (DoB KP
  in metres, RPL KP in kilometres -- converted explicitly, never assumed
  equal), the unresolved burial reference, and 96.4% route coverage (a
  real, un-interpolated survey gap). The canonical route's own
  start-to-end direction against increasing source KP was empirically
  VERIFIED (not assumed) by comparing real route endpoints against the
  real min/max-KP route-position points, confirmed to match. `Section 16`'s
  real-run rule was honoured exactly: no seabed-lowering magnitude was
  invented for Barrow, so the real screening result is
  `NO_DEFENSIBLE_SEABED_LOWERING_INPUT`, and question G ("is a real Barrow
  future exposure susceptibility result defensible?") correctly and
  structurally (via an asserted pure function, mirroring MAR-022A/023's
  `_derive_..._validation_questions` pattern) always answers NO -- while
  the generic engine itself is proven correct via Section 17's exact
  synthetic demo (1.0 m cover - 0.4 m lowering = 0.6 m remaining, positive;
  0.2 m - 0.4 m = -0.2 m, zero-or-negative). Unlike MAR-023's session, no
  real bug was found by running the CLI against real data -- the semantic
  hard-gate investigation (Section 4) was done evidence-first, before any
  code was written, and two real defects (a variable-scoping bug that
  would have crashed on an empty profile, and a redundant geometry
  reconstruction call) were instead caught by direct code review before
  the first real run, which then succeeded cleanly both live
  (`already_cached=False`) and fully offline on rerun
  (`already_cached=True` for both packages). GIS output carries five real
  layers (`asset_route` 1, `burial_measurements` 20,946,
  `source_interpreted_exposure` 1,288, `survey_coverage` 1, `qa_flags`
  2,171 -- the last from real per-record `Data Quality`/`Uncertainty`
  outlier flags). 33 new tests cover reference-must-resolve-before-
  exposure-inference, top-of-asset-vs-centreline distinctness, missing-KP
  never fabricating a route position, duplicate-KP flagging, no
  interpolation over profile gaps, measured-state-vs-exposure-evidence
  independence, trench-scar-never-becomes-exposure, actual-vs-target
  burial distinctness, null margin with no target, positive margin never
  SAFE, the exact Section 17 synthetic screening numbers, no lowering
  input blocking future susceptibility, absence of any probability/risk-
  score/free-span field, absence of hard-coded Barrow/PL854 coordinates in
  the generic engine, and offline-cache-hit acquisition tests for both
  real packages; the full offline suite (1259 tests, 25 live/network tests
  correctly deselected) and repo-wide `ruff format`/`ruff check` pass
  clean. Final real answers: `IS GENERIC OPERATOR-SUPPLIED DEPTH-OF-
  BURIAL -> BURIAL / EXPOSURE STATE ANALYTICS DEMONSTRATED?` YES; `IS
  FUTURE BARROW EXPOSURE SUSCEPTIBILITY DEMONSTRATED FROM THE CURRENT REAL
  DATA?` NO (expected -- no defensible seabed-lowering input exists for
  this route/epoch).
- **MAR-024A (canonical burial-cover semantics repair, real Barrow
  conclusion untouched).** A precise correction to the generic engine, not
  a rewrite: MAR-024's `burial_sign_convention` was recorded but never
  actually used in classification (Problem A), and `TOP_OF_ASSET_BURIAL`
  vs. `CENTRELINE_BURIAL` were interpreted identically (Problem B) --
  neither is acceptable for generic operator input, where a source's own
  "depth of burial" number is meaningless until both its sign and its
  reference point are resolved. Fixed with a new `burial/cover.py`
  module and an explicit sign vocabulary
  (`POSITIVE_VALUE_MEANS_DEEPER_BURIAL` / `NEGATIVE_VALUE_MEANS_DEEPER_BURIAL`
  / `SIGN_CONVENTION_UNRESOLVED` -- free text is no longer accepted for
  physical classification, though it may still be preserved separately as
  `source_sign_convention_text`), normalizing a raw value into
  `canonical_reference_burial_depth_m` exactly once
  (`normalize_canonical_reference_burial_depth_m`), then into
  `cover_above_asset_m` exactly once (`compute_cover_above_asset_m`) --
  `TOP_OF_ASSET_BURIAL` passes through unchanged, `CENTRELINE_BURIAL`/
  `OTHER_SOURCE_SPECIFIC_REFERENCE` both require an explicit, never-
  assumed `reference_to_asset_top_offset_m` (never a hard-coded diameter
  / 2, since not every linear asset is circular). `cover_above_asset_m`
  is now the ONLY numeric quantity permitted to drive measured burial-
  state classification: `classify_current_burial_state` was restructured
  to accept `cover_above_asset_m` plus two booleans and structurally
  cannot see a raw source value at all, closing off any channel for a raw
  magnitude/sign to bypass normalization. Measured-state logic is now
  `cover > tolerance` -> `MEASURED_BURIED`, `|cover| <= tolerance` ->
  `MEASURED_AT_SEABED_LEVEL`, `cover < -tolerance` -> the new
  `MEASURED_ABOVE_SEABED` (added to the vocabulary), with
  `SOURCE_INTERPRETED_EXPOSED` still reachable ONLY via the source's own
  explicit flag and still checked ahead of the cover-based states, so the
  two concepts never collapse into one. The profile column carrying this
  result is renamed `current_burial_state` -> `measured_burial_state`
  (and the frozenset `CURRENT_BURIAL_STATES` ->
  `MEASURED_BURIAL_STATES`) to state its semantics explicitly rather than
  hide the distinction, per the ticket's own preference for the canonical
  schema. `screen_exposure_susceptibility` now consumes
  `cover_above_asset_m`, never a raw source burial value; its two guard
  checks were also reordered (missing-lowering-input checked before
  missing-cover) so that "no defensible lowering input" -- the single
  actionable blocker when neither exists -- is reported ahead of
  "insufficient burial input", which is now reserved for a real lowering
  scenario with no cover to apply it to. This reordering was required to
  keep the real Barrow result honest: cover is now genuinely `None` for
  Barrow (never the raw `Z` median standing in for it, as MAR-024's
  original wiring did), and the ticket required
  `NO_DEFENSIBLE_SEABED_LOWERING_INPUT` to remain the preserved real
  screening result rather than regressing to `INSUFFICIENT_BURIAL_INPUT`.
  The real Barrow conclusion is completely unchanged end to end, verified
  by a real offline rerun: `SOURCE_BURIAL_REFERENCE_UNRESOLVED`, the same
  19,658 `MEASURED_REFERENCE_REQUIRES_REVIEW` / 1,288
  `SOURCE_INTERPRETED_EXPOSED` split, the same `READY_WITH_LIMITATIONS`
  readiness and all eight validation answers unchanged -- with the new
  fields now visibly demonstrating the valid unresolved-reference path:
  `canonical_reference_burial_depth_available_count: 0` and
  `cover_above_asset_available_count: 0` for all 20,946 real records,
  since Barrow's sign convention is honestly asserted
  `SIGN_CONVENTION_UNRESOLVED` too (the same statistical evidence that
  makes the reference point unresolvable -- exposure-flagged and
  non-flagged `Z` values are statistically similar rather than clustering
  near 0 m -- means no sign convention can be honestly asserted for it
  either). `contract.py`'s input contract and `report.py`'s HTML report
  were both updated to name and distinguish all five concepts (raw source
  measurement, canonical reference burial depth, top-of-asset cover,
  source-interpreted exposure, future exposure screening) per the ticket.
  20 new tests cover both sign directions, an unresolved sign yielding no
  canonical depth regardless of the raw value's own magnitude,
  top-of-asset vs. centreline producing physically different cover from
  the same input, a missing centreline offset blocking classification,
  all three cover-based states plus exposure staying independent of them,
  the screening function's new signature and reordered precedence, and a
  synthetic analog of Barrow's exact real split (unresolved reference +
  unresolved sign together); the full offline suite (1279 tests, up from
  1259) and repo-wide `ruff format`/`ruff check` pass clean.
- **MAR-025 (generic pipeline free-span geometry & support-loss
  susceptibility screening POC).** Three tracks kept structurally
  separate throughout: measured/observed free-span geometry, generic
  support-loss susceptibility screening, and source-reported/observed
  free-span evidence -- explicitly excludes structural free-span
  integrity assessment (`STRUCTURAL_FREE_SPAN_ASSESSMENT_NOT_PERFORMED`;
  no DNV-RP-F105, VIV, fatigue, ULS/FLS, or failure probability anywhere).
  Because no verified open project-grade pipeline vertical-profile +
  seabed support-profile dataset exists in this repo, a new
  `src/marine_engine/freespan/` package's full generic engine (pipe
  vertical reference normalization, canonical `pipe_underside_clearance_m
  = pipe_bottom_elevation_m - seabed_support_elevation_m`, measured
  support-state classification, free-span interval extraction with gap
  governance, and support-loss scenario screening) is exercised end to
  end through an explicitly synthetic, analytically-known engineering
  validation case (`SYNTHETIC_EXACT_ENGINE_VALIDATION`, never presented
  as field validation): 20 hand-authored samples along a straight
  650 m route recover exactly the 3 expected measured free-span
  intervals (lengths 100/10/10 m, max clearance 0.6 m, the two 10 m
  spans deliberately gap-split from what would otherwise misleadingly
  look like one continuous span across a 60 m unsurveyed gap exceeding
  the 50 m governance threshold), and a 0.30 m operator-defined lowering
  scenario recovers exactly 1 new support-loss span and extends exactly
  1 existing span -- every number verified by direct hand calculation
  before ever running the code, then confirmed identical via real
  execution. A source-interpreted-free-span sample is deliberately placed
  where its own clearance would otherwise numerically qualify as
  unsupported, proving the explicit source flag overrides geometric
  inference and never leaks into either the measured-interval or
  scenario-interval extraction. Real authoritative evidence is ingested
  on two independent tracks, never used to derive or validate the
  screening physics: the full NSTA UKCS-wide pipeline freespan registry
  (both ArcGIS Feature Services, cached/acquired only when absent --
  real counts 978 total records, 953 current + 25 removed, 212 unique
  pipelines, 2 duplicated feature IDs, all real and none fabricated or
  smoothed), reusing the accepted provider's query/parse primitives but
  built as a new unfiltered "1=1" acquisition since the existing
  PL854/PL855-scoped registry function is real-and-correctly zero for
  those two lines (MAR-014C's own investigated finding, not a gap this
  ticket works around); and the accepted PL854 Table B.1 2018 observed
  freespan evidence, reused byte-for-byte via
  `evidence_atlas_core.build_observed_freespans_2018_layer` and
  `freespan_evidence_map.render_2018_freespan_evidence_map` with only a
  new title -- the exact accepted 8 events / 97.42 m / 23.16 m / 0.41 m
  numbers never recomputed. PL854 itself still has no measured continuous
  pipe vertical profile or embedment/support profile, so
  `PL854_SITE_SPECIFIC_FREE_SPAN_SUSCEPTIBILITY_NOT_AVAILABLE` is
  reported explicitly rather than a fabricated site-specific map. The
  real NSTA UKCS map uses a documented, non-risk-implying
  example-selection rule (the pipeline with the most registry records --
  real result `PL1840A`, 59 records) since the full 222+-pipeline
  registry is unreadable as one detailed map. 56 new tests cover every
  Section 32 proof point (centreline/top references requiring an
  explicit offset, positive-clearance-means-unsupported semantics, no
  threshold meaning no categorical classification, the operator-QC-
  threshold vs. combined-1-sigma-uncertainty distinction staying
  separate, the exact synthetic span/clearance/new-span/extended-span
  recovery, the gap-splits-one-span-into-two proof, all three geometric
  states plus source-interpreted independence, a structural signature
  check proving the scenario-clearance function has no pipe-elevation
  parameter at all (so pipe motion cannot be silently introduced), a
  real cache-hit-never-touches-network acquisition test, and absence
  checks for VIV/fatigue/ULS/FLS/risk-score/failure-probability
  vocabulary and any hard-coded PL854/NSTA identity in the generic
  engine); the full offline suite (1335 tests, up from 1279) and
  repo-wide `ruff format`/`ruff check` pass clean. Final real answers:
  `IS GENERIC OPERATOR-SUPPLIED PIPE/SEABED PROFILE -> FREE-SPAN GEOMETRY
  ANALYTICS DEMONSTRATED?` YES; `IS GENERIC SUPPORT-LOSS SUSCEPTIBILITY
  SCREENING DEMONSTRATED?` YES; `IS SITE-SPECIFIC PL854 FREE-SPAN
  SUSCEPTIBILITY DEFENSIBLE?` NO (expected -- PL854 lacks a measured
  pipe/seabed profile). No further ticket has started.
