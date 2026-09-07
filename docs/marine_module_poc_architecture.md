# Marine Module POC Architecture (Concept)

Status: **concept / documentation only**. No implementation exists yet for
generic operator-supplied project data. This document is the starting point
for the next implementation ticket, not a description of code already in
this repository.

The PL854 work completed so far (MAR-002 through MAR-018) demonstrates every
stage of this architecture end-to-end for exactly one fixed, public-data
project. Generalizing it to arbitrary operator-supplied projects is the
next milestone, described here at the logical/conceptual level only.

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

## 3. Data QA / Readiness

Before any analysis, ingested data must pass integrity and readiness
checks: CRS and vertical datum present and consistent, coordinate scale
plausible, nodata/sentinel values handled, survey epoch recorded, spatial
coverage relative to the route understood. This mirrors the data-integrity
discipline already used throughout the PL854 work (e.g. MAR-005/006/016),
generalized to arbitrary operator-supplied files rather than one fixed set
of public sources.

## 4. Canonical Project Model

A standardized, project-agnostic internal representation that every
downstream geohazard engine consumes: canonical route/chainage geometry,
a common working CRS, a shared section/support-grid concept, and typed
slots for each data category (measured / interpreted / derived, see below).
PL854's own canonical pipeline/AOI/chainage model (MAR-002/003/004) is the
existing single-project prototype of this idea.

## 5. Measured Data

Direct physical observations, supplied by the operator or their surveyors.
Examples: MBES bathymetry, CPT, boreholes, grab samples, current
observations, sidescan sonar, sub-bottom/HR seismic. The canonical project
model records what was measured, when, and by whom -- it does not alter or
reinterpret the measurement itself.

## 6. Interpreted Data

Expert or contractor interpretation layered on top of measured data.
Examples: shallow gas polygons, faults, buried channels, boulders, seabed
feature interpretation. This category is explicitly allowed to remain
operator- or consultant-supplied; the Marine Module does not require every
interpretation to be reproduced in software.

## 7. Derived Geohazard Layers

Software-computed analytics built from measured and/or interpreted data via
domain-specific geohazard engines. Examples already demonstrated for PL854:
regional terrain morphology (MAR-007), combined wave-current bed shear
(MAR-012), noncohesive sediment mobility capacity (MAR-013), scour-onset
screening (MAR-014). Examples not yet demonstrated: free-span
susceptibility, slope-instability screening, and a generalized route-
constraints layer. Each engine remains scientifically distinct -- this
architecture does not fuse engines into a single score.

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

---

## Next milestone boundary

The likely next technical milestone is:

**GENERIC OPERATOR-SUPPLIED PROJECT DATA INGESTION + READINESS POC**

This document exists to support external review of the product concept
first; the ingestion/readiness implementation itself is explicitly out of
scope until that review has happened (see MAR-019).
