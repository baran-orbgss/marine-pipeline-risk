"""Generic local operator-project ingestion and asset readiness (MAR-026).

Sits ABOVE raw operator files and BELOW the independent scientific geohazard engines:

    operator files -> project manifest -> asset registration + provenance ->
    asset integrity/readiness -> canonical project inventory -> existing geohazard engines

This package performs NO hazard-specific science. It never recomputes or alters the
scientific results of `terrain`, `change`, `morphology`, `scour`, `burial`, or `freespan` --
it registers operator-supplied files and DELEGATES readiness assessment to the existing
accepted readiness modules (`terrain.readiness`, `burial.readiness`) wherever those apply.

Four asset evidence roles are orthogonal and never interchangeable (mirrors the MAR-025A
geometry-vs-source-interpretation principle, generalized project-wide):

    PROJECT_GEOMETRY    -- the operator-supplied asset route/corridor geometry itself
    MEASURED            -- direct physical observation (MBES, burial survey, CPT, ...)
    SOURCE_INTERPRETED  -- contractor/expert interpretation layered on measured data
    DERIVED             -- software-computed output of a geohazard engine

This package makes no universal "project is ready for marine hazard analysis" claim --
see `project.registry`'s explicit disclaimer constant.
"""
