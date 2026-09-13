"""Marine Engine Engineering Workbench (UI-001).

Internal, localhost-only Streamlit workbench that makes the current `marine_engine` capabilities
visible and testable WITHOUT changing any scientific implementation. The engine remains the
computation authority: this package only reads existing local outputs and metadata, discovers and
runs registered pytest targets through argument-array subprocess calls, and presents a capability
maturity registry.

UI-003 replaced the primary product surface with a generic GIS workspace (`api/`, `web/`); this
Streamlit app is now reachable only as the internal Engineering Workbench, with no guided-story
mode. See `README.md` ("Guided View (UI-002)", "Marine GIS Workspace (UI-003)") for that history.

Maturity vocabulary displayed here describes SOFTWARE / SCIENTIFIC CAPABILITY MATURITY. It is never
hazard severity, risk level, engineering acceptance or a safe/unsafe classification.

Modules:

* `capability_registry` -- explicit presentation metadata (hazard capabilities, supporting
  capabilities, project/demo contexts). Pure Python; no Streamlit import.
* `test_runner`        -- allowlisted pytest discovery (ast) and safe execution (no shell).
* `output_inspector`   -- metadata-first inspection of local JSON / Parquet / GPKG / GeoTIFF / PNG.
* `repo_state`         -- read-only git facts (branch, HEAD, dirty/clean) via argument arrays.
* `components`         -- Streamlit rendering helpers (imports Streamlit).
* `app`                -- the Streamlit entry point (`uv run streamlit run ui/app.py`).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
