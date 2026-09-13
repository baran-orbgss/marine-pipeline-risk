"""UI-003 local API adapter for the Marine GIS Workspace.

Bounded FastAPI layer between the `web/` frontend and the scientific engine
(`src/marine_engine/**`, never modified here). This package never computes, thresholds, or
reinterprets a scientific result -- it reads/calls accepted engine entry points and existing
generated outputs, and adds presentation-only metadata (palettes, legends, z-order, support-type
labels) needed to render an honest map.
"""

from __future__ import annotations
