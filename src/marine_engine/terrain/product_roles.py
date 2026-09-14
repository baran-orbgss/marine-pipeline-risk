"""Generic terrain product role vocabulary (MAR-034).

The canonical bed-elevation role/layer names are NOT redefined here: they already exist as
`slope_stability.contract.SOURCE_TERRAIN_ROLE_REQUIRED` /
`SOURCE_TERRAIN_LAYER_REQUIRED` (MAR-031), the accepted embedded-tag identity a canonical
terrain raster carries. MAR-034's orchestration adapter imports those directly rather than
declaring a competing name (Section 11: reuse existing terminology).

`TERRAIN_DERIVATIVE_PRODUCT` is new: the one role a generated terrain-derivative suite (slope,
aspect, curvature, relief, ruggedness) is tagged with so a downstream capability can depend on
"terrain derivatives exist" without knowing which individual layers were produced.
"""

from __future__ import annotations

TERRAIN_DERIVATIVE_PRODUCT = "TERRAIN_DERIVATIVE_PRODUCT"
