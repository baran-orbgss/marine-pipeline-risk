"""Raster tile serving via rio-tiler.

`layer_id` resolves to an absolute path SERVER-SIDE ONLY (via the layer catalog) -- a client never
supplies a path, which is the concrete security property that makes this narrower than mounting a
general-purpose tile server. Colormap and value domain always come from the layer's own
`DisplaySpec.palette`, never from client query parameters, so a legend can never disagree with
what is actually drawn. Reprojection to Web Mercator happens on the fly inside `rio_tiler.io.Reader`
(a `WarpedVRT` under the hood) regardless of the source's native CRS; nothing is ever written back
to the source file, and the on-disk cache below is purely a rendered-tile cache, never a rewrite of
the canonical raster.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader

from api import settings
from api.errors import TileOutOfBoundsError
from api.models import LayerDescriptor, PaletteSpec

settings.ensure_runtime_dirs()


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


def _colormap_from_palette(palette: PaletteSpec) -> dict[int, tuple[int, int, int, int]]:
    stops = sorted(palette.stops, key=lambda s: s.value)
    if not stops:
        return dict.fromkeys(range(256), (128, 128, 128, 200))
    lo, hi = palette.domain if palette.domain else (stops[0].value, stops[-1].value)
    span = (hi - lo) or 1.0
    positions = [max(0, min(255, round((s.value - lo) / span * 255))) for s in stops]
    colors = [_hex_to_rgb(s.color) for s in stops]

    colormap: dict[int, tuple[int, int, int, int]] = {}
    for i in range(256):
        if i <= positions[0]:
            r, g, b = colors[0]
        elif i >= positions[-1]:
            r, g, b = colors[-1]
        else:
            j = next(idx for idx, p in enumerate(positions) if p >= i)
            p0, p1 = positions[j - 1], positions[j]
            c0, c1 = colors[j - 1], colors[j]
            t = (i - p0) / ((p1 - p0) or 1)
            r = round(c0[0] + (c1[0] - c0[0]) * t)
            g = round(c0[1] + (c1[1] - c0[1]) * t)
            b = round(c0[2] + (c1[2] - c0[2]) * t)
        colormap[i] = (r, g, b, 255)
    return colormap


def _cache_path(path: Path, z: int, x: int, y: int, palette: PaletteSpec) -> Path:
    stat = path.stat()
    digest_input = (
        f"{path}|{stat.st_mtime_ns}|{stat.st_size}|{z}|{x}|{y}|{palette.model_dump_json()}"
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return settings.TILE_CACHE_DIR / f"{digest}.png"


def render_tile(layer: LayerDescriptor, absolute_path: Path, z: int, x: int, y: int) -> bytes:
    cache_path = _cache_path(absolute_path, z, x, y, layer.display.palette)
    if cache_path.is_file():
        return cache_path.read_bytes()

    domain = layer.display.palette.domain or (0.0, 1.0)
    colormap = _colormap_from_palette(layer.display.palette)
    try:
        with Reader(str(absolute_path)) as src:
            image = src.tile(x, y, z, indexes=1)
    except TileOutsideBounds as exc:
        raise TileOutOfBoundsError(f"tile {z}/{x}/{y} is outside layer bounds") from exc

    image.rescale(in_range=((domain[0], domain[1]),))
    content = image.render(img_format="PNG", colormap=colormap)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    return content


def tilejson_document(layer: LayerDescriptor, absolute_path: Path, tile_url: str) -> dict:
    bounds = layer.bounds_wgs84
    return {
        "tilejson": "2.2.0",
        "name": layer.display.display_name,
        "tiles": [tile_url],
        "minzoom": 0,
        "maxzoom": 22,
        "bounds": (
            [bounds.minx, bounds.miny, bounds.maxx, bounds.maxy] if bounds else [-180, -85, 180, 85]
        ),
    }
