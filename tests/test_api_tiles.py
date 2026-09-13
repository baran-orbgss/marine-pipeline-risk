"""Tests for api.tiles: bounded raster tile reads, clean out-of-bounds handling, and no
mutation of the source raster (reprojection/colorization is genuinely display-only)."""

from __future__ import annotations

from pathlib import Path

import morecantile
import numpy as np
import pytest
import rasterio
from api.discovery import inspect_raster
from api.errors import TileOutOfBoundsError
from api.models import DisplaySpec, LayerDescriptor, LegendSpec, PaletteSpec, PaletteStop
from api.tiles import render_tile, tilejson_document
from rasterio.transform import from_origin

_TMS = morecantile.tms.get("WebMercatorQuad")


def _write_raster(path: Path) -> None:
    data = np.linspace(0.0, 50.0, 100, dtype="float32").reshape(10, 10)
    transform = from_origin(500000, 5900000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def _covering_tile(path: Path, z: int = 6) -> tuple[int, int, int]:
    """A tile guaranteed to cover this tiny raster: a low zoom level covers a huge area, so a
    ~100m-wide test raster is always well within one such tile -- no guessed indices."""
    inspection = inspect_raster(path)
    bounds = inspection.bounds_wgs84
    center_lon = (bounds.minx + bounds.maxx) / 2
    center_lat = (bounds.miny + bounds.maxy) / 2
    tile = _TMS.tile(center_lon, center_lat, z)
    return z, tile.x, tile.y


def _layer_for(path: Path) -> LayerDescriptor:
    stops = [PaletteStop(value=0.0, color="#000000"), PaletteStop(value=50.0, color="#ffffff")]
    return LayerDescriptor(
        layer_id="test:layer",
        project_id="test",
        group="DATA",
        layer_type="raster",
        support_type="AREA_SURFACE",
        relative_path=str(path),
        display=DisplaySpec(
            display_name="Test",
            palette=PaletteSpec(kind="continuous", domain=(0.0, 50.0), stops=stops),
            legend=LegendSpec(title="Test", kind="continuous", stops=stops),
        ),
    )


def test_render_tile_produces_png_bytes_within_bounds(tmp_path: Path) -> None:
    raster_path = tmp_path / "bathy.tif"
    _write_raster(raster_path)
    layer = _layer_for(raster_path)
    z, x, y = _covering_tile(raster_path)

    content = render_tile(layer, raster_path, z, x, y)

    assert content[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_tile_out_of_bounds_raises_clean_error_not_500(tmp_path: Path) -> None:
    raster_path = tmp_path / "bathy.tif"
    _write_raster(raster_path)
    layer = _layer_for(raster_path)
    z, x, y = _covering_tile(raster_path)
    # A tile far from the covering one, at the same zoom, is guaranteed outside the tiny raster.
    far_x, far_y = (x + 5) % (2**z), (y + 5) % (2**z)

    with pytest.raises(TileOutOfBoundsError):
        render_tile(layer, raster_path, z, far_x, far_y)


def test_render_tile_never_mutates_the_source_file(tmp_path: Path) -> None:
    raster_path = tmp_path / "bathy.tif"
    _write_raster(raster_path)
    layer = _layer_for(raster_path)
    z, x, y = _covering_tile(raster_path)
    before_bytes = raster_path.read_bytes()
    before_mtime = raster_path.stat().st_mtime_ns

    render_tile(layer, raster_path, z, x, y)
    render_tile(layer, raster_path, z, x, y)  # second call should hit the tile cache

    assert raster_path.read_bytes() == before_bytes
    assert raster_path.stat().st_mtime_ns == before_mtime


def test_tilejson_document_reports_layer_bounds(tmp_path: Path) -> None:
    raster_path = tmp_path / "bathy.tif"
    _write_raster(raster_path)
    layer = _layer_for(raster_path)
    from api.discovery import inspect_raster

    inspection = inspect_raster(raster_path)
    layer = layer.model_copy(update={"bounds_wgs84": inspection.bounds_wgs84})

    doc = tilejson_document(layer, raster_path, "http://example/tiles/{z}/{x}/{y}.png")

    assert doc["tilejson"] == "2.2.0"
    assert doc["bounds"] != [-180, -85, 180, 85]
