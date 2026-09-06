"""Offline unit tests for marine_engine.morphology.sandwave_morphometry_map
(MAR-017B).

Small synthetic arrays only -- never a real archive, never network access.
"""

import warnings

import numpy as np
import pytest

from marine_engine.morphology import sandwave_morphometry_map as swmap

# --- MAR-017B Section 26 test H: background nodata sentinel never reaches matplotlib ------
# --- normalization ---------------------------------------------------------------------------


def test_H_float32_nodata_sentinel_never_reaches_matplotlib_normalization(tmp_path):
    """A real run previously emitted `RuntimeWarning: overflow encountered
    in multiply` from matplotlib's colour normalization touching the raw
    float32 nodata sentinel still sitting in a masked array's underlying
    `.data` buffer. Reproduces that exact scenario (an extreme float32
    sentinel at every invalid cell) and asserts no warning is raised."""

    elevation = np.full((50, 50), -3.4028235e38, dtype=np.float32)
    elevation[10:40, 10:40] = -45.0
    valid = elevation != np.float32(-3.4028235e38)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        output_path = swmap.render_native_bathymetry_overview(
            elevation=elevation,
            valid=valid,
            extent_m=(0.0, 50.0, 0.0, 50.0),
            crs="EPSG:32631",
            native_pixel_size_m=1.0,
            survey_year=2011,
            nodata_value=-3.4028235e38,
            output_path=tmp_path / "overview.png",
        )
    assert output_path.exists()


def test_H_safe_masked_array_removes_the_extreme_sentinel_from_the_data_buffer():
    """Direct check of the fix's mechanism: `_safe_masked_array` must not
    just MARK invalid cells (leaving the raw sentinel in `.data`), it must
    REPLACE them with `nan`, so no downstream consumer of the masked
    array's raw buffer (e.g. matplotlib's colour-normalization internals,
    which still touch `.data` even for masked entries) can ever see the
    extreme value."""

    sentinel = np.float32(-3.4028235e38)
    elevation = np.array([[sentinel, -45.0], [-46.0, sentinel]], dtype=np.float32)
    valid = elevation != sentinel

    masked = swmap._safe_masked_array(elevation, valid)
    assert np.isnan(masked.data[~valid]).all()
    assert not np.any(masked.data == sentinel)


def test_H_safe_masked_array_never_alters_a_valid_value():
    elevation = np.array([[-3.4028235e38, -45.25], [-46.75, -3.4028235e38]], dtype=np.float32)
    valid = elevation != np.float32(-3.4028235e38)

    masked = swmap._safe_masked_array(elevation, valid)
    assert masked.data[valid] == pytest.approx([-45.25, -46.75])
