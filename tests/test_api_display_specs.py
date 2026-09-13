"""Tests for api.display_specs: every curated spec is a real, renderable legend -- never a color
drawn on the map with nothing in the legend to explain it, and hillshade/unclassified rasters
carry the explicit "why is this here" note the ticket requires."""

from __future__ import annotations

from api import display_specs as ds


def test_hillshade_is_labeled_display_only_and_not_default_visible() -> None:
    spec = ds.hillshade_display_spec()
    assert "visualization only" in spec.legend.note.lower()
    assert spec.default_visible is False


def test_unclassified_raster_is_labeled_as_semantic_role_not_established() -> None:
    spec = ds.unclassified_display_spec("some_raster.tif")
    assert "semantic role not established" in spec.legend.note.lower()
    assert spec.default_visible is False


def test_every_single_color_spec_has_a_labeled_legend_swatch() -> None:
    # Regression guard: a single_color layer with an empty `legend.stops` list draws a real color
    # on the map but was previously invisible to `Legend.tsx`'s early "nothing to render" check,
    # so it had no legend entry at all -- and `route_analysis_display_spec`'s one stop had no
    # `label`, so where a swatch *did* render it showed a bare "0" instead of a name. Every single-
    # color spec must carry at least one stop with a real label instead.
    specs = [
        ds.route_analysis_display_spec("Free span evidence", []),
        ds.point_evidence_display_spec("CPT test locations", []),
        ds.asset_display_spec("Pipeline / cable route"),
        ds.area_vector_display_spec("Analysis corridor (AOI)"),
    ]
    for spec in specs:
        assert spec.legend.kind == "single_color"
        assert len(spec.legend.stops) >= 1
        assert all(stop.label for stop in spec.legend.stops)


def test_bedform_legend_stop_keeps_the_same_label_as_its_palette_stop() -> None:
    spec = ds.bedform_display_spec("Bedforms (matched crests)", [])
    assert spec.legend.stops[0].label == spec.palette.stops[0].label
