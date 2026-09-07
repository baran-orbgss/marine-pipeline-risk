"""Generic high-resolution terrain POC report (MAR-020 Section 18).

Zero dependency on any specific project or dataset, and zero dependency
on `evidence_atlas` (self-contained block renderer, matching this
project's established map/report-module-isolation convention). Every
factual statement is derived from caller-supplied real data, never a
hard-coded stale number.
"""

from __future__ import annotations

import html as html_module
from typing import Any


def build_terrain_poc_report_blocks(
    *,
    project_title: str,
    source_facts: dict[str, Any],
    readiness_status: str,
    readiness_reasons: list[str],
    bathymetry_stats: dict[str, Any],
    derived_layer_facts: list[dict[str, Any]],
    scale_facts: list[dict[str, Any]],
    gis_outputs: list[dict[str, Any]],
    limitations: list[str],
    input_contract_summary: list[str],
) -> list[dict[str, Any]]:
    """The full report content as format-neutral blocks, in the required
    8-section order (Section 18)."""

    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "heading", "level": 1, "text": project_title})
    blocks.append(
        {
            "type": "callout",
            "text": "THIS POC DEMONSTRATES SOFTWARE PROCESSING OF PROJECT-GRADE HIGH-RESOLUTION "
            "BATHYMETRY; IT DOES NOT INTERPRET GEOHAZARD RISK.",
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "1. Source"})
    blocks.append(
        {
            "type": "list",
            "items": [f"{key}: {value}" for key, value in source_facts.items()],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "2. Readiness"})
    blocks.append({"type": "paragraph", "text": f"Status: {readiness_status}"})
    if readiness_reasons:
        blocks.append({"type": "list", "items": readiness_reasons})
    else:
        blocks.append({"type": "paragraph", "text": "No limitation or blocking reasons recorded."})

    blocks.append({"type": "heading", "level": 2, "text": "3. Bathymetry"})
    blocks.append(
        {"type": "list", "items": [f"{key}: {value}" for key, value in bathymetry_stats.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "4. Derived Terrain Layers"})
    blocks.append(
        {
            "type": "table",
            "headers": ["Layer", "Scale", "Units", "Valid cells", "Min", "Max", "Mean"],
            "rows": [
                [
                    f["layer"],
                    f["scale"],
                    f["units"],
                    f["valid_cells"],
                    f["min"],
                    f["max"],
                    f["mean"],
                ]
                for f in derived_layer_facts
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "5. Scale Dependence"})
    blocks.append(
        {
            "type": "table",
            "headers": ["Scale name", "Physical radius/step (m)", "Pixel radius/step", "Rationale"],
            "rows": [
                [f["name"], f["physical_m"], f["pixel_count"], f["rationale"]] for f in scale_facts
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "6. GIS Outputs"})
    blocks.append(
        {
            "type": "table",
            "headers": ["File", "Type", "Description"],
            "rows": [[g["file"], g["type"], g["description"]] for g in gis_outputs],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "7. Limitations"})
    blocks.append({"type": "list", "items": limitations})

    blocks.append({"type": "heading", "level": 2, "text": "8. Operator-Data Transfer Contract"})
    blocks.append({"type": "list", "items": input_contract_summary})

    return blocks


def render_blocks_html(blocks: list[dict[str, Any]], *, title: str) -> str:
    def esc(text: str) -> str:
        return html_module.escape(str(text))

    parts: list[str] = []
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            level = block["level"]
            parts.append(f"<h{level}>{esc(block['text'])}</h{level}>")
        elif kind == "paragraph":
            parts.append(f"<p>{esc(block['text'])}</p>")
        elif kind == "list":
            items = "".join(f"<li>{esc(item)}</li>" for item in block["items"])
            parts.append(f"<ul>{items}</ul>")
        elif kind == "table":
            header_html = "".join(f"<th>{esc(h)}</th>" for h in block["headers"])
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
                for row in block["rows"]
            )
            parts.append(
                f"<table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"
            )
        elif kind == "callout":
            parts.append(f'<div class="callout">{esc(block["text"])}</div>')
        else:  # pragma: no cover -- defensive, every block type above is exhaustive
            raise ValueError(f"unknown report block type: {kind}")

    body = "\n".join(parts)
    style = """
    body { font-family: Georgia, 'Times New Roman', serif; max-width: 960px; margin: 2rem auto;
           padding: 0 1.5rem; color: #1a1a1a; line-height: 1.55; }
    h1 { font-size: 1.8rem; border-bottom: 3px solid #1a1a1a; padding-bottom: 0.3rem; }
    h2 { font-size: 1.2rem; margin-top: 2rem; color: #2a3d5c; border-bottom: 1px solid #ccc; }
    ul { padding-left: 1.4rem; }
    li { margin-bottom: 0.3rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.88rem; }
    th, td { border: 1px solid #999; padding: 0.35rem 0.55rem; text-align: left; }
    th { background: #e8e6dd; }
    .callout { background: #eef3fb; border: 1px solid #2a3d5c; border-radius: 4px;
               padding: 0.7rem 1rem; margin: 1rem 0; font-weight: bold; color: #1c2b45; }
    """
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{esc(title)}</title>"
        f"<style>{style}</style></head><body>{body}</body></html>"
    )
