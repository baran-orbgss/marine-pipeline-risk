"""Generic multi-epoch seabed-change POC report (MAR-021 Section 25).

Zero dependency on any specific project or dataset, and zero dependency
on `evidence_atlas`/`terrain.report` (self-contained block renderer,
matching this project's established map/report-module-isolation
convention).
"""

from __future__ import annotations

import html as html_module
from typing import Any

REQUIRED_DISCLAIMER = (
    "THIS POC MAPS OBSERVED MULTI-EPOCH SEABED ELEVATION CHANGE. IT DOES NOT PREDICT FUTURE "
    "EROSION OR DEPOSITION."
)


def build_seabed_change_report_blocks(
    *,
    project_title: str,
    epoch1_source_facts: dict[str, Any],
    epoch2_source_facts: dict[str, Any],
    per_epoch_readiness: dict[str, Any],
    datum_compatibility: dict[str, Any],
    grid_compatibility: dict[str, Any],
    common_support: dict[str, Any],
    change_facts: dict[str, Any],
    uncertainty_facts: dict[str, Any],
    comparator_facts: dict[str, Any],
    anthropogenic_limitation_text: str,
    input_contract_summary: list[str],
    what_this_does_not_predict: list[str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "heading", "level": 1, "text": project_title})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER})

    blocks.append({"type": "heading", "level": 2, "text": "1. Purpose"})
    blocks.append(
        {
            "type": "paragraph",
            "text": "Demonstrates the generic production workflow: operator-supplied epoch-1 + "
            "epoch-2 MBES -> per-epoch data readiness -> datum/sign/CRS/grid compatibility -> "
            "common-spatial-support model -> DEM of Difference -> observed seabed change map -> "
            "uncertainty/comparison QA -> GIS + report. No future prediction, no risk score, no "
            "ML anywhere in this workflow.",
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "2. Epoch Sources"})
    blocks.append(
        {"type": "list", "items": [f"Epoch 1: {k}: {v}" for k, v in epoch1_source_facts.items()]}
    )
    blocks.append(
        {"type": "list", "items": [f"Epoch 2: {k}: {v}" for k, v in epoch2_source_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "3. Per-Epoch Readiness"})
    blocks.append(
        {
            "type": "table",
            "headers": ["Epoch", "Status", "Reasons"],
            "rows": [
                [epoch, facts["status"], "; ".join(facts["reasons"]) or "none"]
                for epoch, facts in per_epoch_readiness.items()
            ],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "4. Datum/Sign Compatibility"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in datum_compatibility.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "5. Horizontal/Grid Compatibility"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in grid_compatibility.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "6. Common Support"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in common_support.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "7. Observed Seabed Elevation Change"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in change_facts.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "8. Uncertainty"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in uncertainty_facts.items()]})

    blocks.append(
        {"type": "heading", "level": 2, "text": "9. Official Source-Difference Comparison"}
    )
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in comparator_facts.items()]})

    blocks.append(
        {"type": "heading", "level": 2, "text": "10. Anthropogenic/Natural Attribution Limitation"}
    )
    blocks.append({"type": "paragraph", "text": anthropogenic_limitation_text})

    blocks.append({"type": "heading", "level": 2, "text": "11. Product Transfer Contract"})
    blocks.append({"type": "list", "items": input_contract_summary})

    blocks.append({"type": "heading", "level": 2, "text": "12. What This POC Does Not Predict"})
    blocks.append({"type": "list", "items": what_this_does_not_predict})

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
