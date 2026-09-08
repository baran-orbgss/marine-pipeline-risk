"""Generic bedform-morphodynamics POC report (MAR-022 Section 27).

Zero dependency on any specific project or dataset, and zero dependency
on any other report module (self-contained block renderer, matching this
project's established map/report-module-isolation convention).
"""

from __future__ import annotations

import html as html_module
from typing import Any

REQUIRED_DISCLAIMER_1 = (
    "THIS POC CHARACTERISES SANDBED BEDFORM GEOMETRY AND, WHERE DEFENSIBLE, OBSERVED CREST "
    "DISPLACEMENT BETWEEN TWO SURVEY EPOCHS."
)
REQUIRED_DISCLAIMER_2 = "IT DOES NOT PREDICT FUTURE SAND-WAVE MIGRATION."


def build_bedform_morphodynamics_report_blocks(
    *,
    project_title: str,
    source_data_facts: dict[str, Any],
    canonical_support_facts: dict[str, Any],
    natural_vs_anthropogenic_facts: dict[str, Any],
    epoch1_morphometry_facts: dict[str, Any],
    epoch2_morphometry_facts: dict[str, Any],
    crest_matching_facts: dict[str, Any],
    observed_displacement_facts: dict[str, Any],
    dod_supporting_context_text: str,
    source_interpretation_comparison_facts: dict[str, Any],
    limitations: list[str],
    input_contract_summary: list[str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "heading", "level": 1, "text": project_title})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER_1})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER_2})

    blocks.append({"type": "heading", "level": 2, "text": "1. Purpose"})
    blocks.append(
        {
            "type": "paragraph",
            "text": "Demonstrates the generic production workflow: high-resolution bathymetry -> "
            "bedform detection -> bedform morphometry -> multi-epoch crest comparison -> observed "
            "bedform change/displacement map -> GIS + report. Static geometry and observed change "
            "only -- no future migration prediction, no freespan/scour susceptibility, no route "
            "suitability, no hazard/risk score, no ML anywhere in this workflow.",
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "2. Source Data"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in source_data_facts.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "3. Canonical Support"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in canonical_support_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "4. Natural vs Anthropogenic Context"})
    blocks.append(
        {
            "type": "list",
            "items": [f"{k}: {v}" for k, v in natural_vs_anthropogenic_facts.items()],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "5. 2018 Morphometry"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in epoch1_morphometry_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "6. 2020 Morphometry"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in epoch2_morphometry_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "7. Crest Matching"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in crest_matching_facts.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "8. Observed Apparent Displacement"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in observed_displacement_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "9. DoD Supporting Context"})
    blocks.append({"type": "paragraph", "text": dod_supporting_context_text})

    blocks.append({"type": "heading", "level": 2, "text": "10. Source Interpretation Comparison"})
    blocks.append(
        {
            "type": "list",
            "items": [f"{k}: {v}" for k, v in source_interpretation_comparison_facts.items()],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "11. Limitations"})
    blocks.append({"type": "list", "items": limitations})

    blocks.append({"type": "heading", "level": 2, "text": "12. Production Transfer Contract"})
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
