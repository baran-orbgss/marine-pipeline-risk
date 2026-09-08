"""Generic pipeline free-span support-loss POC report (MAR-025 Section 30).

Zero dependency on any specific project or dataset, and zero dependency on any other report
module (matches this project's established map/report-module-isolation convention).
"""

from __future__ import annotations

import html as html_module
from typing import Any

REQUIRED_DISCLAIMER = (
    "MAR-025 IDENTIFIES / SCREENS GEOMETRIC PIPE SUPPORT LOSS. IT DOES NOT ASSESS VIV, FATIGUE "
    "LIFE, ULS/FLS ACCEPTABILITY, OR PIPELINE FAILURE PROBABILITY."
)


def build_free_span_report_blocks(
    *,
    project_title: str,
    purpose_text: str,
    product_boundary_text: str,
    operator_input_model_facts: dict[str, Any],
    canonical_geometry_facts: dict[str, Any],
    measured_free_span_facts: dict[str, Any],
    gap_governance_text: str,
    support_loss_scenario_facts: dict[str, Any],
    synthetic_validation_facts: dict[str, Any],
    nsta_registry_facts: dict[str, Any],
    pl854_observed_context_facts: dict[str, Any],
    structural_boundary_text: str,
    production_transfer_contract_summary: list[str],
    limitations: list[str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "heading", "level": 1, "text": project_title})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER})

    blocks.append({"type": "heading", "level": 2, "text": "1. Purpose"})
    blocks.append({"type": "paragraph", "text": purpose_text})

    blocks.append({"type": "heading", "level": 2, "text": "2. Product Boundary"})
    blocks.append({"type": "paragraph", "text": product_boundary_text})

    blocks.append({"type": "heading", "level": 2, "text": "3. Operator Input Model"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in operator_input_model_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "4. Canonical Pipe/Seabed Geometry"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in canonical_geometry_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "5. Measured Free-Span Extraction"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in measured_free_span_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "6. Gap Governance"})
    blocks.append({"type": "paragraph", "text": gap_governance_text})

    blocks.append({"type": "heading", "level": 2, "text": "7. Support-Loss Scenario Screening"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in support_loss_scenario_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "8. Synthetic Exact Validation"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in synthetic_validation_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "9. NSTA Observed Free-Span Registry"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in nsta_registry_facts.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "10. PL854 Observed Context"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in pl854_observed_context_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "11. Structural Assessment Boundary"})
    blocks.append({"type": "paragraph", "text": structural_boundary_text})

    blocks.append({"type": "heading", "level": 2, "text": "12. Production Transfer Contract"})
    blocks.append({"type": "list", "items": production_transfer_contract_summary})

    blocks.append({"type": "heading", "level": 2, "text": "13. Limitations"})
    blocks.append({"type": "list", "items": limitations})

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
