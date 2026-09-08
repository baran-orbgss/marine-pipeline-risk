"""Generic linear-asset scour susceptibility POC report (MAR-023 Section 20).

Zero dependency on any specific project or dataset, and zero dependency on
any other report module -- matches this project's established
map/report-module-isolation convention (see `bedforms/report.py`'s
docstring).
"""

from __future__ import annotations

import html as html_module
from typing import Any

REQUIRED_DISCLAIMER_1 = "PIPELINE SCOUR SUSCEPTIBILITY REQUIRES ACTUAL PIPELINE EMBEDMENT."
REQUIRED_DISCLAIMER_2 = (
    "PL854 IS A SCENARIO BENCHMARK ONLY BECAUSE A CONTINUOUS OBSERVED EMBEDMENT PROFILE IS "
    "NOT AVAILABLE."
)
REQUIRED_DISCLAIMER_3 = (
    "SHERINGHAM 2024 OBSERVED SCOUR EVIDENCE IS NOT USED TO VALIDATE PIPELINE-SCOUR PHYSICS."
)


def build_scour_poc_report_blocks(
    *,
    project_title: str,
    purpose_text: str,
    pipeline_method_facts: dict[str, Any],
    required_inputs: list[str],
    actual_vs_critical_facts: dict[str, Any],
    exceedance_fraction_facts: dict[str, Any],
    pl854_scenario_facts: dict[str, Any],
    pl854_limitations: list[str],
    sheringham_evidence_facts: dict[str, Any],
    asset_physics_mismatch_text: str,
    production_transfer_contract_summary: list[str],
    not_predicted: list[str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "heading", "level": 1, "text": project_title})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER_1})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER_2})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER_3})

    blocks.append({"type": "heading", "level": 2, "text": "1. Purpose"})
    blocks.append({"type": "paragraph", "text": purpose_text})

    blocks.append(
        {"type": "heading", "level": 2, "text": "2. Pipeline Scour-Onset Scientific Method"}
    )
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in pipeline_method_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "3. Required Operator Inputs"})
    blocks.append({"type": "list", "items": required_inputs})

    blocks.append({"type": "heading", "level": 2, "text": "4. Actual vs Critical Embedment"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in actual_vs_critical_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "5. Forcing-Record Exceedance Fraction"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in exceedance_fraction_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "6. PL854 Scenario Benchmark"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in pl854_scenario_facts.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "7. PL854 Limitations"})
    blocks.append({"type": "list", "items": pl854_limitations})

    blocks.append(
        {"type": "heading", "level": 2, "text": "8. Sheringham 2024 Observed Scour Evidence"}
    )
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in sheringham_evidence_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "9. Asset-Physics Mismatch"})
    blocks.append({"type": "paragraph", "text": asset_physics_mismatch_text})

    blocks.append({"type": "heading", "level": 2, "text": "10. Production Transfer Contract"})
    blocks.append({"type": "list", "items": production_transfer_contract_summary})

    blocks.append({"type": "heading", "level": 2, "text": "11. What This POC Does Not Predict"})
    blocks.append({"type": "list", "items": not_predicted})

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
    .callout { background: #fbeeee; border: 1px solid #8b0000; border-radius: 4px;
               padding: 0.7rem 1rem; margin: 1rem 0; font-weight: bold; color: #5c1414; }
    """
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{esc(title)}</title>"
        f"<style>{style}</style></head><body>{body}</body></html>"
    )
