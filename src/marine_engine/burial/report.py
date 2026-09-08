"""Generic linear-asset burial/exposure POC report (MAR-024 Section 21).

Zero dependency on any specific project or dataset, and zero dependency on
any other report module -- matches this project's established
map/report-module-isolation convention.
"""

from __future__ import annotations

import html as html_module
from typing import Any

REQUIRED_DISCLAIMER_1 = (
    "THIS POC DISTINGUISHES MEASURED BURIAL STATE FROM FUTURE EXPOSURE SUSCEPTIBILITY."
)
REQUIRED_DISCLAIMER_2 = (
    "A SINGLE DEPTH-OF-BURIAL SURVEY DOES NOT, BY ITSELF, ESTABLISH A FUTURE EXPOSURE PROBABILITY."
)


def build_burial_exposure_report_blocks(
    *,
    project_title: str,
    purpose_text: str,
    source_survey_facts: dict[str, Any],
    burial_reference_semantics_facts: dict[str, Any],
    data_readiness_facts: dict[str, Any],
    route_and_coverage_facts: dict[str, Any],
    canonical_cover_semantics_text: str,
    measured_burial_profile_facts: dict[str, Any],
    source_interpreted_exposure_facts: dict[str, Any],
    burial_margin_text: str,
    exposure_screening_contract_facts: dict[str, Any],
    future_susceptibility_text: str,
    gis_outputs: list[str],
    production_transfer_contract_summary: list[str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "heading", "level": 1, "text": project_title})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER_1})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER_2})

    blocks.append({"type": "heading", "level": 2, "text": "1. Purpose"})
    blocks.append({"type": "paragraph", "text": purpose_text})

    blocks.append({"type": "heading", "level": 2, "text": "2. Source Survey"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in source_survey_facts.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "3. Burial-Reference Semantics"})
    blocks.append(
        {
            "type": "list",
            "items": [f"{k}: {v}" for k, v in burial_reference_semantics_facts.items()],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "4. Data Readiness"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in data_readiness_facts.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "5. Route and Coverage"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in route_and_coverage_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "6. Measured Burial Profile"})
    blocks.append({"type": "paragraph", "text": canonical_cover_semantics_text})
    blocks.append(
        {
            "type": "list",
            "items": [f"{k}: {v}" for k, v in measured_burial_profile_facts.items()],
        }
    )

    blocks.append(
        {"type": "heading", "level": 2, "text": "7. Source-Interpreted Exposure Evidence"}
    )
    blocks.append(
        {
            "type": "list",
            "items": [f"{k}: {v}" for k, v in source_interpreted_exposure_facts.items()],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "8. Burial Margin"})
    blocks.append({"type": "paragraph", "text": burial_margin_text})

    blocks.append({"type": "heading", "level": 2, "text": "9. Exposure-Screening Contract"})
    blocks.append(
        {
            "type": "list",
            "items": [f"{k}: {v}" for k, v in exposure_screening_contract_facts.items()],
        }
    )

    blocks.append(
        {
            "type": "heading",
            "level": 2,
            "text": "10. Why Future Susceptibility Is / Is Not Available",
        }
    )
    blocks.append({"type": "paragraph", "text": future_susceptibility_text})

    blocks.append({"type": "heading", "level": 2, "text": "11. GIS Outputs"})
    blocks.append({"type": "list", "items": gis_outputs})

    blocks.append({"type": "heading", "level": 2, "text": "12. Production Transfer Contract"})
    blocks.append({"type": "list", "items": production_transfer_contract_summary})

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
