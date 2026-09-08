"""Generic operator-project registration/readiness report (MAR-026 Section 14).

Zero dependency on any specific project or dataset, and zero dependency on any other report
module (matches this project's established map/report-module-isolation convention). Report
wording never uses "validated", "safe", or "low risk" -- only readiness/registration language.
"""

from __future__ import annotations

import html as html_module
from typing import Any

REQUIRED_DISCLAIMER = (
    "THIS REPORT DESCRIBES PROJECT REGISTRATION AND PER-ASSET READINESS ONLY. IT DOES NOT "
    "CLAIM THE PROJECT IS READY FOR SCOUR, FREE-SPAN, LIQUEFACTION, SHALLOW GAS, OR ANY OTHER "
    "MARINE GEOHAZARD ANALYSIS."
)


def build_project_readiness_report_blocks(
    *,
    project_title: str,
    purpose_text: str,
    project_identity_facts: dict[str, Any],
    asset_rows: list[dict[str, Any]],
    evidence_role_summary: dict[str, Any],
    readiness_summary: dict[str, Any],
    blocking_issues: list[str],
    limitations: list[str],
    unsupported_categories: list[str],
    production_transfer_notes: list[str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "heading", "level": 1, "text": project_title})
    blocks.append({"type": "callout", "text": REQUIRED_DISCLAIMER})

    blocks.append({"type": "heading", "level": 2, "text": "1. Project Identity"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in project_identity_facts.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "2. Purpose"})
    blocks.append({"type": "paragraph", "text": purpose_text})

    blocks.append({"type": "heading", "level": 2, "text": "3. Registered Assets"})
    if asset_rows:
        headers = list(asset_rows[0].keys())
        blocks.append(
            {
                "type": "table",
                "headers": headers,
                "rows": [[row.get(h, "") for h in headers] for row in asset_rows],
            }
        )
    else:
        blocks.append({"type": "paragraph", "text": "No assets registered."})

    blocks.append({"type": "heading", "level": 2, "text": "4. Evidence Roles"})
    blocks.append(
        {"type": "list", "items": [f"{k}: {v}" for k, v in evidence_role_summary.items()]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "5. Readiness Summary"})
    blocks.append({"type": "list", "items": [f"{k}: {v}" for k, v in readiness_summary.items()]})

    blocks.append({"type": "heading", "level": 2, "text": "6. Blocking Issues"})
    blocks.append(
        {"type": "list", "items": blocking_issues or ["none -- no BLOCKING check failed"]}
    )

    blocks.append({"type": "heading", "level": 2, "text": "7. Limitations"})
    blocks.append({"type": "list", "items": limitations or ["none recorded"]})

    blocks.append(
        {"type": "heading", "level": 2, "text": "8. Explicitly Unsupported Readiness Categories"}
    )
    blocks.append(
        {
            "type": "list",
            "items": unsupported_categories
            or ["none -- every registered asset used a category with an implemented adapter"],
        }
    )

    blocks.append({"type": "heading", "level": 2, "text": "9. Production Transfer Notes"})
    blocks.append({"type": "list", "items": production_transfer_notes})

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
    body { font-family: Georgia, 'Times New Roman', serif; max-width: 1100px; margin: 2rem auto;
           padding: 0 1.5rem; color: #1a1a1a; line-height: 1.55; }
    h1 { font-size: 1.8rem; border-bottom: 3px solid #1a1a1a; padding-bottom: 0.3rem; }
    h2 { font-size: 1.2rem; margin-top: 2rem; color: #2a3d5c; border-bottom: 1px solid #ccc; }
    ul { padding-left: 1.4rem; }
    li { margin-bottom: 0.3rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.82rem; }
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
