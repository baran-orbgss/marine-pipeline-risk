"""Streamlit rendering helpers for the engineering workbench (UI-001 Sections 9, 10, 24).

Presentation only. Nothing here computes a scientific quantity; every value displayed comes from
the capability registry, from the engine's own JSON/Parquet/GeoTIFF/GPKG outputs or from pytest.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from typing import Any

import streamlit as st

from ui import capability_registry as registry

__all__ = [
    "inject_css",
    "chip",
    "maturity_chip",
    "evidence_chip",
    "status_chip",
    "semantics_banner",
    "flow_diagram",
    "capability_card",
    "limits_panel",
    "bullet_block",
    "kv_table",
    "liquefaction_card",
    "mono",
]

_MATURITY_CLASS = {
    registry.QUALIFIED_POC: "m-qualified",
    registry.FOUNDATION_READY: "m-foundation",
    registry.PARTIAL: "m-partial",
    registry.UNDER_CONSTRUCTION: "m-construction",
}
_EVIDENCE_CLASS = {
    registry.REAL_SOURCE: "e-real",
    registry.CACHED_REAL_OUTPUT: "e-cached",
    registry.SYNTHETIC_TEST_FIXTURE: "e-synthetic",
    registry.UNDER_CONSTRUCTION: "e-construction",
    registry.NOT_AVAILABLE: "e-missing",
}

_CSS = """
<style>
:root { --wb-bg:#111417; --wb-panel:#181c21; --wb-border:#2a3038; --wb-text:#d6dbe1;
        --wb-muted:#8a939e; --wb-mono:"JetBrains Mono","Cascadia Code",Consolas,monospace; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: .02em; }
.wb-header { font-family: var(--wb-mono); font-size: 1.55rem; font-weight: 700; letter-spacing:.14em;
             color:#e6ebf0; margin:0; }
.wb-sub { font-family: var(--wb-mono); font-size:.9rem; color: var(--wb-muted); letter-spacing:.08em;
          margin:0 0 .6rem 0; }
.wb-meta { font-family: var(--wb-mono); font-size:.78rem; color: var(--wb-muted); }
.wb-meta b { color: var(--wb-text); font-weight:600; }
.chip { display:inline-block; font-family: var(--wb-mono); font-size:.68rem; letter-spacing:.06em;
        padding:2px 8px; border-radius:3px; border:1px solid; margin:0 4px 4px 0; white-space:nowrap; }
.m-qualified    { color:#7fd1c3; border-color:#2f8f83; background:rgba(47,143,131,.14); }
.m-foundation   { color:#9dc0ea; border-color:#3b6ea5; background:rgba(59,110,165,.16); }
.m-partial      { color:#e6c078; border-color:#a8772a; background:rgba(168,119,42,.16); }
.m-construction { color:#aab2bc; border-color:#5c6370; background:rgba(92,99,112,.18); }
.e-real      { color:#b6d7a8; border-color:#4f7d3a; background:rgba(79,125,58,.16); }
.e-cached    { color:#a9c7e8; border-color:#4a6f96; background:rgba(74,111,150,.16); }
.e-synthetic { color:#e8a9d4; border-color:#8e4a7a; background:rgba(142,74,122,.18); }
.e-construction { color:#aab2bc; border-color:#5c6370; background:rgba(92,99,112,.18); }
.e-missing   { color:#c9c9c9; border-color:#6b6b6b; background:rgba(107,107,107,.16); border-style:dashed; }
.s-pass { color:#8fd19e; border-color:#3f8a4f; background:rgba(63,138,79,.16); }
.s-fail { color:#f0a3a3; border-color:#a34848; background:rgba(163,72,72,.18); }
.s-none { color:#aab2bc; border-color:#5c6370; background:rgba(92,99,112,.18); }
.wb-banner { border:1px solid #6b5a2a; background:rgba(107,90,42,.14); color:#e6d3a3;
             font-family: var(--wb-mono); font-size:.76rem; padding:8px 12px; border-radius:3px;
             margin:.4rem 0 .8rem 0; letter-spacing:.03em; }
.wb-card { border:1px solid var(--wb-border); border-radius:4px; padding:10px 12px; background:var(--wb-panel);
           min-height:118px; margin-bottom:8px; }
.wb-card .num { font-family: var(--wb-mono); color: var(--wb-muted); font-size:.72rem; letter-spacing:.1em; }
.wb-card .ttl { font-weight:600; font-size:1rem; color:#e6ebf0; margin:2px 0 4px 0; }
.wb-card .txt { font-size:.8rem; color: var(--wb-text); line-height:1.3; }
.wb-flow { display:flex; align-items:center; flex-wrap:wrap; gap:6px; font-family: var(--wb-mono);
           font-size:.74rem; letter-spacing:.06em; margin:.3rem 0 .8rem 0; }
.wb-flow .stage { border:1px solid var(--wb-border); background:var(--wb-panel); padding:5px 10px; border-radius:3px;
                  color:#dfe5eb; }
.wb-flow .arrow { color: var(--wb-muted); }
.wb-limits { border-left:3px solid #a34848; background:rgba(163,72,72,.08); padding:8px 12px; border-radius:2px; }
.wb-limits .hd { font-family: var(--wb-mono); font-size:.72rem; letter-spacing:.1em; color:#f0a3a3; }
.wb-limits li { font-size:.84rem; }
.wb-mono { font-family: var(--wb-mono); font-size:.78rem; color: var(--wb-text); white-space:pre-wrap; }
.wb-kv { font-family: var(--wb-mono); font-size:.76rem; border-collapse:collapse; width:100%; }
.wb-kv td { border-bottom:1px solid var(--wb-border); padding:3px 8px; vertical-align:top; }
.wb-kv td:first-child { color: var(--wb-muted); width:32%; }
.wb-liq { border:1px solid #3b6ea5; border-radius:4px; padding:10px 14px; background:rgba(59,110,165,.08); }
.wb-liq .col-hd { font-family: var(--wb-mono); font-size:.72rem; letter-spacing:.1em; color: var(--wb-muted); }
.wb-liq li { font-size:.84rem; list-style:none; margin-left:0; }
.wb-liq ul { padding-left:0; margin:.2rem 0 .6rem 0; }
section[data-testid="stSidebar"] .block-container { padding-top: .8rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _esc(text: Any) -> str:
    return html.escape(str(text))


def chip(text: str, css_class: str) -> str:
    return f'<span class="chip {css_class}">{_esc(text)}</span>'


def maturity_chip(maturity: str) -> str:
    return chip(maturity.replace("_", " "), _MATURITY_CLASS.get(maturity, "m-construction"))


def evidence_chip(kind: str) -> str:
    return chip(kind, _EVIDENCE_CLASS.get(kind, "e-missing"))


def status_chip(label: str | None) -> str:
    if label is None:
        return chip("NOT RUN THIS SESSION", "s-none")
    if label.startswith("PASSED"):
        return chip(label, "s-pass")
    return chip(label, "s-fail")


def semantics_banner(compact: bool = False) -> None:
    text = registry.STATUS_SEMANTICS_NOTICE
    if compact:
        text = "Maturity = software / scientific capability maturity. NOT hazard severity or risk."
    st.markdown(f'<div class="wb-banner">{_esc(text)}</div>', unsafe_allow_html=True)


def flow_diagram(stages: Sequence[str]) -> None:
    parts = []
    for i, stage in enumerate(stages):
        if i:
            parts.append('<span class="arrow">&#8594;</span>')
        parts.append(f'<span class="stage">{_esc(stage)}</span>')
    st.markdown(f'<div class="wb-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def capability_card(capability: registry.Capability) -> None:
    tickets = " ".join(_esc(t) for t in capability.tickets) or "&nbsp;"
    st.markdown(
        f"""<div class="wb-card">
<div class="num">HAZARD {capability.hazard_number:02d} &middot; {tickets}</div>
<div class="ttl">{_esc(capability.title)}</div>
{maturity_chip(capability.maturity)}
<div class="txt">{_esc(capability.what_it_does)}</div>
</div>""",
        unsafe_allow_html=True,
    )


def limits_panel(limitations: Iterable[str]) -> None:
    items = "".join(f"<li>{_esc(x)}</li>" for x in limitations)
    st.markdown(
        f'<div class="wb-limits"><div class="hd">SCIENTIFIC LIMITS</div><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def bullet_block(title: str, items: Iterable[str]) -> None:
    items = list(items)
    st.markdown(f"**{_esc(title)}**")
    if not items:
        st.markdown("<span class='wb-meta'>none</span>", unsafe_allow_html=True)
        return
    st.markdown("\n".join(f"- {x}" for x in items))


def kv_table(rows: dict[str, Any]) -> None:
    body = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows.items())
    st.markdown(f'<table class="wb-kv">{body}</table>', unsafe_allow_html=True)


def mono(text: str) -> None:
    st.markdown(f'<div class="wb-mono">{_esc(text)}</div>', unsafe_allow_html=True)


def liquefaction_card() -> None:
    card = registry.LIQUEFACTION_CARD

    def _list(items: Iterable[str], mark: str) -> str:
        return "<ul>" + "".join(f"<li>{mark} {_esc(x)}</li>" for x in items) + "</ul>"

    st.markdown(
        f"""<div class="wb-liq">
<div class="wb-card" style="border:0;background:transparent;min-height:0;padding:0 0 6px 0">
<div class="ttl">{_esc(card["title"])}</div>{maturity_chip(card["maturity"])}</div>
<div style="display:flex;gap:28px;flex-wrap:wrap">
<div><div class="col-hd">AVAILABLE</div>{_list(card["available"], "&#10003;")}</div>
<div><div class="col-hd">UNDER CONSTRUCTION</div>{_list(card["under_construction"], "&#9633;")}</div>
<div><div class="col-hd">NOT IMPLEMENTED</div>{_list(card["not_implemented"], "&#9633;")}</div>
</div>
<div class="wb-meta">Hazard 10 is FOUNDATION_READY, not QUALIFIED_POC: the CPT evidence foundation is
complete; triggering physics is under construction. No CSR, CRR, factor of safety, LPI or
wave pore-pressure result exists.</div>
</div>""",
        unsafe_allow_html=True,
    )
