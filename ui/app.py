"""MARINE ENGINE -- Engineering Workbench (UI-001).

Internal, localhost-only Streamlit console: capability explorer, test lab, local output inspector
and roadmap. Launch with `uv run streamlit run ui/app.py`. It changes no scientific behaviour,
runs no arbitrary shell command, never downloads data and fabricates no output.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # `streamlit run ui/app.py` puts ui/ on sys.path, not the root
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from ui import capability_registry as registry  # noqa: E402
from ui import components as c  # noqa: E402
from ui import output_inspector as oi  # noqa: E402
from ui import repo_state  # noqa: E402
from ui import test_runner as tr  # noqa: E402

PAGES = ("Overview", "Capability Explorer", "Test Lab", "Data & Outputs", "Roadmap")

st.set_page_config(
    page_title="Marine Engine Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)
c.inject_css()

if "test_results" not in st.session_state:
    st.session_state.test_results = {}  # node id or label -> tr.RunResult


# --- sidebar ------------------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="wb-header">MARINE ENGINE</p>', unsafe_allow_html=True)
    st.markdown('<p class="wb-sub">ENGINEERING WORKBENCH</p>', unsafe_allow_html=True)
    project_id = st.selectbox(
        "Project context",
        options=[p.id for p in registry.PROJECTS],
        format_func=lambda pid: registry.project_by_id(pid).title,
    )
    page = st.radio("Page", PAGES, label_visibility="collapsed")
    st.divider()
    state = repo_state.read_repository_state()
    st.markdown('<div class="wb-meta">REPOSITORY (read-only)</div>', unsafe_allow_html=True)
    c.kv_table(
        {
            "branch": state.branch or "unknown",
            "HEAD": state.head_sha or "unknown",
            "worktree": state.status_label,
            "changed files": len(state.changed_files),
        }
    )
    if state.error:
        st.caption(f"git unavailable: {state.error}")
    st.divider()
    c.semantics_banner(compact=True)

project = registry.project_by_id(project_id)


def _header() -> None:
    st.markdown('<p class="wb-header">MARINE ENGINE</p>', unsafe_allow_html=True)
    st.markdown('<p class="wb-sub">Engineering Workbench</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wb-meta">Canonical engine: '
        f"<b>{state.head_sha or 'unknown'}</b> &nbsp;&middot;&nbsp; "
        f"Project: <b>{project.title}</b> {c.evidence_chip(project.evidence_kind)}</div>",
        unsafe_allow_html=True,
    )


# --- pages --------------------------------------------------------------------------------------


def page_overview() -> None:
    _header()
    counts = registry.maturity_counts()
    cols = st.columns(4)
    labels = (
        (registry.QUALIFIED_POC, "Qualified Hazard POCs"),
        (registry.FOUNDATION_READY, "Foundation Ready"),
        (registry.PARTIAL, "Partial"),
        (registry.UNDER_CONSTRUCTION, "Under Construction"),
    )
    for col, (maturity, label) in zip(cols, labels, strict=True):
        col.metric(label, counts[maturity])
    c.semantics_banner()

    st.subheader("Hazard capability matrix")
    grid = st.columns(3)
    for i, capability in enumerate(registry.HAZARDS):
        with grid[i % 3]:
            c.capability_card(capability)

    st.subheader("Liquefaction (Hazard 10)")
    c.liquefaction_card()

    st.subheader("Supporting engine capabilities")
    st.dataframe(
        [
            {
                "capability": s.title,
                "maturity": s.maturity,
                "tickets": ", ".join(s.tickets),
                "tests": len(s.tests),
                "description": s.description,
            }
            for s in registry.SUPPORTING
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Maturity legend: "
        + " ".join(f"{m} = {registry.MATURITY_MEANING[m]}" for m in registry.MATURITY_VOCABULARY)
    )


def _outputs_table(entries: list[oi.OutputEntry]) -> None:
    st.dataframe(
        [
            {
                "status": e.status,
                "label": e.label,
                "path": e.relative_path,
                "kind": e.kind,
                "evidence": e.evidence_kind,
                "bytes": e.byte_size,
            }
            for e in entries
        ],
        use_container_width=True,
        hide_index=True,
    )


def page_capability_explorer() -> None:
    _header()
    options = [cap.id for cap in registry.HAZARDS]
    default = 0
    if project.capability_ids:
        default = options.index(project.capability_ids[0])
    cap_id = st.selectbox(
        "Capability",
        options,
        index=default,
        format_func=lambda cid: (
            f"{registry.hazard_by_id(cid).hazard_number:02d}  {registry.hazard_by_id(cid).title}"
        ),
    )
    cap = registry.hazard_by_id(cap_id)
    st.markdown(
        f"## {cap.hazard_number:02d} {cap.title} {c.maturity_chip(cap.maturity)}",
        unsafe_allow_html=True,
    )
    st.caption(registry.MATURITY_MEANING[cap.maturity])
    c.semantics_banner(compact=True)
    c.flow_diagram(cap.flow)

    left, right = st.columns([3, 2])
    with left:
        c.bullet_block("What it does", [cap.what_it_does])
        c.bullet_block("Scientific role", cap.scientific_role)
        c.bullet_block("Inputs", cap.inputs)
        c.bullet_block("Method", cap.method)
        c.bullet_block("Outputs", cap.outputs)
        c.bullet_block("Real benchmark", [cap.real_benchmark])
        c.bullet_block("Next planned capability", [cap.next_planned])
    with right:
        c.limits_panel(cap.limitations)
        st.markdown("**Tests**")
        if cap.tests:
            for t in cap.tests:
                functions = tr.discover_test_functions(t)
                st.markdown(f"- `{t}` ({len(functions)} test functions)")
        else:
            c.mono("none registered")
        st.markdown("**Configs**")
        for cfg in cap.configs or ("none",):
            exists = (_REPO_ROOT / cfg).is_file() if cfg != "none" else False
            st.markdown(f"- `{cfg}` {'' if exists or cfg == 'none' else '(missing locally)'}")
        st.markdown("**CLI**")
        c.mono(
            "\n".join(f"uv run marine-engine {cmd} <config>" for cmd in cap.cli_commands) or "none"
        )
        st.markdown("**Tickets**")
        c.mono(", ".join(cap.tickets) or "none")

    if cap.id == "liquefaction":
        c.liquefaction_card()

    st.subheader("Known outputs (local)")
    if cap.output_patterns:
        entries = oi.discover_outputs(cap.output_patterns)
        _outputs_table(entries)
        present = [e for e in entries if e.exists and e.kind == "image"]
        if present:
            with st.expander(f"Cached figures ({len(present)})", expanded=False):
                for e in present:
                    st.markdown(
                        f"`{e.relative_path}` {c.evidence_chip(e.evidence_kind)}",
                        unsafe_allow_html=True,
                    )
                    st.image(str(e.absolute_path), use_container_width=True)
    else:
        c.mono("No output patterns registered -- " + registry.UNDER_CONSTRUCTION)


def _run_and_store(key: str, command: list[str], timeout_s: int) -> None:
    with st.spinner(f"running: {' '.join(command)}"):
        result = tr.run_command(command, timeout_s=timeout_s)
    st.session_state.test_results[key] = result


def _show_result(result: tr.RunResult) -> None:
    st.markdown(c.status_chip(result.status_label), unsafe_allow_html=True)
    c.kv_table(
        {
            "command": " ".join(result.command),
            "exit code": result.returncode,
            "elapsed": f"{result.elapsed_s:.1f} s",
            "timed out": result.timed_out,
        }
    )
    st.markdown("**stdout**")
    st.code(result.stdout or "(empty)", language="text")
    if result.stderr:
        st.markdown("**stderr**")
        st.code(result.stderr, language="text")


def page_test_lab() -> None:
    _header()
    st.caption(
        "Only tests registered to a capability are discoverable. Commands are argument arrays "
        "(no shell), always `-m 'not live'`, with a timeout. Nothing runs on page load."
    )
    kind = st.radio("Registry", ("Hazard capabilities", "Supporting capabilities"), horizontal=True)
    if kind == "Hazard capabilities":
        items = [
            (cap.id, f"{cap.hazard_number:02d} {cap.title}", cap.tests) for cap in registry.HAZARDS
        ]
    else:
        items = [(s.id, s.title, s.tests) for s in registry.SUPPORTING]
    by_id = {i[0]: i for i in items}
    selected = st.selectbox("Capability", list(by_id), format_func=lambda k: by_id[k][1])
    _, title, test_files = by_id[selected]
    if not test_files:
        c.mono(f"{title}: no registered tests ({registry.UNDER_CONSTRUCTION})")
        return

    results = st.session_state.test_results
    test_file = st.selectbox("Test file", list(test_files))
    functions = tr.discover_test_functions(test_file)
    file_result = results.get(test_file)
    st.markdown(
        f"`{test_file}` &nbsp; {len(functions)} test functions &nbsp; "
        f"{c.status_chip(file_result.status_label if file_result else None)}",
        unsafe_allow_html=True,
    )
    st.dataframe(
        [
            {
                "function": f.name,
                "line": f.lineno,
                "parametrized": f.parametrized,
                "live": f.live,
                "last status (this session)": (
                    results[f.node_id].status_label if f.node_id in results else "not run"
                ),
            }
            for f in functions
        ],
        use_container_width=True,
        hide_index=True,
        height=min(420, 40 + 35 * len(functions)),
    )
    names = [f.name for f in functions if not f.live]
    fn_name = st.selectbox("Test function", names) if names else None

    col_a, col_b, col_c = st.columns(3)
    if fn_name and col_a.button("Run selected test", type="primary"):
        node_id = f"{test_file}::{fn_name}"
        try:
            _run_and_store(node_id, tr.build_pytest_command([node_id]), tr.DEFAULT_TIMEOUT_S)
        except tr.TestRunnerError as exc:
            st.error(str(exc))
    if col_b.button("Run capability tests"):
        try:
            _run_and_store(
                f"capability:{selected}",
                tr.build_capability_command(test_files),
                tr.DEFAULT_TIMEOUT_S,
            )
        except tr.TestRunnerError as exc:
            st.error(str(exc))
    show_source = col_c.toggle("View test source", value=True)

    if fn_name and show_source:
        st.markdown(f"**Source** &nbsp; `{test_file}` &nbsp; `{fn_name}`")
        st.code(tr.extract_function_source(test_file, fn_name), language="python")

    st.subheader("Results (this session)")
    keys = [
        k
        for k in (f"{test_file}::{fn_name}", f"capability:{selected}", "full-suite")
        if k in results
    ]
    for key in keys:
        with st.expander(key, expanded=True):
            _show_result(results[key])

    with st.expander("Advanced", expanded=False):
        st.warning(
            "Full offline suite: ~1850 tests, several minutes, high CPU and memory. Not needed for "
            "normal workbench use. Live/network tests stay excluded."
        )
        confirm = st.checkbox("I understand; run the entire offline suite")
        if st.button("Run full offline suite", disabled=not confirm):
            _run_and_store("full-suite", tr.build_full_suite_command(), tr.FULL_SUITE_TIMEOUT_S)
            _show_result(st.session_state.test_results["full-suite"])


def _inspect_entry(entry: oi.OutputEntry) -> None:
    st.markdown(
        f"`{entry.relative_path}` {c.evidence_chip(entry.evidence_kind)} "
        f"{c.chip(entry.status, 's-pass' if entry.exists else 's-none')}",
        unsafe_allow_html=True,
    )
    if not entry.exists:
        c.mono(oi.NOT_PRESENT_LOCALLY + " -- no data is fabricated for a missing output.")
        return
    path = entry.absolute_path
    if entry.kind == "json":
        info = oi.inspect_json(path)
        summary = oi.summarize_readiness(info["data"])
        if summary:
            st.markdown("**Engine-reported status fields (echoed, not reinterpreted)**")
            st.json(summary, expanded=True)
        with st.expander("Full JSON", expanded=not summary):
            st.json(info["data"], expanded=False)
    elif entry.kind == "parquet":
        info = oi.inspect_parquet(path)
        c.kv_table(
            {
                "rows": info["rows"],
                "row groups": info["row_groups"],
                "columns": len(info["columns"]),
                "bytes": info["byte_size"],
                "created by": info["created_by"],
            }
        )
        if info["marine_engine_metadata"]:
            st.markdown("**marine_engine_* file metadata**")
            st.json(info["marine_engine_metadata"])
        st.dataframe(
            [
                {
                    "column": s.split(":", 1)[0],
                    "type": s.split(":", 1)[1].strip(),
                    "nulls (stats)": info["null_counts"].get(s.split(":", 1)[0]),
                }
                for s in info["schema"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        rows = st.slider("Head rows", 5, 200, 20, key=f"rows-{entry.relative_path}")
        if entry.is_large:
            st.warning(
                f"Large file ({entry.byte_size:,} bytes): only the first batch is read on request."
            )
        if st.button("Load head rows", key=f"head-{entry.relative_path}"):
            st.dataframe(oi.parquet_head(path, rows), use_container_width=True)
    elif entry.kind == "gpkg":
        info = oi.inspect_gpkg(path)
        st.dataframe(
            [
                {
                    "layer": layer["layer"],
                    "geometry": layer["geometry_type"],
                    "features": layer["feature_count"],
                    "crs": layer["crs"],
                    "bounds": layer["bounds"],
                    "fields": len(layer["fields"]),
                }
                for layer in info["layers"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif entry.kind == "geotiff":
        info = oi.inspect_geotiff(path)
        c.kv_table({k: v for k, v in info.items() if k not in ("status", "overviews")})
        st.caption("Header only; no pixel is read on load.")
        if st.button("Render display preview (decimated)", key=f"prev-{entry.relative_path}"):
            import matplotlib.pyplot as plt

            preview = oi.geotiff_preview(path, max_pixels=512)
            st.markdown(c.chip(preview["label"], "e-synthetic"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            image = ax.imshow(preview["array"], cmap="viridis")
            ax.set_title(
                f"{entry.relative_path}\n{preview['source_shape']} -> {preview['preview_shape']} "
                f"(x{preview['decimation_factor']:.1f} decimation)",
                fontsize=8,
            )
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, shrink=0.7)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    elif entry.kind == "image":
        st.image(str(path), use_container_width=True)
    elif entry.kind == "html":
        st.caption(f"{entry.byte_size:,} bytes")
        if st.checkbox("Render HTML report (sandboxed iframe)", key=f"html-{entry.relative_path}"):
            import streamlit.components.v1 as components

            components.html(
                path.read_text(encoding="utf-8", errors="replace"), height=900, scrolling=True
            )
    elif entry.kind == "text":
        st.code(path.read_text(encoding="utf-8", errors="replace")[:20000], language="markdown")
    else:
        c.mono(f"no previewer for kind {entry.kind!r}; {entry.byte_size:,} bytes")


def page_data_outputs() -> None:
    _header()
    st.caption(project.description)
    if project.source_note:
        st.caption(project.source_note)

    st.subheader("Configuration")
    for cfg in project.configs:
        exists = (_REPO_ROOT / cfg).is_file()
        st.markdown(f"- `{cfg}` {'' if exists else '(missing locally)'}")
        if exists and st.checkbox(f"show {cfg}", key=f"cfg-{cfg}"):
            st.code((_REPO_ROOT / cfg).read_text(encoding="utf-8"), language="yaml")

    st.subheader("Readiness (engine-reported)")
    for rel in project.readiness_files:
        try:
            info = oi.inspect_json(rel)
        except oi.OutputInspectorError as exc:
            st.error(str(exc))
            continue
        if info["status"] != "PRESENT":
            st.markdown(
                f"`{rel}` {c.evidence_chip(registry.NOT_AVAILABLE)} {oi.NOT_PRESENT_LOCALLY}",
                unsafe_allow_html=True,
            )
            continue
        summary = oi.summarize_readiness(info["data"])
        st.markdown(
            f"`{rel}` {c.evidence_chip(registry.CACHED_REAL_OUTPUT)}", unsafe_allow_html=True
        )
        st.json(
            summary or {"note": "no top-level status field written by the engine"}, expanded=False
        )

    st.subheader("Local outputs")
    entries: list[oi.OutputEntry] = []
    for d in project.processed_dirs:
        entries += oi.list_directory_outputs(f"processed/{d}", project_id=project.id)
    for d in project.interim_dirs:
        entries += oi.list_directory_outputs(f"interim/{d}", project_id=project.id)
    entries = [oi.with_registry_evidence_kind(e) for e in entries]
    if not entries:
        c.mono(oi.NOT_PRESENT_LOCALLY + " -- no processed/interim outputs exist for this project.")
    else:
        kinds = sorted({e.kind for e in entries})
        chosen_kinds = st.multiselect("Kinds", kinds, default=kinds)
        filtered = [e for e in entries if e.kind in chosen_kinds]
        _outputs_table(filtered)
        paths = [e.relative_path for e in filtered]
        if paths:
            pick = st.selectbox("Inspect file", paths)
            entry = next(e for e in filtered if e.relative_path == pick)
            _inspect_entry(entry)

    st.subheader("Capability outputs expected for this project")
    for cap in registry.capabilities_for_project(project.id):
        with st.expander(f"{cap.hazard_number:02d} {cap.title} -- {cap.maturity}", expanded=False):
            if cap.output_patterns:
                _outputs_table(oi.discover_outputs(cap.output_patterns))
            else:
                c.mono(f"no output patterns registered ({cap.maturity})")


def page_roadmap() -> None:
    _header()
    c.semantics_banner()
    for cap in registry.HAZARDS:
        cols = st.columns([2, 5, 4, 10])
        cols[0].markdown(f"**`{cap.hazard_number:02d}`**")
        cols[1].markdown(f"**{cap.title}**")
        cols[2].markdown(c.maturity_chip(cap.maturity), unsafe_allow_html=True)
        if cap.id == "liquefaction":
            note = "CPT evidence foundation complete; triggering physics UNDER CONSTRUCTION."
        elif cap.maturity == registry.QUALIFIED_POC:
            note = ", ".join(cap.tickets) or "no ticket recorded"
        else:
            note = cap.next_planned
        cols[3].markdown(note)
    st.caption("No ETA is stated for any capability. Order is the canonical hazard-family order.")


{
    "Overview": page_overview,
    "Capability Explorer": page_capability_explorer,
    "Test Lab": page_test_lab,
    "Data & Outputs": page_data_outputs,
    "Roadmap": page_roadmap,
}[page]()
