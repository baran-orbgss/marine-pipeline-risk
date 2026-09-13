"""Tests for api.allowlist: every launchable command is a real, registered CLI subcommand, and
capability availability reflects only structurally-verifiable facts (a declared config exists, a
registered asset resolves to a real file) -- never an invented readiness judgement."""

from __future__ import annotations

from pathlib import Path

import yaml
from api.allowlist import TIER_A, describe_capabilities

from marine_engine.cli import build_parser


def _registered_subcommands() -> set[str]:
    parser = build_parser()
    # argparse exposes subparsers via the private `_subparsers` action choices mapping -- this is
    # the same introspection approach a "prove the allowlist is real" test needs.
    for action in parser._actions:  # noqa: SLF001
        if hasattr(action, "choices") and action.choices:
            return set(action.choices.keys())
    raise AssertionError("could not find the subcommand action on the CLI parser")


def test_every_allowlisted_command_is_a_real_registered_cli_subcommand() -> None:
    registered = _registered_subcommands()
    for definition in TIER_A:
        assert definition.cli_command in registered, (
            f"{definition.cli_command!r} is not a real marine-engine CLI subcommand"
        )


def test_capability_keys_are_unique() -> None:
    keys = [d.capability_key for d in TIER_A]
    assert len(keys) == len(set(keys))


def test_capability_with_no_declared_study_config_is_not_applicable(api_sandbox: Path) -> None:
    (api_sandbox / "configs" / "project_manifests" / "no_config.yaml").parent.mkdir(
        parents=True, exist_ok=True
    )
    (api_sandbox / "configs" / "project_manifests" / "no_config.yaml").write_text(
        yaml.safe_dump(
            {"project": {"id": "no_config", "name": "x", "working_crs": "EPSG:32631"}, "assets": []}
        ),
        encoding="utf-8",
    )
    (api_sandbox / "data" / "processed" / "no_config").mkdir(parents=True)

    descriptors = describe_capabilities("no_config")
    terrain = next(d for d in descriptors if d.capability_key == "terrain")

    assert terrain.availability == "NOT_APPLICABLE"
    assert terrain.disabled is True


def test_capability_requiring_bathymetry_is_missing_inputs_without_a_registered_asset(
    api_sandbox: Path,
) -> None:
    (api_sandbox / "configs" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {"study": {"id": "pl854", "name": "PL854"}, "crs": {"horizontal": "EPSG:32631"}}
        ),
        encoding="utf-8",
    )
    (api_sandbox / "configs" / "project_manifests" / "pl854.yaml").write_text(
        yaml.safe_dump(
            {"project": {"id": "pl854", "name": "PL854", "working_crs": "EPSG:32631"}, "assets": []}
        ),
        encoding="utf-8",
    )
    (api_sandbox / "data" / "processed" / "pl854").mkdir(parents=True)

    descriptors = describe_capabilities("pl854")
    terrain = next(d for d in descriptors if d.capability_key == "terrain")

    assert terrain.availability == "MISSING_INPUTS"
    assert any("bathymetry" in reason for reason in terrain.reasons)


def test_not_generalized_capabilities_always_report_a_reason(api_sandbox: Path) -> None:
    descriptors = describe_capabilities("anything")
    not_generalized_keys = {"sediment_mobility", "free_span", "bathymetry"}
    seen = {d.capability_key for d in descriptors if d.capability_key in not_generalized_keys}
    assert seen == not_generalized_keys
    for descriptor in descriptors:
        if descriptor.capability_key in not_generalized_keys:
            assert descriptor.availability == "NOT_APPLICABLE"
            assert descriptor.reasons
