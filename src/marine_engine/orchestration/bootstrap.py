"""Explicit, idempotent engine bootstrap (MAR-034 Section 10).

MAR-033A registered its one real capability as an import-time side effect: merely importing
`orchestration.execution` populated `orchestration.runtime.CAPABILITY_RUNTIMES`. That made
capability availability depend on import order -- fragile, and increasingly so as more
capabilities are added. `register_builtin_runtimes` replaces it: CLI, API, and tests all call
this ONE function explicitly, and importing any capability's adapter module by itself registers
nothing anymore.

Idempotent: calling this function more than once is always safe and a no-op after the first
call, regardless of what else has or has not been imported in between (Section 47: "builtin
runtime bootstrap is explicit", "builtin bootstrap is idempotent", "runtime availability is
independent of import order").
"""

from __future__ import annotations

__all__ = ["register_builtin_runtimes"]

_bootstrapped = False


def register_builtin_runtimes() -> None:
    from marine_engine.orchestration import execution as earthquake_triggering_execution
    from marine_engine.orchestration.adapters import bedforms as bedforms_adapter
    from marine_engine.orchestration.adapters import change as change_adapter
    from marine_engine.orchestration.adapters import terrain_canonical, terrain_derivatives

    global _bootstrapped
    if _bootstrapped:
        return

    earthquake_triggering_execution.register_earthquake_triggering_runtime()
    terrain_canonical.register_canonicalize_bathymetry_runtime()
    terrain_derivatives.register_terrain_derivatives_runtime()
    bedforms_adapter.register_bedform_morphodynamics_runtime()
    change_adapter.register_observed_multi_epoch_seabed_change_runtime()

    _bootstrapped = True
