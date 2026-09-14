"""Capability-runtime adapters (MAR-034 Section 53).

Each module here binds one already-accepted domain package (`terrain`, `bedforms`, `change`) to
the generic `orchestration.runtime.CapabilityRuntime` shape -- reusing that package's existing
science exactly, never reimplementing it. See `orchestration.bootstrap.register_builtin_runtimes`
for the one place all of them get registered.
"""

from __future__ import annotations
