# CLAUDE.md

This file provides repository-level operating instructions to Claude Code when working with code in this repository.

---

# Project overview

`marine-engine` is a research proof-of-concept engine that evaluates how seabed conditions, morphodynamics, hydrodynamic forcing, and marine geohazards interact with subsea pipelines/cables.

Current demonstrated capability families include:

* bathymetry / seabed terrain;
* multi-epoch seabed change;
* sand-wave / bedform morphometry;
* sediment evidence and noncohesive mobility;
* wave/current forcing and combined bed shear;
* scour susceptibility;
* burial/exposure;
* free-span geometry and support-loss susceptibility;
* engineering evidence/provenance workflows;
* generic operator-project registration and per-asset readiness.

It is:

* a plain importable Python package;
* a thin CLI on top;
* a scientific/engineering research POC;
* not a notebook collection;
* not a web application;
* not a production engineering assurance system;
* not a substitute for qualified engineering judgement.

The first fully worked study case is PL854 (Anglia A -> LOGGS, Southern North Sea).

Sheringham Shoal and Barrow datasets are independent real-data demonstrations used to prove selected capabilities generalize beyond PL854.

---

# 1. Authority hierarchy

When information conflicts, use this order of authority:

1. **Current repository state at the ticket's canonical Git commit**
2. **The current MAR ticket contract**
3. `CLAUDE.md`
4. Relevant accepted tests and machine-readable outputs
5. README / architecture documentation
6. Previous Claude conversation history
7. Assumptions

Repository state and the current ticket override memory from previous Claude sessions.

Previous conversation history is not evidence that a function, file, dataset, result, test, branch, or architectural decision currently exists.

Verify it.

If repository state materially contradicts the current ticket contract:

**STOP and report the discrepancy.**

Do not invent a reconciliation.

---

# 2. Session policy

## One ticket = one fresh Claude Code session

Every new MAR ticket starts in a fresh Claude Code session.

This includes repair tickets:

```text
MAR-026
MAR-026A
MAR-026B
MAR-027
```

Each is a separate session.

A session may continue through:

```text
implementation
→ debugging
→ targeted tests
→ full verification
→ real-data verification
→ repair of defects discovered within the same ticket
```

only while working on the same atomic ticket.

After:

```text
verification
→ commit
→ push
→ local/remote HEAD confirmation
→ final report
```

**STOP.**

Never begin the next MAR ticket in the session that completed the previous one.

A fresh session does **not** mean reading the entire repository again.

It means re-establishing repository truth and loading only the context required by the current ticket.

---

# 3. Session bootstrap

At the start of every MAR ticket, before implementation:

```bash
git status
git fetch origin
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
```

Then:

1. confirm the expected branch;
2. confirm the current HEAD;
3. compare HEAD against the canonical base SHA specified in the ticket;
4. read `CLAUDE.md`;
5. read the current MAR ticket contract;
6. inspect only ticket-relevant code.

If a ticket declares:

```text
Canonical base:
main @ <exact SHA>
```

verify the SHA.

Do not trust it merely because it appears in the prompt.

## Dirty worktree rule

If the worktree is unexpectedly dirty at session start:

**STOP.**

Report:

* modified files;
* untracked files;
* current branch;
* local HEAD;
* `origin/main` HEAD.

Do not assume pre-existing uncommitted changes belong to the current ticket.

Do not delete, overwrite, revert, stage, commit, or absorb them without explicit authority.

---

# 4. Context and token discipline

A fresh session must not perform broad repository ingestion.

Use targeted discovery.

Prefer:

```bash
rg "symbol_name" src tests
git grep "symbol_name"
```

before opening files.

Read:

* implementation directly relevant to the ticket;
* direct callers/callees where needed;
* tests covering the relevant contract;
* only relevant documentation sections.

Expand context only when evidence requires it.

## Large-file rule

Do not open large files in full unless necessary.

In particular:

```text
src/marine_engine/cli.py
README.md
```

can become very large.

When a ticket touches the CLI:

1. locate the relevant `_cmd_<name>` function;
2. locate the relevant `subparsers.add_parser(...)` registration;
3. inspect only those ranges;
4. expand further only if required.

When documentation needs updating, locate the relevant MAR entry or section first.

Do not broadly read:

```text
src/
tests/
README.md
cli.py
```

simply to "understand the project."

Repository-wide comprehension should be assembled incrementally from evidence.

---

# 5. No-assumption policy

Never assume that any of the following exists:

* file;
* function;
* class;
* CLI command;
* dataset;
* cached asset;
* test;
* output;
* previous result;
* branch;
* commit;
* scientific-method implementation.

Search or inspect first.

Avoid reasoning such as:

```text
"this should already exist"
"we probably implemented this earlier"
"the previous ticket likely added..."
```

Replace it with repository evidence.

Previous test counts are historical information only.

If code changed during the current session, any claimed passing result must have been rerun during the current session.

---

# 6. Scientific-method authority

Claude Code is the implementation, testing, CLI, and verification agent.

Scientific-method selection is controlled by the current MAR ticket contract.

Claude Code must not independently introduce or replace:

* scientific equations;
* empirical correlations;
* hazard thresholds;
* sediment-transport formulations;
* scour formulations;
* liquefaction criteria;
* slope-stability methods;
* datum conventions;
* burial-reference conventions;
* uncertainty interpretations;
* evidence semantics;
* validation definitions.

Implementation decisions are allowed.

Scientific-method decisions require explicit ticket authority.

If implementation reveals that the scientific contract is:

* underspecified;
* contradictory;
* unsupported by the cited evidence;
* impossible to implement faithfully;

**STOP and report the exact scientific decision required.**

Do not silently choose a substitute method.

---

# 7. Commands

```bash
uv sync
uv run pytest
uv run pytest tests/test_config.py
uv run pytest tests/test_config.py::test_name -q
uv run pytest -m live
uv run ruff check .
uv run ruff format .
uv run marine-engine --help
```

`pytest` default configuration excludes the `live` marker.

Therefore:

```bash
uv run pytest
```

is intended to be fully offline and deterministic unless repository configuration explicitly changes.

Before a ticket is considered complete, normally run:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

unless the MAR ticket explicitly defines a different verification requirement.

## Local verification vs GitHub CI verification

Since MAR-028 the repository has an independent GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

It runs on `push` to `main` and on `pull_request`, from a clean clone, and verifies only:

```text
uv lock --check
uv sync --frozen
ruff format --check
ruff check
pytest            (offline suite; live marker excluded by pyproject)
uv audit --frozen (blocking dependency vulnerability/adverse-status audit)
```

GitHub CI never performs real-data scientific validation. The PL854 / Sheringham Shoal / Barrow datasets are gitignored and are not available on a runner. Real-data regression remains local/manual.

There is still no `mypy` or pre-commit gate.

Keep the two kinds of verification distinct:

* **local verification** — commands Claude Code ran in the working tree during the session;
* **GitHub CI verification** — the actual GitHub Actions run for the pushed commit.

Do not describe local verification as CI.

Do not claim:

```text
CI passed
```

unless the actual GitHub Actions run for the final commit was inspected and reported `success`.

Do not weaken the workflow (skip a step, add an audit ignore, unpin an action, loosen a marker) to make a ticket pass. A failing gate is a finding to report.

---

# 8. CLI

The CLI follows:

```text
marine-engine <subcommand> <config-or-manifest>
```

Use:

```bash
uv run marine-engine --help
```

for the current command list.

Do not trust a manually duplicated complete subcommand list in documentation to remain current.

Representative commands include:

```bash
uv run marine-engine version
uv run marine-engine validate-config configs/pl854.yaml
uv run marine-engine ingest-pipeline configs/pl854.yaml
uv run marine-engine build-free-span-poc configs/pl854.yaml
uv run marine-engine build-project-readiness configs/project_manifests/sheringham_shoal_2020.yaml
```

Most commands use study configs under:

```text
configs/
```

validated by `StudyConfig`.

`build-project-readiness` instead uses project manifests under:

```text
configs/project_manifests/
```

validated by `ProjectManifest`.

---

# 9. Network and local-data rules

Some commands access live public services such as:

* NSTA;
* MEDIN;
* BGS;
* EMODnet;
* SeaDataNet CDI;
* Copernicus Marine.

Network access must not occur merely because a verification step happens to need data unless the ticket explicitly authorizes it.

Real data already acquired for development may exist only in the local working tree.

The following are gitignored:

```text
data/raw/
data/interim/
data/processed/
```

as are many large geospatial formats.

A fresh clone may therefore not contain real proof datasets.

Missing local data must be reported as missing.

Do not fabricate, substitute, or silently redownload it.

Copernicus authentication is managed externally.

Implementations must not introduce interactive credential prompts.

---

# 10. Architecture

## Pipeline shape

```text
open-data providers
    ↓
preprocessing
    ↓
seabed / morphology features
    ↓
metocean features
    ↓
sediment mobility
    ↓
pipeline interaction
    ↓
hazard / scenario interpretation
    ↓
validation
    ↓
GIS / tabular / report outputs
```

Current package responsibilities:

| Package          | Responsibility                                                         |
| ---------------- | ---------------------------------------------------------------------- |
| `providers`      | External/public source clients and real-data acquisition               |
| `preprocessing`  | AOI, chainage/KP, bathymetry preprocessing, source resolution          |
| `terrain`        | Canonical raster processing, terrain derivatives, bathymetry readiness |
| `change`         | Multi-epoch seabed change / DoD, alignment, uncertainty                |
| `morphology`     | Regional morphology and sand-wave morphometry                          |
| `bedforms`       | Bedform interpretation and natural-context matching                    |
| `sediment`       | Grain size and noncohesive mobility                                    |
| `metocean`       | Waves, currents, combined wave-current bed shear                       |
| `scour`          | Scour-onset screening, susceptibility, external evidence               |
| `burial`         | Burial/exposure state and burial-profile readiness                     |
| `freespan`       | Pipe/seabed clearance, support state, support-loss scenarios           |
| `validation`     | Independent evidence and analog validation/audit workflows             |
| `analogs`        | Named real analog datasets                                             |
| `evidence_atlas` | Engineering evidence bundling/reporting                                |
| `project`        | Generic operator-project registration and per-asset readiness          |
| `resources`      | Bundled static reference resources                                     |
| `pipeline`       | Reserved                                                               |
| `risk`           | Reserved                                                               |
| `export`         | Reserved                                                               |

Do not infer implementation merely because a reserved package exists.

---

# 11. Generic `project/` layer

`project/` was introduced by MAR-026 and its integration semantics were repaired by MAR-026A.

It sits:

```text
operator files
    ↓
project manifest
    ↓
asset registration / identity / provenance
    ↓
project-integration checks
    ↓
per-asset effective readiness
    ↓
independent geohazard engines
```

It is not itself a geohazard engine.

It must not accumulate duplicated hazard science.

For already-implemented domains:

```text
project adapter
    ↓
assemble existing Facts object
    ↓
call accepted readiness function
```

Examples:

```text
terrain.readiness.assess_bathymetry_readiness
burial.readiness.assess_burial_profile_readiness
```

The project layer delegates.

It does not silently rewrite accepted scientific readiness semantics.

---

# 12. Intrinsic vs effective readiness

MAR-026A established a mandatory distinction:

```text
readiness_status_intrinsic
readiness_status_effective
```

## Intrinsic readiness

The readiness conclusion produced by the accepted domain-specific readiness function.

Example:

```text
terrain.readiness.assess_bathymetry_readiness(...)
```

The project layer must not mutate this result merely to encode project-level integration problems.

## Effective readiness

The asset's readiness after project-integration conditions are applied.

A material unresolved integration conflict may cause:

```text
intrinsic = READY
effective = NOT_READY
```

This is valid and intentional.

Both conclusions must remain visible.

The compatibility property:

```text
readiness_status
```

means the **effective** readiness.

Do not silently change that meaning.

---

# 13. CRS integrity

MAR-026A established project-wide CRS integrity rules.

## Declared vs observed CRS

Declared CRS and observed/embedded CRS are separate facts.

They must never be merged.

For supported spatial assets, compare CRS semantically using:

```text
pyproj.CRS
```

rather than raw string equality.

Equivalent CRS representations must not produce false conflicts.

Different projected CRSs must not silently pass merely because both are projected.

For example:

```text
declared: EPSG:32632
observed: EPSG:32631
```

is a material conflict.

## Material CRS conflict

A material declared-vs-observed CRS conflict is a blocking project-integration condition.

The software must not decide which side is correct.

It must not:

* silently reproject;
* silently prefer declared CRS;
* silently prefer embedded CRS;
* infer a replacement CRS.

A material unresolved conflict forces:

```text
readiness_status_effective = NOT_READY
```

while preserving the intrinsic readiness separately.

## Project working CRS

The canonical project `working_crs` must be:

* syntactically valid;
* projected;
* metric.

This validation is a project-level concern and must not depend on a route asset being present.

Do not automatically choose a UTM zone or substitute an asset CRS.

Invalid project working CRS must produce a controlled readiness/registration finding, not an unhandled reprojection exception.

Canonical route reprojection may occur only after all required CRS prerequisites have passed.

---

# 14. Cross-cutting scientific and semantic invariants

These principles must be treated as architectural invariants.

## 14.1 No fabricated scores

Do not invent:

* 0–100 readiness scores;
* arbitrary risk scores;
* hazard percentages;
* confidence percentages;
* weighted traffic-light indexes.

Use explicit state vocabularies and named reasons.

Examples:

```text
READY
READY_WITH_LIMITATIONS
NOT_READY
```

and:

```text
BLOCKING
LIMITATION
```

unless an accepted module defines another explicit vocabulary.

---

## 14.2 Evidence axes remain orthogonal

Keep separate:

* measured/geometric observations;
* source/operator interpretations;
* derived/software outputs;
* scenario/model inference.

Project evidence roles include:

```text
PROJECT_GEOMETRY
MEASURED
SOURCE_INTERPRETED
DERIVED
```

Do not infer evidence role from filenames.

Do not automatically convert:

```text
SOURCE_INTERPRETED → MEASURED
DERIVED → SOURCE_INTERPRETED
```

or collapse evidence axes in any other way.

Source interpretation must not modify a measured/geometric classifier unless an explicit accepted contract requires it.

---

## 14.3 Declared vs observed metadata

Keep operator/source-declared metadata structurally separate from facts observed directly from the file.

Examples of declared metadata:

* stated CRS;
* vertical datum;
* survey epoch;
* units;
* measurement reference;
* supplier;
* source URI.

Examples of observed facts:

* embedded CRS;
* actual geometry type;
* raster dimensions;
* raster pixel size;
* table columns;
* record count;
* byte size;
* SHA-256.

Never merge them into one ambiguous field.

Never silently choose which side is correct when they disagree.

---

## 14.4 Source immutability

Operator/source files are immutable inputs.

Never modify them.

Derived artifacts go elsewhere.

Content identity must be content-based rather than filename-based.

---

## 14.5 Missing means missing

Do not manufacture:

* CRS;
* datum;
* epoch;
* units;
* interpretation;
* provenance;
* uncertainty;
* measurement reference;
* sign convention.

If evidence does not support a value, preserve it as missing or unresolved.

---

## 14.6 Delegate, don't duplicate

A new layer built on an accepted scientific/readiness implementation should call that implementation rather than reimplement its rules.

Where delegation matters, tests should demonstrate it explicitly.

Project-level integration semantics belong above intrinsic scientific readiness, not inside it.

---

# 15. Testing philosophy

Use both:

1. small synthetic tests;
2. real-data proof where the ticket requires it.

Synthetic fixtures should normally be generated inside tests rather than committed as large binary fixtures.

Tests should prove invariants rather than merely execute code.

Examples:

```text
source interpretation cannot alter measured geometry
duplicate timestamps cannot silently disagree
relative paths resolve against manifest location
content identity changes when bytes change
different evidence roles remain distinct
different projected CRSs cannot silently match
material CRS conflicts cannot remain effectively READY
invalid working CRS cannot cause an uncontrolled reprojection
```

Where a new layer delegates to an accepted implementation, test delegation explicitly.

A passing test suite is necessary but does not by itself prove a GIS/scientific output is correct.

---

# 16. Real-output verification

If a ticket produces:

* GeoTIFF;
* GeoPackage;
* Parquet;
* JSON;
* HTML;
* map;
* engineering table;
* route/KP result;

inspect the actual output when the ticket requires real proof.

Do not treat:

```text
command returned exit code 0
```

as sufficient evidence.

Check relevant properties such as:

* CRS;
* geometry;
* dimensions;
* columns;
* record counts;
* semantic labels;
* provenance;
* intrinsic/effective readiness;
* actual numerical values;
* report wording.

Never claim a result was inspected if it was not.

---

# 17. Documentation discipline

`README.md` contains the historical MAR status narrative.

Do not read the whole README by default.

Locate the relevant MAR entry first.

When updating documentation:

* preserve historical statements;
* add new ticket outcomes additively;
* do not rewrite an earlier ticket as if its repair had always existed;
* distinguish original capability tickets from later integrity repairs;
* do not claim support for formats, datasets, hazards, or workflows not demonstrated in code.

Architecture documentation must reflect the actual implementation boundary.

Documentation is not allowed to turn a POC into a production-assurance claim.

---

# 18. Git discipline

Before implementation begins:

```bash
git status
git fetch origin
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
```

Before commit:

```bash
git status
git diff --check
git diff
```

Inspect the actual changed-file set.

Do not commit unrelated modifications.

Do not automatically stage every file without checking what changed.

After required verification passes:

```bash
git add <intentional files>
git status
git diff --cached
```

Inspect the staged diff.

Then commit using an appropriate Conventional Commit message:

```text
feat: ...
fix: ...
```

After push:

```bash
git rev-parse HEAD
git rev-parse origin/main
```

or otherwise fetch/verify the remote ref if required.

The ticket is not complete merely because `git push` returned success.

Confirm the final local and remote canonical commit when the ticket requires push-to-main completion.

---

# 19. STOP conditions

STOP rather than improvise when any of the following occurs:

* canonical base SHA does not match;
* unexpected dirty worktree;
* required canonical file is missing;
* required local real dataset is missing and acquisition is not authorized;
* repository state contradicts the ticket;
* scientific method is underspecified;
* a required source/provenance statement cannot be verified;
* implementation would require silently changing accepted scientific semantics;
* a material CRS/datum/reference conflict cannot be resolved from authoritative evidence;
* completing the task would require beginning another MAR ticket.

A STOP report should state:

1. what was expected;
2. what was actually found;
3. evidence;
4. why continuing would require invention or unauthorized scope expansion.

Do not turn a STOP into an improvised implementation.

---

# 20. Ticket completion standard

Unless the active ticket explicitly says otherwise, a completed implementation ticket should include:

```text
implementation
→ targeted tests during development
→ ruff format
→ ruff check
→ full offline pytest
→ required real-data runs
→ actual output inspection
→ git diff/staged-diff review
→ commit
→ push
→ local/remote HEAD verification
→ GitHub Actions CI result inspection (when the ticket requires push-to-main completion)
→ final report
→ STOP
```

Never begin the next ticket automatically.

---

# 21. Final report requirements

A final MAR report must distinguish:

* what was implemented;
* what was merely tested;
* what was verified against real data;
* what remains unsupported.

Include, where applicable:

1. ticket name;
2. root cause / objective;
3. files changed;
4. public schema/API changes;
5. scientific-method changes — or explicit statement that there were none;
6. tests added/changed;
7. targeted test result;
8. full offline test result;
9. lint/format result;
10. real-data verification result;
11. actual outputs inspected;
12. limitations;
13. final commit SHA;
14. local/remote HEAD confirmation;
15. local verification result (what was run in the session, stated on its own);
16. GitHub CI verification result (the actual workflow run status for the final commit, stated on its own, or an explicit statement that it was not inspected).

Local verification and GitHub CI verification must be reported independently. One does not imply the other.

Do not report historical test counts as if they were executed in the current session.

Do not claim network/live verification if only cached offline data was used.

Do not claim validation, safety, or engineering fitness where only readiness or POC behavior was demonstrated.

---

# 22. Current project-layer accepted invariants

As of accepted MAR-026 + MAR-026A behavior:

```text
operator source bytes
    ≠ declared metadata
    ≠ observed metadata
    ≠ source interpretation
    ≠ measured/geometric evidence
    ≠ derived evidence
    ≠ scenario/model inference
```

These evidence and provenance dimensions remain separate.

For project readiness:

```text
intrinsic readiness
    +
project-integration integrity
    ↓
effective readiness
```

A project or asset may therefore have:

```text
intrinsic = READY
effective = NOT_READY
```

without altering or falsifying the intrinsic scientific readiness conclusion.

A registered asset being `READY` does **not** imply:

```text
the project is ready for every marine geohazard
```

No universal marine-project readiness claim is permitted.

---

# 23. Core working principle

When uncertain:

```text
SEARCH
→ INSPECT
→ VERIFY
→ IMPLEMENT
```

not:

```text
REMEMBER
→ ASSUME
→ IMPLEMENT
```

Repository evidence beats conversation memory.

Scientific honesty beats apparent completeness.

A controlled `NOT_READY`, unresolved field, limitation, or STOP is preferable to a fabricated engineering conclusion.
