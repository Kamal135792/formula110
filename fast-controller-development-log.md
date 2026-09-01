# Fast controller development laboratory notebook

## Retrospective

No laboratory notebook was tracked when this rebuild began. The repository instead contained a chronological controller lineage, search scripts, training scripts, and generated parameter files. The evidence in those files shows these distinct approaches:

- **Actual reinforcement learning:** `controllers.v0` implements a small residual actor and critic trained by episodic REINFORCE in `scripts/train_v0.py`. V21 later retrained and selected a 36 m/s policy across random starts. V75 still uses the two-row V21 actor; the critic and training machinery are not used at inference.
- **Imitation learning:** `scripts/train_v11_imitation.py` fits actor residuals to recorded human controls, and V11 optionally loads the result. No `v11_weights.json` is present, and the V75 dependency graph does not include V11, so this experiment was not retained by V75.
- **Deterministic controller design:** V0's geometric camera planner, three-tick control cadence, overspeed braking, alternating brake pulses, wall recovery, V41's localized sector schedules, V64's first-corner guard, V66's spawn paths, V69's launch/recurring handoff, and V72-V75's final-corner corrections all remain in V75.
- **Grid, coordinate, and random optimization:** the `search_v*.py` scripts tune planner gains, racing-line offsets, sector speed gains, braking boundaries, hazard stabilization, spawn paths, and final-corner steering/throttle. `search_individual_spawn_paths.py` uses seeded random search. These searches produced most fixed V75 schedules.
- **Evolutionary optimization:** `scripts/evolve_v51_partitions.py` mutates a joint line/speed genome. V52 records that branch, but V75's reachable constants come through V41/V42/V46/V50 and later coordinate searches rather than V52.
- **Retained hybrid:** V75 is not an end-to-end network. It retains a deterministic geometric planner with a learned actor residual, fixed-map public-sensor localization, spawn classification, spawn-specific launch guards, track-sector racing-line/speed schedules, an explicit first-lap handoff, and final-corner steering/throttle corrections.

## 2026-09-01 — baseline and dependency audit

- **Participants:** Kamal V. (requirements) and Codex (audit and implementation).
- **Question:** What code and data are actually reachable from v75, and what is the pre-change standard-seed result?
- **Baseline:** `controllers.v75`, commit `f5377d4afea358cd1836486b704739378aef01ac` (`v75 fastest`).
- **Change:** No controller behavior changed. Created branch `refactor/v75-compact-rebuild` from the clean baseline. Traced static imports recursively.
- **Command:** `uv run python scripts/benchmark_controller.py controllers.v75 --seeds 110,2026 --seconds 30 --log artifacts/v75-rebuild-baseline-standard.json`
- **Seeds/duration:** 110 and 2026; 30 seconds each.
- **Metrics:** mean progress 776.3 m; minimum progress 773.6 m; 8 laps; best lap 7.72 s; mean maximum speed 37.58 m/s; mean damage 0; wall contact 0 s; off-track 0.87 s; 0 eliminations.
- **Observation:** The runtime graph is 22 controller modules (23 Python package files including `controllers/__init__.py`), 1,782 physical source lines, and three data assets (`v24_config.json`, `v21_weights.json`, and `track_signature_map.json`). This matches the request's approximate 23-file/1,783-line estimate.
- **Decision:** Accept this commit as the immutable behavioral baseline.
- **Rationale:** The worktree was clean and the standard benchmark completed safely.
- **Next step:** Flatten every reached runtime branch without cleanup or tuning.

## 2026-09-01 — Pass A behavior-preserving flattening

- **Participants:** Kamal V. and Codex.
- **Hypothesis:** The runtime path can be copied into one module without changing any command if initialization order, duplicated localizers, eager branch construction, planner cadence, brake pulses, guard behavior, and handoff timing remain unchanged.
- **Baseline:** `controllers.v75` at `f5377d4`.
- **Change:** Added `src/controllers/v75_compact.py` with flattened localizer, V24 residual/geometric inference, V41 sector logic, V63 launch classifier, V66 spawn paths, V64 guard, V69 handoff, and V72/V75 final-corner behavior. During Pass A it continued to load `v24_config.json` and both actor and critic data from `v21_weights.json` and eagerly created all launch policies.
- **Command:** A `SoloRaceRunner` differential wrapper sent each `RobotSensors` object to v75 and v75_compact and returned the v75 command to the simulator; Ruff and Pyright were also run on the new module.
- **Seeds/duration:** 110, 2026, 10, 31, 64, 96, and 100; 30 seconds each; 12,607 controller ticks total.
- **Metrics before/after:** 0 throttle mismatches; 0 steering mismatches; maximum absolute throttle difference 0; maximum absolute steering difference 0; 0 damage on every run.
- **Observation:** Exact equality held across standard starts, difficult launches, the first-corner guard, launch-to-recurring handoff, recurring laps, and the final corner.
- **Decision:** Accept Pass A and open the Pass B cleanup gate.
- **Rationale:** Command equivalence was demonstrated on live simulator sequences, not inferred from imports or unit tests.
- **Next step:** Remove inference-dead structures and reduce construction/localization work one change class at a time.

## 2026-09-01 — Pass B inference cleanup

- **Participants:** Kamal V. and Codex.
- **Hypothesis:** Fixed V21 actor/config data can be embedded, unselected launch policies can be constructed lazily, and localizers with identical histories can be merged without changing commands.
- **Baseline:** Pass A `controllers.v75_compact`, compared continuously with `controllers.v75` at `f5377d4`.
- **Change:** Embedded the two V21 actor rows and fixed V24 planner configuration; removed critic/checkpoint representations; retained no reward, training, exploration, random state, checkpoint writing, training mode, diagnostic sector telemetry, or module-global fallback controller. Constructed only the selected spawn policy. Shared the launch localizer across the classifier, guard, and launch sector controller because all received the same history from tick zero. Shared the recurring profile/sector localizer because both are created and called together. Stopped updating the launch/handoff localizer once recurring state begins. Removed the inherited but unreachable duplicate near-turn classification in the extended branch.
- **Command:** Ruff, Pyright, the same 12,607-tick differential wrapper, and `uv run pytest tests/test_v75_compact.py -q`.
- **Seeds/duration:** 110, 2026, 10, 31, 64, 96, and 100; 30 seconds each.
- **Metrics before/after:** 0/12,607 command mismatches before cleanup and 0/12,607 after cleanup. New controller tests: 9 passed.
- **Observation:** All cleanup changes preserved exact float-valued commands. The final module requires only `track_signature_map.json`.
- **Decision:** Accept all listed cleanup changes.
- **Rationale:** Exact differential results remained unchanged after the meaningful cleanup.
- **Next step:** Exercise the complete seed union and a delayed-failure run.

## 2026-09-01 — comprehensive equivalence and safety benchmark

- **Participants:** Kamal V. and Codex.
- **Question:** Does the compact controller preserve both commands and race outcomes across standard seeds, launch-search hazards, recurring-lap seeds, and a long run?
- **Baseline:** `controllers.v75` at `f5377d4`.
- **Change:** No tuning. Compared the final compact implementation with the unchanged baseline.
- **Commands:**
  - `uv run python scripts/benchmark_controller.py controllers.v75 --seeds 3,10,13,21,31,44,59,64,68,74,78,86,87,91,93,96,97,100,110,177,191,240,275,467,1337,2026 --seconds 30 --log artifacts/v75-rebuild-baseline-comprehensive.json`
  - The identical command for `controllers.v75_compact`, logging to `artifacts/v75-rebuild-compact-comprehensive.json`.
  - A timed differential wrapper over the same seeds/duration.
  - Separate 120-second seed-110 benchmark commands logged to `artifacts/v75-rebuild-{baseline,compact}-long.json`.
- **Seeds/duration:** 26-seed union shown above at 30 seconds; seed 110 at 120 seconds.
- **30-second metrics (both controllers):** mean progress 772.2 m; minimum progress 749.9 m; 104 total laps; mean per-seed best lap 7.764744 s; best lap 7.016667 s; mean maximum speed 37.56 m/s; overall maximum speed 37.624516 m/s; mean damage approximately 0.000949; total wall contact 0.30 s; total off-track 12.18 s; 0 eliminations.
- **Safety detail:** Baseline and compact episode JSON are identical. Baseline damage was already present on seeds 177 (0.000702), 275 (0.023239), and 1337 (0.000718); the compact controller introduced no new damage, contact, off-track time, elimination, or worst-seed regression.
- **Tick equivalence/runtime:** 46,826 identical public-sensor calls, 0 command mismatches. Approximate call time was 220.595 µs/tick for v75 and 80.207 µs/tick for v75_compact in the same wrapper.
- **120-second metrics (both controllers):** progress 3,155.415 m; 17 laps; best lap 7.716667 s; maximum speed 37.565888 m/s; damage 0; wall contact 0 s; off-track 0.883333 s; 0 eliminations. Episode JSON is identical.
- **Decision:** Accept v75_compact on safety and performance.
- **Rationale:** Commands are exact tick-for-tick, all episode metrics match, the baseline's worst seed is unchanged, and the longer run exposes no delayed divergence.
- **Next step:** Verify packaging isolation and repository-wide static/tests.

## 2026-09-01 — packaging and test verification

- **Participants:** Kamal V. and Codex.
- **Hypothesis:** The selected-controller exporter can discover literal sibling assets and produce a self-contained compact archive.
- **Baseline:** Existing `scripts/export_student_controllers.py`, which followed Python imports but did not include sibling JSON assets.
- **Change:** Added static discovery for literal `Path(__file__).with_name(...)` assets. Added tests for generic asset inclusion and isolated v75_compact extraction/loading.
- **Command:** `uv run pytest tests/test_v75_compact.py tests/test_gradescope_autograder.py -q` plus focused reruns.
- **Seeds/duration:** Not applicable; packaging/static tests.
- **Metrics before/after:** Compact export before the change omitted the required map; after the change the archive contains exactly `controllers/__init__.py`, `controllers/v75_compact.py`, and `controllers/track_signature_map.json`, and loads/runs from an isolated temporary directory. All 9 compact tests and all 3 focused exporter/isolation tests passed.
- **Failed test attempt:** The combined selected run had 24 passes and one unrelated failure: `test_controller_startup_diagnostic_includes_child_exit_and_stderr` timed out after two seconds while starting `bash` on Windows. This test does not exercise the compact controller or exporter; it is retained here rather than silently omitted and is rerun in final verification.
- **Decision:** Accept the exporter change; do not alter the platform-sensitive autograder diagnostic as part of this controller refactor.
- **Rationale:** The new archive is minimal and independently loadable, while changing an unrelated process-timeout test would expand scope.
- **Next step:** Run full Ruff, Pyright, pytest, rebuild the archive, and record final status.

## 2026-09-01 — final verification

- **Participants:** Kamal V. and Codex.
- **Question:** Are the affected files clean, the relevant/full tests accounted for, and the deliverable minimal?
- **Baseline:** Final accepted v75_compact from the comprehensive benchmark.
- **Change:** Ruff formatted only the four affected Python files. No controller parameters or behavior changed. Built `artifacts/v75-compact-submission.zip`.
- **Commands:** `uv run ruff check` and `uv run ruff format --check` on affected files; `uv run pyright`; `uv run pytest -q`; `uv run pytest -q -k 'not test_controller_startup_diagnostic_includes_child_exit_and_stderr'`; exporter CLI; `python -m zipfile -l`.
- **Seeds/duration:** Not applicable to static/package checks; race evidence is in the preceding entry.
- **Metrics:** Ruff lint and format pass on the new controller and all affected Python files. Pyright reports 0 errors, 0 warnings, and 0 informational messages. Focused compact/export tests: 11 passed. Full pytest: 126 passed and 1 environment-only failure; excluding that one test: 126 passed, 1 deselected. A final post-format replay of near-turn seeds 59, 74, and 87 produced 0 mismatches across 5,403 ticks and 0 damage. Final module: 699 physical lines and 26,934 bytes, versus 1,782 lines across the v75 reachable modules. Final archive: 15,307 bytes.
- **Archive contents:** `controllers/__init__.py` (78 bytes), `controllers/v75_compact.py` (26,934 bytes), and `controllers/track_signature_map.json` (19,107 bytes). The isolated extraction/load test passes.
- **Failed checks retained:** Full repository Ruff reports nine pre-existing findings in unrelated files (`capture_race_scene.py`, `search_v36_launch.py`, racing graphics/main/sensors modules, and `test_track_rendering.py`). Full pytest's only failure is the Gradescope process diagnostic: on this Windows machine WSL cannot execute `/bin/bash`, so the child exits 1 instead of the test's expected 126. Neither issue is in an affected file or controller path.
- **Decision:** Accept the final deliverables with the unrelated environment/pre-existing checks explicitly disclosed.
- **Rationale:** All affected static checks, all relevant tests, isolated loading, tick equivalence, comprehensive benchmarks, and the long run pass.
- **Next step:** Commit or submit the compact three-file archive. Any future tuning should begin with a separately logged hypothesis and this exact baseline.

## Intentionally retained behavior

The following structures remain because removing or coalescing them would change state evolution or commands rather than merely remove inference-dead code:

- Separate stable and scheduled residual/geometric planners inside each active sector controller. Both are evaluated every tick and maintain independent held commands, speed smoothing, and brake-pulse state before policy selection.
- Three-tick decision cadence and held-command behavior.
- Alternating reverse/brake pulses and the rule preventing the residual actor from cancelling deterministic overspeed braking.
- A fresh recurring localizer at the handoff. It does not have the same history as the launch localizer on its first recurring tick, so those two localizers were not merged.
- Initial 60-tick stabilization, spawn-specific 90/120-tick stabilization, start-beyond-guard behavior, and start-beyond-merge/wrap behavior.
- All V21 actor weights, geometric planner calculations, V75 launch paths and classifications, sector schedules, and final-corner corrections.

No historical `v0`-`v75` source file was modified or deleted.

## 2026-09-01 — final naming and release branch

- **Participants:** Kamal V. (requested final naming and commit) and Codex (rename, verification, packaging, and commit preparation).
- **Question:** Can the completed controller be presented through a clear, version-independent interface and committed on its own branch?
- **Baseline:** The accepted compact implementation and benchmark evidence above, based on commit `f5377d4`.
- **Change:** Created branch `feature/fast-controller`. Renamed the active module to `controllers.fast_controller`, the display name to `Fast Hybrid Racer`, the test file to `tests/test_fast_controller.py`, and this notebook to `fast-controller-development-log.md`. Historical command names elsewhere in this notebook remain unchanged because they record commands that were actually executed before the rename. Rebuilt the submission as `artifacts/fast-controller-submission.zip` and removed the superseded generated archive.
- **Commands:** Ruff lint/format checks on all affected Python files; full Pyright; pytest excluding the documented Windows `/bin/bash` environment failure; the selected-controller exporter; and `uv run python scripts/benchmark_controller.py controllers.fast_controller --seeds 110,2026 --seconds 30 --log artifacts/fast-controller-verification.json`.
- **Seeds/duration:** Seeds 110 and 2026 for 30 seconds each.
- **Metrics:** Ruff passed; Pyright reported 0 errors and warnings; pytest reported 126 passed and 1 deselected; mean progress 776.3 m; minimum progress 773.6 m; 8 laps; best lap 7.72 s; mean maximum speed 37.58 m/s; damage 0; wall contact 0 s; off-track 0.87 s. The isolated archive contains only `controllers/__init__.py`, `controllers/fast_controller.py`, and `controllers/track_signature_map.json`.
- **Decision:** Accept the version-independent `fast_controller` name and prepare one commit containing the complete implementation, tests, exporter support, and notebook.
- **Rationale:** The renamed public entry point is easier to understand and retains the previously demonstrated behavior and package isolation.
- **Next step:** Commit the staged changes on `feature/fast-controller`.
