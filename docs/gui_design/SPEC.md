# VELES GUI — Design & Logic Spec

Handoff spec for rebuilding the VELES GUI (PyQt5) for the RCP reaching analysis pipeline.
Screens live in `docs/gui_design/*.png`. This file records the **decisions** behind them.
If the images and this file disagree, this file wins.

Sources: current `VELES.py`, `VELES_gui.py`, `pipeline_hierarchy.py`, RCP docs (rcp-reaching-analysis.readthedocs.io), design canvas "VELES GUI".

---

## 0. How to work on this (for Claude Code)

- The developer (Keondre) is learning. Act as a tutor: explain the concept and Qt pattern, show a small example, let him write the real code, then review it. Don't write whole features unasked.
- Work milestone by milestone (§9). Plan first, wait for OK, commit after each milestone.
- Keep logic out of widgets. Planner and data code must be plain Python with tests and no Qt imports.
- Anything marked **OPEN** in §10 needs an answer from Keondre, not a guess.

---

## 1. Goals

1. Replace the single-window GUI with: **Home**, **Runs**, **Pipeline**, and a 5-step **New run** wizard.
2. Run scripts in **dependency order per session**, rerunning only what is missing, stale or failed (by default).
3. Support **per-session condition selection** (BR_File indices), filtered by metadata columns.
4. Replace the console `input()` "type RUN" prompt with a confirmation on the Review step.
5. Show live progress, the current script and the log for running jobs.

Non-goals: editing pipeline scripts, cluster/remote execution, multi-user.

---

## 2. Screens

Global header on Home/Runs/Pipeline: logo + "VELES", underlined tabs (Home · Runs · Pipeline), repo path at right.
The wizard has its own header (VELES · New run · summary · Cancel) and a 5-step stepper. Each finished step can be clicked to go back.

### 2.1 Home
- **Hero:** logo, title, version (from git tag), one-line description, buttons: **New run** (primary), All runs, How the pipeline works.
- **Running now:** run id, animal, #sessions, #scripts, start time, current script + session, session progress bar, elapsed time. Link to Runs.
- **Recent runs:** last 5 runs (id, animal, sessions, scripts, started, duration, status).
- **Data locations** panel:
  - Input data root: from `config/params.yaml` (+ `machines.yaml`), shown as `…\<animal>`, with a Browse button.
  - Output folder: default `<data_root>\results`. A checkbox allows a custom folder (input + Browse, disabled until unticked).
  - Run logs folder: `<repo>\logs`.
  - Checks shown as pills: input folders `Intan`, `Blackrock`, `Video`, `Metadata`, `Impedances` exist; `data_status_reaching.csv` found; number of sessions with a metadata CSV; CUDA available + conda env name.
- **Version history:** from git tags (version, one-line change, date).

### 2.2 Runs
- Filter buttons with counts: All · Running · Complete · Needs review · Failed · Stopped. Search box. Back to home, New run.
- Table: Run · Animal · Sessions · Scripts · Started · Duration · Status (+ short note, e.g. which script failed).
- **Expanding a row** shows three columns:
  1. **Sessions** with a status each (Complete / Running / Queued / Skipped / Failed).
  2. **Scripts** for the current session: up to date (skipped) / done + duration / running + elapsed / queued. The running script also shows progress by BR_File ("BR_File 3 of 5").
  3. **Log**: tail of `logs/run_YYYYMMDD_HHMMSS.log`, plus buttons: Open log file, Open figures, Show in folder, Copy path, Stop after this script.
- **Needs review** status: the run included `inspect_kinematics_trajectories.py`. Note: "Check the kinematics plots, then update manual_trial_remove.csv".

### 2.3 Pipeline
- Title "How the pipeline works" and Back to home.
- Shows the pipeline schematic image and text **from the repo's README.md**, rendered at runtime (not copied into the GUI).

### 2.4 Wizard step 1 — Animal
- One card each for Nike, Ada, Bert (radio). Card shows data-root path, session count (or "Not scanned" until selected), last run, status-CSV found.
- Note: only sessions with `Metadata/<session>_metadata.csv` are listed.

### 2.5 Wizard step 2 — Sessions
- Left, **ways to select** (each adds to the selection):
  - Range: from/to session.
  - By pipeline status: `Missing or stale` / `Failed` / `Up to date` for a chosen step.
  - By Location (from `data_status_reaching.csv`).
  - Paste a list.
  - Select all / Clear.
- Right, **session table:** checkbox, Session, # conditions (BR files), status strip, last processed. Counter "N of M selected". Filter box.
  - Status strip columns: OCR, DLC, VOG, NPRW, SHIFT, UA, ALIGN, PERI.
  - Strip states: up to date / stale / failed / not run / **n/a** (e.g. VOG with no `VOG_File`).
- Flag sessions with missing raw files (e.g. "2 raw files missing") based on `Intan_File`, `Video_File`, `VOG_File`, BR file existence.

### 2.6 Wizard step 3 — Conditions
- Mode: **Run all conditions for every selected session** or **Filter conditions**.
- Filters use the metadata CSV columns exactly:
  `Channels`, `Stim_Frequency_Hz`, `Current_uA`, `Depth_mm`, `Stim_Duration_ms`, `UA_port`, `Movement_Trigger`, `Delay`.
  - Toggle chips for columns with few distinct values: `Channels`, `Stim_Frequency_Hz`, `Stim_Duration_ms`, `UA_port`, `Movement_Trigger`.
  - Min/max inputs for numeric ranges: `Current_uA`, `Depth_mm`, `Delay`.
  - Chip options come from the **distinct values in that column across the selected sessions**.
- Right, **matching conditions**, one row per session: `matched / total`, plus one chip per BR_File:
  - **black** = will run
  - **outlined** = matches, but outputs already up to date, so skipped (unless "rerun all")
  - **grey** = filtered out
  - Clicking a chip includes/excludes it **for that session only**.
- A session with 0 matches shows "No match — this session will be skipped".
- Result: `dict[session, list[int]]` (see §3).

### 2.7 Wizard step 4 — Scripts
- Left, **"What do you want to produce?"** — the script catalog grouped Preprocessing / Analysis / Nikita scripts.
  - The user ticks **targets**.
  - Prerequisites are auto-ticked, locked, and tagged "needed".
  - `UA_BR_analysis` shows an **mf | ssmf** toggle; **ssmf is the default**; it's one choice because both write the same status column. (Double Check this: point explicitly in the code where this is)
  - `inspect_kinematics_trajectories.py` is tagged **manual review**.
- **Prerequisite policy** (radio):
  - Rerun what's missing, stale or failed (default)
  - Rerun all prerequisites
  - Selected only, no checks
- Right, **run plan grid**: rows = steps in dependency order, columns = sessions.
  - Cell states: `run` (target) · `missing` · `stale` · `failed` · `rerun` (an earlier step reruns) · `skip` (up to date) · `n/a` · hatched = session skipped.
  - Footer: runs per session; total script runs; notes such as "NRR_RW012: manual_trial_remove.csv edited after extract_peri ran".

### 2.8 Wizard step 5 — Review
- Summary rows with Edit links: Animal, Sessions (will run / skipped + reason), Conditions (count + filter summary), Scripts (targets + prerequisites + UA variant), Plan (runs per session, total), Output (folder + log path).
- **Confirm before starting:** one checkbox per stale/failed/missing/CSV-changed warning (replaces `prompt_dependency_warning`). **Start run** is disabled until all are ticked.
- **If a script fails:** stop that session and continue the others (default), or stop the whole run.
- Start run → back to Home; the run appears in Running now.

---

## 3. Data model (plain Python, no Qt)

```python
@dataclass
class Condition:          # one metadata CSV row
    br_file: int
    values: dict[str, str]   # all columns, raw strings

@dataclass
class Session:
    name: str; location: str
    conditions: list[Condition]
    status: dict[str, str]    # row from data_status_reaching.csv

@dataclass
class RunRequest:
    animal: str
    conditions: dict[str, list[int]]   # session -> BR_File indices (replaces global process_only)
    targets: list[str]                 # scripts the user ticked
    ua_variant: Literal["ssmf", "mf"] = "ssmf"
    policy: Literal["needed", "all", "none"] = "needed"
    on_failure: Literal["skip_session", "stop_run"] = "skip_session"
    output_root: Path | None = None    # None -> <data_root>/results

class CellState(Enum):
    RUN, MISSING, STALE, FAILED, RERUN, SKIP, NA, SESSION_SKIPPED

@dataclass
class Plan:
    steps: list[str]                               # dependency order
    cells: dict[tuple[str, str], CellState]        # (session, step) -> state
    warnings: list[Warning]                        # shown on Review
```

`run_scripts(...)` must accept `conditions: dict[str, list[int]]` instead of `process_only: list[int]`.

---

## 4. Dependency map (corrected)

Keep `SCRIPT_STATUS_COLUMNS` and `COLUMN_DEPENDENCIES` in `pipeline_hierarchy.py`, with these changes:

| Change | Why |
|---|---|
| Add `"plot_complete_shaded_BT.py": "plot_complete_shaded"` with dependency `["extract_peri"]` | Documented "Peri-Stim Firing Rate Plots" step; currently runs with no checks |
| `extract_peri` depends on `["make_aligned"]` (unchanged) + **file input** `config/manual_trial_remove.csv` | `compute_shifts` is already an indirect ancestor via `UA_BR_analysis`/`make_aligned`; decided 2026-09-24 to leave it out as a direct edge — no functional difference, avoids a redundant edge in the map |
| `manual_inspection` (inspect_kinematics) stays `["extract_peri"]` but is flagged `manual=True` | It's a human curation step (see §5.4) |
| Add `plot_firing_rates.py` and `plot_bin_counts_per_target.py` to the script catalog | Already in the map; just not selectable yet (see §10 #6) |
| `UA_BR_analysis_mf.py` / `_ssmf.py` are mutually exclusive | Same status column |
| VOG is optional per session | Docs: "if present" |

Add a per-step `outputs(session, br) -> list[Path]` for steps that write per-BR files. Known so far
(checked the scripts directly, 2026-09-24):
- `extract_peri`: `results/checkpoints/PeriStim/**/peristim__<intan>__BR_<idx>.npz`
- `inspect_kinematics`: `results/figures/kinematics_trajectories/<condition>/<target>/BR_<idx>_xy.jpg`
- `make_aligned`: `aligned__<intan_filename>__Intan_<idx:03d>__BR_<idx:03d>.npz` (per-BR — has its own
  `outputs()` entry)
- `NPRW_Intan_analysis` and `UA_BR_analysis`: **session-level, not per-BR** —
  `rates__<session.name>__bin<BIN_MS>ms_sigma<SIGMA_MS>ms.npz`, one file per session covering all
  conditions. §5 rule 3 (per-condition skip) does not apply to these two steps — no BR-level
  staleness check is possible, only session-level.
- `OCR`: per input video file (`<video_stem>_ocr.csv`), not BR-indexed by number.

---

## 5. Planner rules (`planner.py`)

Input: `RunRequest`, sessions, status rows, file mtimes. Output: `Plan`. Must be deterministic and fully unit-tested.

1. **Close over prerequisites:** steps = targets ∪ all their ancestors. Then topologically sort (a step always comes after its prerequisites). Use catalog order to break ties.
2. **Per session, for each step in order, pick a state:**
   - `NA` if the step's input doesn't exist for the session (e.g. VOG with no `VOG_File`).
   - `RUN` if it's a target.
   - `MISSING` if the status cell is empty; `FAILED` if `FAIL`; treat `IN-PROGRESS` as `FAILED` (interrupted).
   - `STALE` if any parent's timestamp is > 2 s newer than this step's, or a file input (e.g. `manual_trial_remove.csv`) changed after this step ran.
   - `RERUN` if any **ancestor of this step** is planned to run for this session (cascade). Ancestors only, not every earlier step: a missing `DLC` must not rerun `NPRW_Intan_analysis`, which doesn't depend on it.
   - otherwise `SKIP`.
   - With policy `all`, every prerequisite is `RERUN`. With policy `none`, only targets run and no checks happen.
3. **Per-condition skip:** for steps with per-BR outputs, a BR_File whose outputs are newer than all of its inputs can be dropped from that session's list (shown as outlined chips in §2.6).
4. **`DONE` without a timestamp** can't be checked for staleness. Treat it as up to date, but add a warning.
5. **Sessions with 0 selected conditions** → `SESSION_SKIPPED`.
6. `MISSING`, `STALE` and `FAILED` each produce a `PlanWarning` for the Review step (§2.8 checkboxes), and so does `DONE` without a timestamp. `RERUN` doesn't produce one, because it follows from another warning. `NA` and `SESSION_SKIPPED` don't either; the Review summary lists them instead. (The class is `PlanWarning` so it doesn't shadow Python's built-in `Warning`.)

### 5.4 Manual curation loop
Order: extract_peri → inspect_kinematics (plots) → human edits `config/manual_trial_remove.csv` → extract_peri reruns → downstream reruns.
- A run that includes inspect_kinematics ends as **Needs review**.
- The CSV is shared across sessions. Compare only **that session's rows** (by content hash or last-edit time), otherwise one edit marks every session stale (**OPEN**: CSV columns).

---

## 6. Execution

- Keep `PipelineWorker` on a `QThread`. Communicate only through signals: `session_started`, `step_started(session, script)`, `step_progress(session, script, br_done, br_total)`, `step_finished(session, script, ok, duration)`, `log_line`, `finished`.
- **No `input()` anywhere in the run path.** Confirmations happen before the run (§2.8).
- Run each script with the session's BR list. Update `data_status_reaching.csv` exactly as scripts do now (timestamp / FAIL / IN-PROGRESS).
- Log to `logs/run_YYYYMMDD_HHMMSS.log`. Also save the `RunRequest` + `Plan` as `logs/run_…json`, so the Runs page can rebuild history.
- **Stop after this script:** set a flag the worker checks between scripts. Never kill in the middle of a script.
- Only one run at a time for now. If two copies of VELES run at once, they both write `data_status_reaching.csv` (**OPEN**: add a lock file).

---

## 7. Visual style

- **Fonts:** Geist (UI), Geist Mono (labels, paths, ids, numbers). Bundle the `.ttf` files and load them with `QFontDatabase.addApplicationFont`; otherwise Qt falls back to Segoe UI. (Seguo UI is fine for a first step)
- **Colours:**

| Token | Hex | Use |
|---|---|---|
| ground | `#EEEEEE` | window background |
| surface | `#FFFFFF` | panels, header |
| subtle | `#F6F6F6` | table header rows, code blocks |
| line | `#D4D4D4` / `#E5E5E5` | panel borders / row dividers |
| ink | `#111111` | text, primary buttons, selected chips, running |
| muted | `#5A5A5A` | secondary text |
| faint | `#8C8C8C` | disabled |
| stale | `#FFF0BF` bg, `#6E5200` text | stale / needs review |
| fail | `#FCE3E4` bg, `#B42318` text | failed |

- **Shapes:** panels 10 px radius, buttons 6 px, status tags 4 px with a small square dot.
- **Colour is never the only signal:** every status shows its word too.
- **Tabs:** 2 px underline on the active tab.
- Put all styling in one `veles.qss`, not in widget code.

---

## 8. Suggested module layout

```
veles/
  app.py            # QApplication, fonts, stylesheet, MainWindow
  data.py           # read params, sessions, metadata, status CSV, output files (no Qt)
  planner.py        # §5 (no Qt)
  hierarchy.py      # dependency map (moved from pipeline_hierarchy.py) (Should this be a copy or not?)
  runner.py         # PipelineWorker + run records
  ui/
    main_window.py  # header + QStackedWidget
    home.py  runs.py  pipeline_page.py
    wizard/ animal.py sessions.py conditions.py scripts.py review.py
    widgets/ status_pill.py chip.py plan_grid.py
  veles.qss
tests/
  test_planner.py  test_data.py
```

---

## 9. Milestones (learning order)

1. **Planner + tests.** Topological sort, state rules, cascade, the CSV-changed case. Use made-up status rows in the tests. No GUI.
2. **Data layer.** Sessions, metadata, filter options, output-file checks, per-session conditions dict.
3. **Shell.** MainWindow, header tabs, QStackedWidget, empty pages, stylesheet + fonts.
4. **Wizard:** Animal → Sessions (`QAbstractTableModel`) → Conditions (filters → dict) → Scripts (plan grid from the planner) → Review (warnings gate Start).
5. **Runner + Runs page.** Signals, live progress, log tail, run records, Stop after this script.
6. **Home + Pipeline.** Running now, recent runs, data checks, README rendering.
7. **Polish.** Empty/error states, keyboard navigation, remember window size.

Each milestone ends with: tests pass, app launches, commit.

---

## 10. OPEN questions

1. `Channels`: a count (`4`) or a list of channel numbers (`1;5;9`)? This decides the filter type.
  ANS: `Channels` is a count.
2. `Movement_Trigger` values (Yes/No? 1/0?) and `Delay` units.
  ANS: `Movement_Trigger` is IR, -, or key. `Delay` units are 0,50,100 so unitless for now but probably in ms. Will double checked
3. `manual_trial_remove.csv` columns — can rows be matched to a session?
  ANS: check the csv
4. Output file patterns for per-BR steps other than extract_peri / inspect_kinematics.
  ANS: check the codebase
5. Do analysis scripts take a peri-stim category (stim/control reaches, target A/B, at_rest, Grasp, IMU, continuous_stim)? If so, the Scripts step needs a category filter.
  ANS: No category argument. Checked `plot_peri_stim_raster.py`: it globs the whole monkey's
  `PeriStim` output (`PERI_ROOT.rglob("peristim__*.npz")`, not scoped to one session) and filters
  by one global BR-index list read from `config/params.yaml` (`PARAMS.preprocessing.process_only`,
  line 12 / used line 1398). There is no per-session BR dict today — this is the exact global-vs-
  per-session gap Goal #3 is meant to fix, but the scripts themselves aren't session-scoped.
  DECISION (2026-09-24): per-session Condition filtering (§2.6) applies only through `extract_peri`
  and earlier. Analysis-script steps (raster, group responses, overlays, plateau, RSA, shaded_BT,
  peak_csv_summaries, plot_FRs, plot_bin_counts) are **out of scope for per-session condition
  selection** — they run over whatever the selected sessions already produced, with no per-session
  BR filter in the Plan grid. Revisit if/when those scripts are changed to accept a per-session
  scope.
6. `plot_firing_rates.py`, `plot_bin_counts_per_target.py`: add to the catalog or remove from the map?
  ANS: add to catolog. Note: already mapped in `pipeline_hierarchy.py` (`plot_FRs`, `plot_bin_counts`),
  no hierarchy.py change needed, just add to the wizard catalog. The day-one script catalog is
  `VELES_gui.py:SCRIPT_CATALOG` (lines 103-124) — Preprocessing (11, incl. `analyze_lfp_bands.py`),
  Analysis (7, incl. `plot_complete_shaded_BT.py`), Nikita Scripts (2). `VELES.py`'s `SCRIPTS` list
  is just one person's run config (mostly commented out) and is NOT the catalog source.
7. Multiple copies of VELES at once: add a lock file on `data_status_reaching.csv`?
  ANS: Yes. DECISION (2026-09-24): advisory lock file `data_status_reaching.csv.lock` (pid + start
  time), written by the runner before its first status write, removed on normal end and on crash
  cleanup. A second run checks for it (lock exists + pid still alive) and refuses to start / warns,
  rather than silently racing writes to the CSV. Build in Milestone 5 (Runner).
8. Conda env name to show on Home. ANS: pipeline
