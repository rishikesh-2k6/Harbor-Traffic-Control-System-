# Engineering Report — Harbor Traffic Control System Upgrade

All work was completed against the existing repository. No module was rebuilt
from scratch, and no simulation behaviour was silently altered.

---

## A. Changes made

### Phase 2 — Crash notifications moved out of the simulation

| File | Change |
|---|---|
| `main.py` | Added `Typography`, `NotificationLevel`, `Notification`, `NotificationQueue`, `NotificationSidebar`, `install_exception_handler()`, and `notify_*()` helpers |
| `main.py` | Removed the `net_broken_banner` overlay from the viewport; replaced with a no-op stand-in so legacy call sites stay safe |
| `main.py` | Relocated the squall flash from a canvas overlay into the HUD bar |
| `main.py` | Network-collapse now emits a deduplicated `FATAL_ERROR` into the sidebar |
| `main.py` | Animation loop split into `master_loop` (guarded) + `_master_loop_body`, with a 30-frame circuit breaker |
| `main.py` | Replaced two silent `except: pass` blocks with reported warnings |

Behaviour: four severities, timestamps, expandable monospace traceback with a
copy button, duplicate suppression on `level:message:source`, thread-safe queue
with `root.after(400)` polling, `sys.excepthook` and `threading.excepthook`
routed to the panel, and `on_closing()` cancelling pending callbacks.

### Phase 3 — Grid-based responsive UI

| File | Change |
|---|---|
| `main.py` | Root layout migrated from nested `pack()` to `grid()` |
| `main.py` | Added `Typography` class defining nine consistent text styles |
| `main.py` | Added `_apply_sidebar_geometry()` so the viewport always keeps ≥ 75 % |
| `main.py` | Sidebar auto-collapses below 1700 px so the viewport is never squeezed |
| `main.py` | Header/HUD/footer unified into one navy control-station palette |

Verified: viewport proportion ≥ 75 % at every tested width (1024 → 3440 px).

### Phase 4 — Web dashboard themes

| File | Change |
|---|---|
| `dashboard.html` | Added dual-theme token blocks, theme toggle button, persistence, accessibility features |
| `networking_dashboard.html` | Same treatment |

Every hardcoded colour was converted to a CSS custom property. Verified
**0** unresolved `var()` references and **0** leftover white-alpha hardcodes in
either body. Both pages share one `localStorage` key so they stay in sync.

### Phase 5 — Prediction logging (left as-is at your request)

`prediction_logger.py`, `test_prediction_validation.py`, `logs/`.

### Phase 6 — Evaluation pipeline

| File | Change |
|---|---|
| `evaluate_predictions.py` | Reproducible report: coverage, aggregate metrics, per-scenario split, model audit, leakage check, limitations |
| `mlmodel.py` | `_log_validation()` hook logs every live inference against ground truth |

### Phase 7 — README

Fully rewritten; **49/49** factual claims machine-verified against the source.

### Phase 8/9 — Tests and quality

| File | Change |
|---|---|
| `test_integration.py` | 52 new integration and regression tests |
| `run_all_tests.py` | Single runner summarising all suites |
| `simulator.py` | Fixed `DtypeWarning`; narrowed a silent `except` |
| `network.py` | Narrowed over-broad handlers; corrected stale duty-cycle comments |
| `mlmodel.py` | Replaced silent `except: pass` with an explicit `UNKNOWN` fallback |
| `traffic_algo.py` | Fixed berth collision bug (see below) |

---

## B. Prediction validation

**Logged:** vessel class and displacement — the only two heads `VesselMLP` has.
**Ground truth:** the simulator's own `Actual_Type` / `Actual_Weight_kg`, written
onto the *same detection row* as the features, guaranteeing the same vessel at
the same simulation frame.
**Metrics:** exact-match accuracy, MAE, RMSE, median AE, 90th-percentile AE,
MAPE, circular angular error.
**Storage:** `logs/prediction_log.csv`, append-only, persists across restarts.
**View:** `python evaluate_predictions.py`, or `history_snapshot()` in the UI.

### Do speed and direction models exist?

**No.** Neither task has a trained model.

- Speed is *measured* by `simulator` and stored as `Actual_Speed_knots`.
- Direction is a geometric threshold on `vessel['angle']` in
  `simulator.get_vessel_direction()`.

Both are logged with `source_kind = MEASURED` / `RULE_BASED` and
`validation_status = NOT_PREDICTED`, so simulator state is never credited to ML.

### Measured accuracy

| Task | Result |
|---|---|
| Vessel class | **100 %** (23/23 validated) |
| Displacement | **MAE 22,713 kg · RMSE 36,121 kg · MAPE 97.7 %** |
| Monsoon run | 100 % class accuracy, MAE 22,703 kg |

**The displacement head does not work.** Cause diagnosed and documented in
`logs/PREDICTION_VALIDATION_FINDINGS.md`: `_retrain()` combines
`CrossEntropy + MSELoss × 1e-8`, but the weight term measures 6.3 × 10⁹ against
a classification term of 5.6, so the shared trunk receives only **0.6 %** of its
gradient from the regression head. Left unpatched per your instruction.

---

## C. ML documentation

| Model | Input | Output | Purpose | Status |
|---|---|---|---|---|
| `VesselMLP` (classification head) | 21 features | 6 vessel classes | LOC screening, berth assignment | **Verified working** (100 %) |
| `VesselMLP` (regression head) | 21 features | displacement (kg) | Informational | **Inaccurate** — diagnosed defect |

`network.py`, `auv.py` and `traffic_algo.py` are deterministic algorithms and
are **not** described as ML anywhere.

---

## D. Test results

### Passed — 75 tests

| Suite | Count | Result |
|---|---|---|
| `verify_environment.py` | 4 stages | PASS |
| `test_prediction_validation.py` | 23 | PASS |
| `test_integration.py` | 52 | PASS |
| **Total** | **75** | **ALL PASS** |

Static audits: README claims 49/49, dashboard tokens 0 unresolved vars, WCAG
contrast all pairs ≥ 4.5:1.

### Failed — none outstanding

One failure occurred during development and was fixed (below).

### Not executed

- **Tkinter GUI automation** — the desktop UI was verified by compilation and
  code inspection, not by automated clicks or screenshots.
- **Browser rendering** — no desktop browser was connected to this session, so
  theme switching was validated by static analysis rather than live rendering.
- **Long-duration stability** — no multi-hour soak test was run.

### Defect found and fixed by testing

`test_berths_are_not_double_assigned` failed and exposed a real pre-existing
bug: the fleet needs **22 berths** but only **13** exist, and `_assign_dock()`
fell back to `slots[0]`, stacking up to 13 speedboats onto the single `Dock-C`
berth.

Fixed by queuing instead of colliding — `_assign_dock()` returns `(None, None)`
when no berth is free, the vessel enters `HOLDING_FOR_BERTH` at zero speed, and
`release_dock()` / `clear_dock_assignments()` free berths. Two regression tests
cover it.

---

## E. Execution instructions

```bash
pip install numpy pandas matplotlib scikit-learn joblib torch networkx

python verify_environment.py       # dependency check
python main.py                     # launch simulation + telemetry server

python run_all_tests.py            # all 75 tests
python evaluate_predictions.py     # validation report
python evaluate_predictions.py --collect --scenario MONSOON
python diagnose_weight_head.py     # weight-head diagnostic
```

Dashboards (auto-opened, or manually):

- `http://127.0.0.1:8000/dashboard.html`
- `http://127.0.0.1:8000/networking_dashboard.html`

Controls: `Space` pause · `Ctrl+S` start · `Ctrl+R` reset · `⇄ Season` switch ·
`🌪 Squall` · `⚓ Re-Anchor` · sidebar tabs for notifications/network.

---

## F. Remaining issues

| # | Issue | Why it remains |
|---|---|---|
| 1 | **Displacement regression inaccurate** | You chose to leave the model unchanged. Fix is diagnosed and ready but not applied. |
| 2 | **No speed or heading model** | None exists. Building one needs a defined horizon and leakage controls. |
| 3 | **Small validation sample** | 23 validated samples from one short run. A longer run would firm up the accuracy figure. |
| 4 | **Berth capacity shortfall** | 13 berths for 22 vessels. Now queued safely, but adding berths is the real fix. |
| 5 | **GUI untested by automation** | Needs a display and a GUI automation tool. |
| 6 | **Prediction logging on the main thread** | Negligible at current rates, but could block the UI under heavy load. |
| 7 | **Presentation assets not updated** | `PRESENTATION_CONTENT.md` and the .pptx predate the new findings. |

---

## G. Recommended next steps

1. **Fix the displacement objective** — standardise or log-scale the regression
   target, combine losses with an explicit documented weight, retrain, and
   re-validate. The single highest-value item.
2. **Run a long evaluation** (30+ min per scenario) so accuracy rests on a
   meaningful sample rather than 23 observations.
3. **Reconcile the presentation** with the measured results — especially the
   weight finding, since a claimed-accuracy slide that the data contradicts is
   the biggest academic risk here.
4. **Add berths** so all 22 vessels can dock without queueing.
5. **Add a speed-prediction head** with a stated horizon and leakage controls.
6. **Move logging to a worker thread** and surface live validation metrics on
   the dashboards.
7. **Add GUI automation** to cover resize, theme toggle and shutdown behaviour.

---

## Definition of done

| Criterion | Status |
|---|---|
| Desktop interface uses available space effectively | Done — viewport ≥ 75 % at all widths |
| Crash notifications do not obstruct the simulation | Done — sidebar panel, overlays removed |
| Web dashboards have working light/dark themes | Done — verified, WCAG AA |
| Supported predictions logged vs real ground truth | Done — `prediction_logger.py` |
| Model evaluation reproducible | Done — `evaluate_predictions.py` |
| README accurately explains the project | Done — 49/49 claims verified |

One criterion is met **honestly rather than favourably**: the displacement model
does not work, and the documentation says so with reproducible evidence.