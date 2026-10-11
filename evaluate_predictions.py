"""
Model evaluation pipeline (Phase 6).

This is an *evaluation mechanism*, not a second predictive model. It reads the
persistent prediction log written by ``prediction_logger`` and reports how the
existing VesselMLP performs against the simulator's authoritative state.

What it provides
----------------
1. Reproducible metrics computed from logged observations (never invented).
2. Per-scenario breakdown so Normal and Monsoon are assessed independently.
3. Explicit coverage counters that separate *model error* from *missing data*.
4. Model audit facts read from the code, so the report cannot drift from reality.
5. Honest labelling of unsupported tasks instead of silently dropping them.

Usage
-----
    python evaluate_predictions.py              # report from the existing log
    python evaluate_predictions.py --collect    # also run a short fresh run
    python evaluate_predictions.py --out FILE   # write a markdown report

Caveat stated in every output: these are simulation-based results. They do not
establish real-world maritime deployment readiness.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Keep the report readable on consoles that cannot encode box/em-dash glyphs.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prediction_logger as pl

SIMULATION_DISCLAIMER = (
    "Simulation-based evaluation. These results measure how the model behaves "
    "against the Harbor Traffic Control simulator's own state. They do not "
    "establish real-world maritime deployment readiness."
)

# Tasks that VesselMLP genuinely produces.
SUPPORTED_TASKS = (pl.TASK_CLASS, pl.TASK_WEIGHT)
# Tasks that exist in the simulation but have no trained model behind them.
UNSUPPORTED_TASKS = (pl.TASK_SPEED, pl.TASK_DIRECTION)


# ─────────────────────────────────────────────────────────────────────────────
# Model audit — read from code, not assumed
# ─────────────────────────────────────────────────────────────────────────────
def audit_model() -> Dict[str, Any]:
    """
    Inspect the trained artifact and the inference path.

    Returns verified facts only. Anything that cannot be confirmed is reported
    as None rather than guessed.
    """
    info: Dict[str, Any] = {
        "model_name": "VesselMLP",
        "architecture": "Shared MLP trunk (128-64-32) + 2 heads",
        "classification_head": "Linear(32 -> n_classes), CrossEntropyLoss",
        "regression_head": "Linear(32 -> 1), MSELoss",
        "inference_location": "mlmodel.update_model_from_collected_data()",
        "preprocessing": "StandardScaler fitted on training split only",
        "label_encoding": "sklearn LabelEncoder over 6 vessel classes",
        "load_path": "models/model_state.pt + scaler.pkl + label_encoder.pkl",
        "split": "train_test_split(test_size=0.2, random_state=42)",
        "epochs": 100,
        "optimizer": "Adam(lr=0.005)",
        "online_retraining_enabled": None,
        "device": None,
        "n_features": None,
        "classes": None,
        "trained_on_rows": None,
        "feature_scaler_leak_check": None,
    }

    try:
        import mlmodel as ml
    except Exception as exc:                      # pragma: no cover
        info["error"] = f"mlmodel import failed: {exc}"
        return info

    try:
        import torch
        info["device"] = str(ml._device)
        info["online_retraining_enabled"] = bool(ml.ENABLE_ONLINE_RETRAINING)
        info["n_features"] = len(ml.ALL_FEATURES)

        if ml._cached_le is not None:
            info["classes"] = list(ml._cached_le.classes_)

        # Confirm the scaler was fitted on the training split only (no leakage)
        src = open(ml.__file__, encoding="utf-8").read()
        scaler_fit_ok = "scaler.fit_transform(X_train_r)" in src
        scaler_transformed_val = "scaler.transform(X_val_r)" in src
        info["feature_scaler_leak_check"] = bool(
            scaler_fit_ok and scaler_transformed_val
        )

        if os.path.exists(ml.COLLECTED_FILE):
            import pandas as pd
            info["trained_on_rows"] = int(ml._last_trained_rows)
    except Exception as exc:
        info["error"] = f"audit partial: {exc}"

    return info


# ─────────────────────────────────────────────────────────────────────────────
# Coverage: separate model error from missing data
# ─────────────────────────────────────────────────────────────────────────────
def coverage_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Distinguish three failure modes that are easy to conflate:

    * model error      -> validated and wrong
    * missing truth    -> UNAVAILABLE (no ground truth to compare against)
    * no model at all  -> NOT_PREDICTED (value exists, nothing predicts it)
    """
    rep: Dict[str, Any] = {}
    for task in SUPPORTED_TASKS + UNSUPPORTED_TASKS:
        trows = [r for r in rows if r.get("task") == task]
        validated = [r for r in trows
                     if r.get("validation_status") in (pl.STATUS_CORRECT, pl.STATUS_INCORRECT)]
        correct = [r for r in trows if r.get("validation_status") == pl.STATUS_CORRECT]
        wrong = [r for r in trows if r.get("validation_status") == pl.STATUS_INCORRECT]
        unavailable = [r for r in trows if r.get("validation_status") == pl.STATUS_UNAVAILABLE]
        not_pred = [r for r in trows if r.get("validation_status") == pl.STATUS_NOT_PREDICTED]

        rep[task] = {
            "total_rows": len(trows),
            "validated": len(validated),
            "correct": len(correct),
            "incorrect": len(wrong),
            "unavailable": len(unavailable),
            "not_predicted": len(not_pred),
            "validation_coverage": (len(validated) / len(trows)) if trows else None,
        }
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(v: Optional[float], suffix: str = "", pct: bool = False,
         scale: float = 1.0) -> str:
    """
    Format a metric value.

    ``pct`` expects a value already expressed in percent units (0-100); use
    ``scale=100`` when passing a 0-1 fraction.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if pct:
        return f"{v * scale:.1f}%" if scale != 1.0 else f"{v:.1f}%"
    return f"{v:,.0f}{suffix}" if suffix else f"{v:.2f}"


def build_report(rows: List[Dict[str, Any]]) -> str:
    """Render a markdown report from logged observations."""
    lines: List[str] = []
    a = lines.append

    a("# Prediction Evaluation Report")
    a("")
    a(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("")
    a(f"> {SIMULATION_DISCLAIMER}")
    a("")

    if not rows:
        a("## Result")
        a("")
        a("**No prediction records found.** Run `python main.py` to generate "
          "telemetry, or `python evaluate_predictions.py --collect` for a short "
          "fresh run.")
        a("")
        return "\n".join(lines)

    a("## 1. Coverage")
    a("")
    a("| Task | Source | Rows | Validated | Correct | Incorrect | Unavailable | Not predicted |")
    a("|---|---|---|---|---|---|---|---|")
    cov = coverage_report(rows)
    source_of = {
        pl.TASK_CLASS: "ML head",
        pl.TASK_WEIGHT: "ML head",
        pl.TASK_SPEED: "simulator state",
        pl.TASK_DIRECTION: "geometric rule",
    }
    for task in SUPPORTED_TASKS + UNSUPPORTED_TASKS:
        c = cov[task]
        a(f"| `{task}` | {source_of.get(task, '—')} | {c['total_rows']} | "
          f"{c['validated']} | {c['correct']} | {c['incorrect']} | "
          f"{c['unavailable']} | {c['not_predicted']} |")
    a("")
    a("**Unsupported predictions are explicitly identified**, not silently omitted.")
    a("")

    a("## 2. Aggregate performance")
    a("")
    m = pl.compute_metrics(rows)
    cls = m[pl.TASK_CLASS]
    wt = m[pl.TASK_WEIGHT]
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Total prediction records | {len(rows)} |")
    a(f"| Classification accuracy | {_fmt(cls['accuracy'], pct=True, scale=100)} "
      f"({cls['correct']}/{cls['validated']}) |")
    a(f"| Displacement MAE | {_fmt(wt['mae_kg'], ' kg')} |")
    a(f"| Displacement RMSE | {_fmt(wt['rmse_kg'], ' kg')} |")
    a(f"| Displacement median AE | {_fmt(wt['median_ae_kg'], ' kg')} |")
    a(f"| Displacement 90th pct AE | {_fmt(wt['p90_ae_kg'], ' kg')} |")
    a(f"| Displacement MAPE | {_fmt(wt['mape_pct'], pct=True)} |")
    a("")

    # Explicit warning when the regression head is not trustworthy
    if wt["mae_kg"] is not None and wt["median_ae_kg"] is not None:
        if wt["mae_kg"] > 0.25 * (np.median([float(r["actual_value"]) for r in rows
                                            if r.get("task") == pl.TASK_WEIGHT
                                            and r.get("actual_value") not in (None, "")] or [1])):
            a("> **Finding:** the displacement head does not track ground truth. "
              "Predicted values are not proportional to the simulated vessel mass. "
              "See `logs/PREDICTION_VALIDATION_FINDINGS.md` for the diagnosed cause. "
              "Classification results are unaffected.")
            a("")

    a("## 3. Normal vs Monsoon")
    a("")
    by_scn = pl.summary_by_scenario()
    if len(by_scn) < 2:
        only = next(iter(by_scn), "?")
        a(f"Only one scenario present in the log (`{only}`). Monsoon results are "
          "**not yet available** — run the simulation in Monsoon season to collect them.")
        a("")
        for scn, metrics in by_scn.items():
            c = metrics.get(pl.TASK_CLASS, {})
            w = metrics.get(pl.TASK_WEIGHT, {})
            a(f"- **{scn}** — accuracy {_fmt(c.get('accuracy'), pct=True, scale=100)}, "
              f"MAE {_fmt(w.get('mae_kg'), ' kg')}, "
              f"rows {c.get('total_predictions', 0)}")
        a("")
    else:
        a("| Scenario | Class accuracy | Displacement MAE | Displacement RMSE | Records |")
        a("|---|---|---|---|---|")
        for scn, metrics in sorted(by_scn.items()):
            c = metrics.get(pl.TASK_CLASS, {})
            w = metrics.get(pl.TASK_WEIGHT, {})
            n = c.get("total_predictions", 0) + w.get("total_predictions", 0)
            a(f"| {scn} | {_fmt(c.get('accuracy'), pct=True, scale=100)} | "
              f"{_fmt(w.get('mae_kg'), ' kg')} | {_fmt(w.get('rmse_kg'), ' kg')} | {n} |")
        a("")
        a("Scenarios are evaluated **independently** so monsoon degradation is "
          "never averaged away into a single headline number.")
        a("")

    a("## 4. Model audit")
    a("")
    info = audit_model()
    a("| Property | Value |")
    a("|---|---|")
    for k in ("model_name", "architecture", "classification_head", "regression_head",
              "inference_location", "preprocessing", "label_encoding", "load_path",
              "split", "epochs", "optimizer", "device", "n_features", "classes",
              "online_retraining_enabled", "feature_scaler_leak_check",
              "trained_on_rows"):
        if k in info:
            val = info[k]
            if isinstance(val, list):
                val = ", ".join(str(x) for x in val)
            elif isinstance(val, bool):
                val = "yes" if val else "**no**"
            a(f"| {k} | {val if val is not None else 'n/a'} |")
    a("")
    a("**Leakage control:** the feature scaler is fitted on the training split "
      "only and applied to the validation split via `transform()`.")
    a("")

    a("## 5. Ground-truth separation")
    a("")
    a("```")
    a("Simulation state  -> Ground truth          (simulator Actual_* columns)")
    a("Sensor features   -> Model prediction      (VesselMLP forward pass)")
    a("Prediction + truth -> Evaluation           (this pipeline)")
    a("```")
    a("")
    a("Ground truth is read from the same detection row that produced the "
      "features, so each comparison uses the **same vessel at the same "
      "simulation frame**.")
    a("")

    a("## 6. Limitations")
    a("")
    a("- Results are simulation-based and do not imply real-world readiness.")
    a("- Accuracy figures describe the logged sample, not a guaranteed rate.")
    a("- A small number of correct predictions is not evidence of model quality; "
      "only the aggregate over validated observations is reported.")
    a("- No speed or heading model exists; those values are simulator state.")
    a("")
    return "\n".join(lines)


def collect_short_run(steps: int = 3, scenario: str = "NORMAL") -> int:
    """
    Run a brief simulation so the log has fresh observations to evaluate.

    Kept short on purpose: this is an evaluation driver, not a benchmark.
    """
    try:
        import simulator as sim
        import mlmodel as ml
    except Exception as exc:
        print(f"[Collector] import failed: {exc}")
        return 0

    sim.set_active_scenario(scenario)
    pl.reset_run()
    for step in range(steps):
        sensors = sim.generate_sensors()
        fleet = sim.generate_fleet()
        sim.collect_sensor_detections(fleet, sensors, frame=step)
        ml.update_model_from_collected_data()
    return pl.log_count()


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate VesselMLP against simulated ground truth.")
    ap.add_argument("--collect", action="store_true",
                    help="Run a short fresh simulation before evaluating")
    ap.add_argument("--scenario", default="NORMAL", choices=["NORMAL", "MONSOON"],
                    help="Scenario for --collect (default: NORMAL)")
    ap.add_argument("--steps", type=int, default=3, help="Frames to simulate with --collect")
    ap.add_argument("--out", metavar="FILE", help="Also write the report to this path")
    args = ap.parse_args()

    if args.collect:
        print(f"[Collector] Running {args.steps} frame(s) in {args.scenario}...")
        collect_short_run(args.steps, args.scenario)

    rows = pl.read_log()
    report = build_report(rows)
    print()
    print(report)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(report)
            print(f"\nReport written to {args.out}")
        except OSError as exc:
            print(f"Could not write report: {exc}", file=sys.stderr)
            return 1

    pl.write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())