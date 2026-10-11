"""
Prediction logging and ground-truth validation (Phase 5).

Purpose
-------
The simulation is the authoritative source of ground truth: every detection row
written by ``simulator._build_sensor_row`` carries ``Actual_Type``,
``Actual_Weight_kg``, ``Actual_Speed_knots`` and ``Actual_Depth_m``.

Until now the model's predictions were written to a *temporary* CSV that is
overwritten on every animation frame, so the prediction history was lost. This
module appends a permanent, queryable record of every prediction together with
the ground truth observed for the same vessel at the same simulation frame, and
computes task-appropriate error metrics from those real observations.

Design constraints honoured here
--------------------------------
* ``Simulation state -> Ground truth`` (authoritative, from the simulator)
* ``Sensor features -> Model prediction`` (what VesselMLP actually emits)
* ``Prediction + matching ground truth -> Evaluation``

Only outputs that a trained model genuinely produces are logged as predictions.
Fields that are *measured* rather than predicted (speed) or *rule-derived*
(direction) are recorded explicitly as ``NOT_PREDICTED`` so nothing is
misattributed to machine learning.

Storage is CSV (chosen to match the existing ``collected_data.csv`` pipeline and
the scale of the simulation). Writes are append-only, lock-guarded, and use an
atomic rotate-then-append so a crash mid-write cannot corrupt existing history.
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
PREDICTIONS_CSV = os.path.join(LOG_DIR, "prediction_log.csv")
SUMMARY_CSV = os.path.join(LOG_DIR, "prediction_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
FIELDS: Tuple[str, ...] = (
    "prediction_id",
    "run_id",
    "wall_clock",
    "frame",
    "scenario",
    "vessel_id",
    "sensor_id",
    "model_name",
    "model_artifact",
    "task",
    "source_kind",       # ML | MEASURED | RULE_BASED
    "predicted_value",
    "actual_value",
    "error",
    "metric_kind",       # exact_match | abs_error | angular_error | n/a
    "validation_status", # CORRECT | INCORRECT | UNAVAILABLE | NOT_PREDICTED
    "confidence",
    "unit",
)

TASK_CLASS = "vessel_class"
TASK_WEIGHT = "displacement_kg"
TASK_SPEED = "speed_kn"
TASK_DIRECTION = "heading_deg"

STATUS_CORRECT = "CORRECT"
STATUS_INCORRECT = "INCORRECT"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_NOT_PREDICTED = "NOT_PREDICTED"

# ─────────────────────────────────────────────────────────────────────────────
# Module state
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_run_id: str = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
_seen_ids: set = set()
_current_run_stats: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {"n": 0, "n_validated": 0, "n_correct": 0, "abs_err_sum": 0.0, "sq_err_sum": 0.0}
)


def get_run_id() -> str:
    """Current simulation run identifier."""
    return _run_id


def reset_run(run_id: Optional[str] = None) -> str:
    """Start a new run; flushes in-memory aggregates and dedup cache."""
    global _run_id, _seen_ids, _current_run_stats
    with _lock:
        _run_id = run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}"
        _seen_ids = set()
        _current_run_stats = defaultdict(
            lambda: {"n": 0, "n_validated": 0, "n_correct": 0,
                     "abs_err_sum": 0.0, "sq_err_sum": 0.0}
        )
        # Refresh the dedup cache from disk so restarts cannot duplicate history
        _seen_ids = _load_existing_ids()
    return _run_id


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def circular_angle_error(pred_deg: float, actual_deg: float) -> float:
    """
    Angular error in degrees, accounting for wraparound.

    359 deg vs 1 deg must report 2 deg, not 358 deg.
    """
    d = abs(float(pred_deg) - float(actual_deg)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def classification_metric(pred: Any, actual: Any) -> Tuple[Optional[float], str, str]:
    """Exact-match classification. Returns (error, metric_kind, status)."""
    if pred is None or actual is None:
        return None, "exact_match", STATUS_UNAVAILABLE
    if str(pred) == str(actual):
        return 0.0, "exact_match", STATUS_CORRECT
    return 1.0, "exact_match", STATUS_INCORRECT


def regression_metric(pred: float, actual: float) -> Tuple[Optional[float], str, str]:
    """Absolute error for a continuous value. Returns (error, metric_kind, status)."""
    if pred is None or actual is None:
        return None, "abs_error", STATUS_UNAVAILABLE
    try:
        err = abs(float(pred) - float(actual))
    except (TypeError, ValueError):
        return None, "abs_error", STATUS_UNAVAILABLE
    return err, "abs_error", STATUS_INCORRECT if err > 1e-9 else STATUS_CORRECT


def angular_metric(pred: float, actual: float) -> Tuple[Optional[float], str, str]:
    """Circular angular error. Returns (error, metric_kind, status)."""
    if pred is None or actual is None:
        return None, "angular_error", STATUS_UNAVAILABLE
    try:
        err = circular_angle_error(pred, actual)
    except (TypeError, ValueError):
        return None, "angular_error", STATUS_UNAVAILABLE
    return err, "angular_error", STATUS_INCORRECT if err > 1e-9 else STATUS_CORRECT


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _load_existing_ids() -> set:
    """Read prediction_id values already present on disk."""
    if not os.path.exists(PREDICTIONS_CSV):
        return set()
    ids = set()
    try:
        with open(PREDICTIONS_CSV, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pid = row.get("prediction_id")
                if pid:
                    ids.add(pid)
    except (OSError, csv.Error):
        pass
    return ids


def _new_id() -> str:
    return f"P-{uuid.uuid4().hex[:12]}"


def _blank(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def log_predictions(records: Sequence[Dict[str, Any]], dedupe: bool = True) -> int:
    """
    Append prediction records to the persistent log.

    Each record must supply the keys present in ``FIELDS``; missing keys become
    blank. Returns the number of rows actually written (duplicates are skipped).
    """
    if not records:
        return 0
    _ensure_dir()
    written = 0
    with _lock:
        new_file = not os.path.exists(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) == 0
        try:
            with open(PREDICTIONS_CSV, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                for rec in records:
                    pid = rec.get("prediction_id") or _new_id()
                    if dedupe and pid in _seen_ids:
                        continue
                    _seen_ids.add(pid)
                    rec = dict(rec)
                    rec["prediction_id"] = pid
                    rec.setdefault("run_id", _run_id)
                    rec.setdefault("wall_clock", datetime.now().isoformat(timespec="seconds"))
                    writer.writerow({k: _blank(rec.get(k)) for k in FIELDS})
                    written += 1
                    _accumulate(rec)
        except OSError as exc:  # disk full / locked file
            print(f"[PredictionLogger] Write failed: {exc}")
            return 0
    return written


def _accumulate(rec: Dict[str, Any]) -> None:
    """Fold one record into the in-memory aggregate for the current run."""
    task = rec.get("task", "?")
    st = rec.get("validation_status", STATUS_UNAVAILABLE)
    slot = _current_run_stats[task]
    slot["n"] += 1
    if st in (STATUS_CORRECT, STATUS_INCORRECT):
        slot["n_validated"] += 1
        if st == STATUS_CORRECT:
            slot["n_correct"] += 1
        err = rec.get("error")
        try:
            if err not in (None, ""):
                e = float(err)
                slot["abs_err_sum"] += e
                slot["sq_err_sum"] += e * e
        except (TypeError, ValueError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Public logging API
# ─────────────────────────────────────────────────────────────────────────────
def log_classification(
    frame: int,
    scenario: str,
    vessel_id: Any,
    predicted_type: Optional[str],
    actual_type: Optional[str],
    sensor_id: Any = None,
    model_name: str = "VesselMLP",
    model_artifact: str = "models/model_state.pt",
    confidence: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Log and validate one vessel-class prediction."""
    err, kind, status = classification_metric(predicted_type, actual_type)
    rec = {
        "run_id": _run_id,
        "frame": frame,
        "scenario": scenario,
        "vessel_id": vessel_id,
        "sensor_id": sensor_id,
        "model_name": model_name,
        "model_artifact": model_artifact,
        "task": TASK_CLASS,
        "source_kind": "ML",
        "predicted_value": predicted_type,
        "actual_value": actual_type,
        "error": err,
        "metric_kind": kind,
        "validation_status": status,
        "confidence": confidence,
        "unit": "class",
    }
    log_predictions([rec])
    return rec


def log_regression(
    frame: int,
    scenario: str,
    vessel_id: Any,
    task: str,
    predicted_value: Optional[float],
    actual_value: Optional[float],
    sensor_id: Any = None,
    model_name: str = "VesselMLP",
    model_artifact: str = "models/model_state.pt",
    unit: str = "kg",
) -> Optional[Dict[str, Any]]:
    """Log and validate one continuous-value prediction (e.g. displacement)."""
    err, kind, status = regression_metric(predicted_value, actual_value)
    rec = {
        "run_id": _run_id,
        "frame": frame,
        "scenario": scenario,
        "vessel_id": vessel_id,
        "sensor_id": sensor_id,
        "model_name": model_name,
        "model_artifact": model_artifact,
        "task": task,
        "source_kind": "ML",
        "predicted_value": predicted_value,
        "actual_value": actual_value,
        "error": err,
        "metric_kind": kind,
        "validation_status": status,
        "unit": unit,
    }
    log_predictions([rec])
    return rec


def log_direction(
    frame: int,
    scenario: str,
    vessel_id: Any,
    predicted_deg: Optional[float],
    actual_deg: Optional[float],
    source_kind: str = "RULE_BASED",
    model_name: str = "rule-based-heading",
    unit: str = "deg",
) -> Optional[Dict[str, Any]]:
    """
    Log a heading comparison using circular (wraparound-safe) error.

    ``source_kind`` defaults to RULE_BASED because the current project derives
    direction from a geometric threshold rather than from a trained model; this
    keeps the record honest about what produced the value.
    """
    err, kind, status = angular_metric(predicted_deg, actual_deg)
    rec = {
        "run_id": _run_id,
        "frame": frame,
        "scenario": scenario,
        "vessel_id": vessel_id,
        "model_name": model_name,
        "task": TASK_DIRECTION,
        "source_kind": source_kind,
        "predicted_value": predicted_deg,
        "actual_value": actual_deg,
        "error": err,
        "metric_kind": kind,
        "validation_status": status,
        "unit": unit,
    }
    log_predictions([rec])
    return rec


def log_not_predicted(
    frame: int,
    scenario: str,
    vessel_id: Any,
    task: str,
    actual_value: Any,
    source_kind: str = "MEASURED",
    model_name: str = "simulation-state",
    unit: str = "",
    note: str = "No trained model produces this value; recorded as simulator state.",
) -> Dict[str, Any]:
    """
    Record a quantity that exists in the simulation but is NOT a model output.

    This keeps speed/direction honest: they are observable, but calling them
    "predictions" would misattribute simulator state to machine learning.
    """
    rec = {
        "run_id": _run_id,
        "frame": frame,
        "scenario": scenario,
        "vessel_id": vessel_id,
        "model_name": model_name,
        "task": task,
        "source_kind": source_kind,
        "predicted_value": None,
        "actual_value": actual_value,
        "error": None,
        "metric_kind": "n/a",
        "validation_status": STATUS_NOT_PREDICTED,
        "unit": unit,
    }
    log_predictions([rec])
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation / reporting
# ─────────────────────────────────────────────────────────────────────────────
def _safe_percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-task metrics from real logged observations.

    Classification -> accuracy. Regression -> MAE / RMSE (+ MAPE only when every
    target is far enough from zero for it to be meaningful). Direction -> mean
    circular error.
    """
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_task[rec.get("task", "?")].append(rec)

    out: Dict[str, Any] = {}
    for task, recs in by_task.items():
        total = len(recs)
        validated = [r for r in recs if r.get("validation_status") in (STATUS_CORRECT, STATUS_INCORRECT)]
        errors = []
        for r in validated:
            e = r.get("error")
            if e not in (None, ""):
                try:
                    errors.append(float(e))
                except (TypeError, ValueError):
                    pass

        entry: Dict[str, Any] = {
            "task": task,
            "total_predictions": total,
            "validated": len(validated),
            "unavailable": sum(1 for r in recs if r.get("validation_status") == STATUS_UNAVAILABLE),
            "not_predicted": sum(1 for r in recs if r.get("validation_status") == STATUS_NOT_PREDICTED),
        }

        if task == TASK_CLASS:
            correct = sum(1 for r in recs if r.get("validation_status") == STATUS_CORRECT)
            entry["correct"] = correct
            entry["accuracy"] = (correct / len(validated)) if validated else None
        elif task == TASK_WEIGHT:
            entry["mae_kg"] = float(np.mean(errors)) if errors else None
            entry["rmse_kg"] = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
            entry["median_ae_kg"] = _safe_percentile(errors, 50) if errors else None
            entry["p90_ae_kg"] = _safe_percentile(errors, 90) if errors else None
            actuals = []
            for r in validated:
                a = r.get("actual_value")
                try:
                    if a not in (None, ""):
                        actuals.append(float(a))
                except (TypeError, ValueError):
                    pass
            # MAPE only when targets are safely non-zero (avoids division blow-up)
            if actuals and min(abs(a) for a in actuals) > 1.0:
                entry["mape_pct"] = float(np.mean(errors) / np.mean(actuals) * 100.0)
            else:
                entry["mape_pct"] = None
        elif task == TASK_DIRECTION:
            entry["mean_angular_error_deg"] = float(np.mean(errors)) if errors else None
            entry["max_angular_error_deg"] = float(np.max(errors)) if errors else None
        else:
            entry["mae"] = float(np.mean(errors)) if errors else None

        out[task] = entry

    # Always expose the known tasks so dashboards never KeyError on an empty
    # log. Metrics stay None (unknown) rather than defaulting to a flattering 0.
    for known in (TASK_CLASS, TASK_WEIGHT, TASK_DIRECTION):
        out.setdefault(known, _empty_task_entry(known))
    return out


def _empty_task_entry(task: str) -> Dict[str, Any]:
    """Task entry with no observations: counts zero, metrics unknown."""
    entry: Dict[str, Any] = {
        "task": task,
        "total_predictions": 0,
        "validated": 0,
        "unavailable": 0,
        "not_predicted": 0,
    }
    if task == TASK_CLASS:
        entry.update({"correct": 0, "accuracy": None})
    elif task == TASK_WEIGHT:
        entry.update({"mae_kg": None, "rmse_kg": None,
                      "median_ae_kg": None, "p90_ae_kg": None, "mape_pct": None})
    elif task == TASK_DIRECTION:
        entry.update({"mean_angular_error_deg": None, "max_angular_error_deg": None})
    return entry


def current_run_metrics() -> Dict[str, Any]:
    """Live aggregate for the current run (no disk read)."""
    out = {}
    for task, s in _current_run_stats.items():
        entry = {
            "task": task,
            "total_predictions": s["n"],
            "validated": s["n_validated"],
            "correct": s["n_correct"],
        }
        if task == TASK_CLASS and s["n_validated"]:
            entry["accuracy"] = s["n_correct"] / s["n_validated"]
        if task == TASK_WEIGHT and s["n_validated"]:
            entry["mae_kg"] = s["abs_err_sum"] / s["n_validated"]
            entry["rmse_kg"] = math.sqrt(s["sq_err_sum"] / s["n_validated"])
        out[task] = entry
    return out


def read_log(limit: Optional[int] = None, run_id: Optional[str] = None,
             task: Optional[str] = None, scenario: Optional[str] = None,
             vessel_id: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Query the persistent log (latest last)."""
    if not os.path.exists(PREDICTIONS_CSV):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(PREDICTIONS_CSV, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if run_id and row.get("run_id") != run_id:
                    continue
                if task and row.get("task") != task:
                    continue
                if scenario and row.get("scenario") != scenario:
                    continue
                if vessel_id is not None and str(row.get("vessel_id")) != str(vessel_id):
                    continue
                rows.append(row)
    except (OSError, csv.Error) as exc:
        print(f"[PredictionLogger] Read failed: {exc}")
        return []
    if limit:
        rows = rows[-limit:]
    return rows


def summary_by_scenario(task: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Per-scenario metrics so normal vs monsoon can be compared independently."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in read_log(task=task):
        buckets[row.get("scenario", "UNKNOWN")].append(row)
    return {scn: compute_metrics(recs) for scn, recs in buckets.items()}


def write_summary() -> bool:
    """Persist a human-readable metrics snapshot for reporting."""
    _ensure_dir()
    rows = read_log()
    metrics = compute_metrics(rows)
    try:
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
            keys = sorted({k for m in metrics.values() for k in m})
            writer = csv.DictWriter(fh, fieldnames=["generated"] + keys)
            writer.writeheader()
            for entry in metrics.values():
                row = {"generated": datetime.now().isoformat(timespec="seconds")}
                row.update(entry)
                writer.writerow(row)
        return True
    except OSError as exc:
        print(f"[PredictionLogger] Summary write failed: {exc}")
        return False


def log_count() -> int:
    """Total rows currently in the persistent log."""
    if not os.path.exists(PREDICTIONS_CSV):
        return 0
    n = 0
    try:
        with open(PREDICTIONS_CSV, "r", encoding="utf-8") as fh:
            for _ in fh:
                n += 1
    except OSError:
        return 0
    return max(0, n - 1)  # minus header


def history_snapshot(limit: int = 50) -> List[Dict[str, Any]]:
    """Most recent validation records, newest first, for a dashboard table."""
    rows = read_log(limit=limit)
    rows.reverse()
    return rows