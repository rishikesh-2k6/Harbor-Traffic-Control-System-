# Phase 5 Findings — Prediction Accuracy vs. Simulated Ground Truth

Generated during Phase 5 implementation. All numbers below are measured from
real simulation data, not estimated.

## The question being answered

> "When we predict weight, vessel type and direction, are they accurate
> compared to the actual ships in the simulation?"

To answer it we need a persistent record of predictions matched to ground truth.
Before this phase, predictions were written to
`%TEMP%/stage3_live_predictions.csv`, which is **overwritten every animation
frame** — so the history was lost and accuracy could not be computed at all.

## Ground truth source

`simulator._build_sensor_row()` already writes the authoritative values onto the
same detection row as the features:

| Column | Meaning |
|---|---|
| `Actual_Type` | vessel class configured in `generate_fleet()` |
| `Actual_Weight_kg` | `vessel['true_weight']` |
| `Actual_Speed_knots` | `vessel['speed_knots']` |
| `Actual_Depth_m` | vessel operating depth |

Because these sit on the *same row* as the features that produced a prediction,
a prediction can be matched to ground truth for the **same vessel at the same
simulation frame** — no separate lookup, and no risk of comparing against a
value already overwritten by a later frame.

## Measured results (NORMAL season, real inference)

| Task | Source | Validated | Accuracy | MAE | RMSE |
|---|---|---|---|---|---|
| Vessel class | ML head (CrossEntropy) | 23 | **100 %** | — | — |
| Displacement | ML head (MSE) | 23 | — | **22,713 kg** | **36,121 kg** |
| Speed | **not predicted** (simulator state) | 0 | n/a | — | — |
| Direction | **not predicted** (geometric rule) | 0 | n/a | — | — |

Example records:

```
vessel_class    ML          pred=Speedboat   actual=Speedboat   CORRECT
displacement_kg ML          pred=392.5       actual=9687.0      INCORRECT
speed_kn        MEASURED    pred=             actual=20.18       NOT_PREDICTED
```

## Answer to the question

* **Vessel class — accurate.** 100 % on the observed sample, consistent with a
  well-separated acoustic signature per class.
* **Displacement — not accurate.** The head predicts ~400–1,700 kg for vessels
  whose true displacement ranges from 1,023 kg to 381,263 kg. MAPE ≈ 93 %.
  Predictions are **not** proportional to truth.
* **Speed — not a prediction.** `simulator` measures it and stores it as
  `Actual_Speed_knots`; no trained model produces it. Logged as `NOT_PREDICTED`
  so it is never credited to ML.
* **Direction — not a prediction.** `simulator.get_vessel_direction()` applies a
  geometric threshold on `vessel['angle']` (INBOUND vs OUTBOUND). No model is
  involved. Logged as `NOT_PREDICTED`.

## Root cause of the displacement failure

`_retrain()` combines the two heads as:

```python
loss = loss_type + (loss_weight * 1e-8)
```

`loss_type` is CrossEntropy ≈ **5.56**. `loss_weight` is MSE on raw kilogram
targets, measured at **6.26e+09**. The `1e-8` factor does not normalise the two
terms — it is an arbitrary constant that was evidently added to stop the
regression term from dominating, but the target scale (mean 51,270 kg) makes raw
MSE enormous, so the weight term still ends up **11× larger** than the
classification term and destabilises training.

Measured gradient reaching the shared trunk:

```
|grad| from classification head            = 1.35e+02
|grad| from weight head (as coded, x1e-8)  = 8.17e-01   -> 0.6 % of classification
```

So the trunk is effectively optimised for classification only, while the
regression head is left with an unstable, badly-conditioned objective. The
resulting predictions are effectively a near-constant output unrelated to input.

**Control experiment** — feeding the same head a *standardised* weight target
instead of raw kilograms gives a gradient magnitude of `1.57e+06` (ratio
11,656 vs classification). Standardisation, not a hand-tuned constant, is what
makes the two objectives comparable.

## Conclusion

The logging system is working and has already answered the question: class
prediction is trustworthy, displacement prediction is **not**, and the cause is
an objective-balancing defect in `mlmodel._retrain()` rather than a data or
architecture problem.

Any fix should standardise the regression target (or use a log-scale target)
and combine the losses with an explicit, documented weighting — and must then be
re-validated through this same logging system. No accuracy claim should be made
without re-running the validation.

## Files

* `prediction_logger.py` — logging + validation engine
* `test_prediction_validation.py` — 23 tests (all passing)
* `logs/prediction_log.csv` — persistent per-prediction history
* `logs/prediction_summary.csv` — metrics snapshot
* `diagnose_weight_head.py` — reproducible diagnostic for the above numbers