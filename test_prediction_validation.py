"""
Focused tests for prediction logging and ground-truth validation.

Run:  python test_prediction_validation.py
"""

import os
import shutil
import sys
import tempfile
import unittest

import prediction_logger as pl


class TestCircularAngleError(unittest.TestCase):
    """Direction comparison must respect the circular nature of angles."""

    def test_wraparound_small_difference(self):
        # 359 deg vs 1 deg is 2 deg apart, NOT 358 deg
        self.assertAlmostEqual(pl.circular_angle_error(359, 1), 2.0, places=6)

    def test_wraparound_opposite_wrap(self):
        self.assertAlmostEqual(pl.circular_angle_error(1, 359), 2.0, places=6)

    def test_same_angle(self):
        self.assertAlmostEqual(pl.circular_angle_error(90, 90), 0.0, places=6)

    def test_normal_difference(self):
        self.assertAlmostEqual(pl.circular_angle_error(30, 100), 70.0, places=6)

    def test_half_turn_boundary(self):
        self.assertAlmostEqual(pl.circular_angle_error(0, 180), 180.0, places=6)
        self.assertAlmostEqual(pl.circular_angle_error(0, 181), 179.0, places=6)

    def test_values_beyond_360_normalised(self):
        # 370 deg is equivalent to 10 deg
        self.assertAlmostEqual(pl.circular_angle_error(370, 10), 0.0, places=6)


class TestClassificationMetric(unittest.TestCase):
    def test_correct(self):
        err, kind, status = pl.classification_metric("Tanker", "Tanker")
        self.assertEqual(err, 0.0)
        self.assertEqual(kind, "exact_match")
        self.assertEqual(status, pl.STATUS_CORRECT)

    def test_incorrect(self):
        err, kind, status = pl.classification_metric("Ferry", "Tanker")
        self.assertEqual(err, 1.0)
        self.assertEqual(status, pl.STATUS_INCORRECT)

    def test_missing_ground_truth_is_unavailable_not_incorrect(self):
        err, kind, status = pl.classification_metric("Ferry", None)
        self.assertIsNone(err)
        self.assertEqual(status, pl.STATUS_UNAVAILABLE)


class TestRegressionMetric(unittest.TestCase):
    def test_absolute_error(self):
        err, kind, status = pl.regression_metric(12000, 12400)
        self.assertAlmostEqual(err, 400.0)
        self.assertEqual(kind, "abs_error")

    def test_exact_match(self):
        err, _, status = pl.regression_metric(5000, 5000)
        self.assertEqual(status, pl.STATUS_CORRECT)

    def test_missing_values_unavailable(self):
        err, _, status = pl.regression_metric(None, 5000)
        self.assertIsNone(err)
        self.assertEqual(status, pl.STATUS_UNAVAILABLE)


class TestPersistence(unittest.TestCase):
    """Records must survive a process restart (module reload)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig_dir = pl.LOG_DIR
        self._orig_csv = pl.PREDICTIONS_CSV
        self._orig_summary = pl.SUMMARY_CSV
        pl.LOG_DIR = self.tmp
        pl.PREDICTIONS_CSV = os.path.join(self.tmp, "prediction_log.csv")
        pl.SUMMARY_CSV = os.path.join(self.tmp, "prediction_summary.csv")
        pl.reset_run("test-run")

    def tearDown(self):
        pl.LOG_DIR = self._orig_dir
        pl.PREDICTIONS_CSV = self._orig_csv
        pl.SUMMARY_CSV = self._orig_summary
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_records_persist_across_reload(self):
        pl.log_classification(1, "NORMAL", 7, "Cargo Ship", "Cargo Ship")
        pl.log_classification(2, "NORMAL", 8, "Ferry", "Tanker")
        pl.log_regression(2, "NORMAL", 8, pl.TASK_WEIGHT, 12400, 12000)

        # Simulate a restart: forget the in-memory dedup cache
        pl._seen_ids = set()

        rows = pl.read_log()
        self.assertEqual(len(rows), 3)

        metrics = pl.compute_metrics(rows)
        self.assertEqual(metrics[pl.TASK_CLASS]["total_predictions"], 2)
        self.assertEqual(metrics[pl.TASK_CLASS]["correct"], 1)
        self.assertAlmostEqual(metrics[pl.TASK_CLASS]["accuracy"], 0.5)
        self.assertAlmostEqual(metrics[pl.TASK_WEIGHT]["mae_kg"], 400.0)

    def test_duplicate_prediction_ids_skipped(self):
        pl.log_predictions([{"prediction_id": "dup-1", "task": pl.TASK_CLASS,
                             "predicted_value": "A", "actual_value": "A",
                             "validation_status": pl.STATUS_CORRECT}])
        pl.log_predictions([{"prediction_id": "dup-1", "task": pl.TASK_CLASS,
                             "predicted_value": "A", "actual_value": "A",
                             "validation_status": pl.STATUS_CORRECT}])
        self.assertEqual(pl.log_count(), 1)

    def test_not_predicted_flag(self):
        rec = pl.log_not_predicted(3, "MONSOON", 9, pl.TASK_SPEED, 12.5)
        self.assertEqual(rec["validation_status"], pl.STATUS_NOT_PREDICTED)
        self.assertIsNone(rec["predicted_value"])
        rows = pl.read_log()
        m = pl.compute_metrics(rows)
        # A non-predicted value must not inflate validated counts or accuracy
        self.assertEqual(m[pl.TASK_SPEED]["validated"], 0)
        self.assertEqual(m[pl.TASK_SPEED]["not_predicted"], 1)

    def test_scenario_separation(self):
        pl.log_classification(1, "NORMAL", 1, "Tanker", "Tanker")
        pl.log_classification(2, "MONSOON", 2, "Tanker", "Ferry")
        normal = pl.summary_by_scenario(task=pl.TASK_CLASS)
        self.assertIn("NORMAL", normal)
        self.assertIn("MONSOON", normal)
        self.assertAlmostEqual(normal["NORMAL"][pl.TASK_CLASS]["accuracy"], 1.0)
        self.assertAlmostEqual(normal["MONSOON"][pl.TASK_CLASS]["accuracy"], 0.0)

    def test_run_id_filter(self):
        pl.reset_run("run-A")
        pl.log_classification(1, "NORMAL", 1, "Tanker", "Tanker")
        pl.reset_run("run-B")
        pl.log_classification(2, "NORMAL", 2, "Tanker", "Tanker")
        self.assertEqual(len(pl.read_log(run_id="run-A")), 1)
        self.assertEqual(len(pl.read_log(run_id="run-B")), 1)

    def test_mape_skipped_when_targets_near_zero(self):
        pl.log_regression(1, "NORMAL", 1, pl.TASK_WEIGHT, 100, 0)
        m = pl.compute_metrics(pl.read_log())
        self.assertIsNone(m[pl.TASK_WEIGHT]["mape_pct"])

    def test_summary_file_written(self):
        pl.log_classification(1, "NORMAL", 1, "Tanker", "Tanker")
        self.assertTrue(pl.write_summary())
        self.assertTrue(os.path.exists(pl.SUMMARY_CSV))


class TestAggregation(unittest.TestCase):
    """Aggregate metrics must come from real observations, not guesses."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig_dir, self._orig_csv = pl.LOG_DIR, pl.PREDICTIONS_CSV
        pl.LOG_DIR = self.tmp
        pl.PREDICTIONS_CSV = os.path.join(self.tmp, "p.csv")
        pl.reset_run("agg")

    def tearDown(self):
        pl.LOG_DIR, pl.PREDICTIONS_CSV = self._orig_dir, self._orig_csv
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_regression_mae_rmse(self):
        pl.log_regression(1, "NORMAL", 1, pl.TASK_WEIGHT, 110, 100)   # err 10
        pl.log_regression(2, "NORMAL", 2, pl.TASK_WEIGHT, 130, 100)   # err 30
        m = pl.compute_metrics(pl.read_log())[pl.TASK_WEIGHT]
        self.assertAlmostEqual(m["mae_kg"], 20.0)
        # RMSE = sqrt((100 + 900)/2) = sqrt(500)
        self.assertAlmostEqual(m["rmse_kg"], (500 ** 0.5))
        # RMSE must always be >= MAE
        self.assertGreaterEqual(m["rmse_kg"], m["mae_kg"])

    def test_direction_aggregate(self):
        pl.log_direction(1, "NORMAL", 1, 359, 1)   # err 2
        pl.log_direction(2, "NORMAL", 2, 90, 90)    # err 0
        m = pl.compute_metrics(pl.read_log())[pl.TASK_DIRECTION]
        self.assertAlmostEqual(m["mean_angular_error_deg"], 1.0)
        self.assertAlmostEqual(m["max_angular_error_deg"], 2.0)

    def test_zero_predictions_is_not_a_perfect_score(self):
        m = pl.compute_metrics([])[pl.TASK_CLASS]
        self.assertIsNone(m["accuracy"])

    def test_current_run_metrics(self):
        pl.log_classification(1, "NORMAL", 1, "Tanker", "Tanker")
        pl.log_classification(2, "NORMAL", 2, "Ferry", "Ferry")
        cur = pl.current_run_metrics()
        self.assertAlmostEqual(cur[pl.TASK_CLASS]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)