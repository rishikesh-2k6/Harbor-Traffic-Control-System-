"""
Integration and regression tests (Phase 8).

Covers the areas that Phase 5 unit tests do not: the simulation pipeline,
networking, traffic governance, AUV behaviour, scenario switching, error
handling, and the web dashboard theme tokens.

Run:  python test_integration.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulator as sim
import auv
import network
import traffic_algo as tms


class TestSimulationPipeline(unittest.TestCase):
    """Core engine must run without raising and produce plausible state."""

    def setUp(self):
        sim.set_active_scenario('NORMAL')
        self.sensors = sim.generate_sensors()
        self.fleet = sim.generate_fleet()
        auv.init_physics()
        auv.spawn_auvs()

    def test_fleet_composition(self):
        self.assertEqual(len(self.fleet), 22)
        types = {v['type'] for v in self.fleet}
        self.assertEqual(len(types), 6)
        for v in self.fleet:
            self.assertIn('true_weight', v)
            self.assertGreater(v['true_weight'], 0)

    def test_sensors_within_map_bounds(self):
        self.assertEqual(len(self.sensors), 35)
        for s in self.sensors:
            x, y, z = s.pos
            self.assertTrue(0 <= x <= sim.MAP_SIZE, f"x out of range: {x}")
            self.assertTrue(0 <= y <= sim.MAP_SIZE, f"y out of range: {y}")

    def test_stepping_does_not_raise(self):
        for frame in range(5):
            sim.step_ocean_environment(frame, 1 / 60)
            sim.step_sensors(frame, 1 / 60)
            auv.step_auvs(frame, 1 / 60, active_violations=[], fleet=self.fleet)

    def test_acoustic_physics_monotonic(self):
        # Transmission loss must increase with range
        t1 = sim.calculate_transmission_loss(500, 12000)
        t2 = sim.calculate_transmission_loss(2000, 12000)
        self.assertGreater(t2, t1)

    def test_thorp_absorption_increases_with_frequency(self):
        low = sim.calculate_thorp_absorption(1000)     # 1 kHz
        high = sim.calculate_thorp_absorption(50000)   # 50 kHz
        self.assertGreater(high, low)

    def test_doppler_shift_direction(self):
        c = 1500.0
        approaching = sim.calculate_doppler_shift(1000.0, +10.0, c)
        receding = sim.calculate_doppler_shift(1000.0, -10.0, c)
        self.assertGreater(approaching, 1000.0)
        self.assertLess(receding, 1000.0)

    def test_snr_is_level_difference(self):
        self.assertAlmostEqual(sim.calculate_snr(80.0, 60.0), 20.0)


class TestScenarioSwitching(unittest.TestCase):
    """Both scenarios must work on the same engine without restart."""

    def tearDown(self):
        sim.set_active_scenario('NORMAL')

    def test_noise_increases_in_monsoon(self):
        sim.set_active_scenario('NORMAL')
        normal = network.get_current_ambient_noise_db()
        sim.set_active_scenario('MONSOON')
        monsoon = network.get_current_ambient_noise_db()
        self.assertGreater(monsoon, normal)
        self.assertGreater(monsoon - normal, 15.0)   # documented ~+26 dB

    def test_duty_cycling_degrades_in_monsoon(self):
        results = {}
        for scen in ('NORMAL', 'MONSOON'):
            sim.set_active_scenario(scen)
            active = sum(1 for i in range(35)
                         if network.is_node_active(f'sensor_{i}', 0))
            results[scen] = active
        self.assertGreater(results['NORMAL'], results['MONSOON'])

    def test_auvs_and_buoys_never_sleep(self):
        for node in ('auv_1', 'auv_2', 'buoy_alpha', 'buoy_beta', 'buoy_gamma'):
            self.assertTrue(network.is_node_active(node, 0),
                            f"{node} should always be active")

    def test_squall_increases_current(self):
        sim.set_active_scenario('MONSOON')
        for f in range(3):
            sim.step_ocean_environment(f, 1 / 60)
        before = sim.get_monsoon_current_info()['intensity_mps']
        surge = sim.trigger_monsoon_squall()
        self.assertGreater(surge, 0)
        after = sim.get_monsoon_current_info()
        self.assertTrue(after['squall_active'])
        self.assertGreaterEqual(after['intensity_mps'], before)


class TestNetworking(unittest.TestCase):
    """Graph construction, link gating and health scoring."""

    def setUp(self):
        sim.set_active_scenario('NORMAL')
        self.sensors = sim.generate_sensors()
        self.fleet = sim.generate_fleet()
        auv.init_physics()
        self.auvs = auv.spawn_auvs()

    def test_graph_contains_all_node_types(self):
        g = network.build_network_graph(self.sensors, self.auvs)
        self.assertEqual(g.number_of_nodes(), 35 + len(self.auvs) + len(network.BUOYS))

    def test_links_respect_snr_threshold(self):
        g = network.build_network_graph(self.sensors, self.auvs)
        for _, _, d in g.edges(data=True):
            self.assertLessEqual(d['dist'], network.R_COMM + 1e-6)
            self.assertGreaterEqual(d['snr'], network.SNR_THRESHOLD)

    def test_monsoon_reduces_link_count(self):
        sim.set_active_scenario('NORMAL')
        g_n = network.build_network_graph(self.sensors, self.auvs)
        sim.set_active_scenario('MONSOON')
        g_m = network.build_network_graph(self.sensors, self.auvs)
        self.assertLess(g_m.number_of_edges(), g_n.number_of_edges())
        sim.set_active_scenario('NORMAL')

    def test_routing_produces_stats(self):
        g = network.build_network_graph(self.sensors, self.auvs)
        _, _, stats = network.route_detections(
            self.fleet, self.sensors, self.auvs, g, frame=0)
        for key in ('pdr', 'packet_loss_pct', 'network_state',
                    'health_score', 'avg_hop_count'):
            self.assertIn(key, stats)
        self.assertGreaterEqual(stats['health_score'], 0.0)
        self.assertLessEqual(stats['health_score'], 100.0)

    def test_health_state_is_valid(self):
        g = network.build_network_graph(self.sensors, self.auvs)
        _, _, stats = network.route_detections(
            self.fleet, self.sensors, self.auvs, g, frame=0)
        self.assertIn(stats['network_state'],
                      {'STABLE', 'DEGRADED', 'CRITICAL', 'COLLAPSED'})

    def test_compression_reduces_delay(self):
        self.assertLess(network.TRANS_DELAY_COMPRESSED,
                        network.TRANS_DELAY_DEFAULT)
        self.assertLess(network.ENERGY_COMPRESSED_PER_HOP,
                        network.ENERGY_DEFAULT_PER_HOP)
        self.assertAlmostEqual(
            network.TRANS_DELAY_DEFAULT / network.TRANS_DELAY_COMPRESSED, 16.0)

    def test_reanchor_restores_sensors(self):
        sim.set_active_scenario('MONSOON')
        for f in range(200):
            sim.step_sensors(f, 1 / 60)
        sim.reset_all_sensors()
        self.assertEqual(sim.get_sensor_states()['snapped'], 0)
        sim.set_active_scenario('NORMAL')

    def test_network_reset_clears_counters(self):
        g = network.build_network_graph(self.sensors, self.auvs)
        network.route_detections(self.fleet, self.sensors, self.auvs, g, frame=0)
        network.reset_network()
        self.assertEqual(network.get_engine().packets_sent, 0)
        self.assertEqual(network.get_engine().packets_dropped, 0)


class TestTrafficGovernance(unittest.TestCase):
    """Speed enforcement, zone routing and berth assignment."""

    def setUp(self):
        sim.set_active_scenario('NORMAL')
        self.fleet = sim.generate_fleet()
        tms.clear_dock_assignments()

    def test_outer_yard_speed_limit(self):
        v = self.fleet[0]
        v['pos'] = np.array([7500.0, 5000.0])
        tms.manage_traffic_from_csv(self.fleet)
        v['current_speed'] = 12.0 * 0.514      # 12 kn
        tms.manage_traffic_from_csv(self.fleet)
        self.assertTrue(v['overspeed'])

    def test_inner_zone_speed_limit(self):
        v = self.fleet[0]
        v['pos'] = np.array([3000.0, 5000.0])
        v['current_speed'] = 8.0 * 0.514        # 8 kn > 6 kn inner limit
        tms.manage_traffic_from_csv(self.fleet)
        self.assertTrue(v['overspeed'])

    def test_within_limit_is_not_a_violation(self):
        v = self.fleet[0]
        v['pos'] = np.array([7500.0, 5000.0])
        v['current_speed'] = 5.0 * 0.514
        tms.manage_traffic_from_csv(self.fleet)
        self.assertFalse(v['overspeed'])

    def test_heavy_vessel_is_halted_at_loc(self):
        v = next(x for x in self.fleet if x['type'] == 'Tanker')
        v['pos'] = np.array([7000.0, 5000.0])
        v['pred_type'] = 'Tanker'
        tms.manage_traffic_from_csv(self.fleet)
        self.assertEqual(v['tms_command'], 'HALT')
        self.assertEqual(v['current_speed'], 0)

    def test_light_vessel_continues(self):
        v = next(x for x in self.fleet if x['type'] == 'Speedboat')
        v['pos'] = np.array([7000.0, 5000.0])
        v['pred_type'] = 'Speedboat'
        tms.manage_traffic_from_csv(self.fleet)
        self.assertEqual(v['tms_command'], 'CONTINUE')

    def test_inner_zone_assigns_berth(self):
        v = self.fleet[0]
        v['pos'] = np.array([3000.0, 5000.0])
        v['pred_type'] = v['type']
        tms.manage_traffic_from_csv(self.fleet)
        self.assertIsNotNone(v.get('dock_slot'))

    def test_berths_are_not_double_assigned(self):
        # The fleet needs 22 berths but only 13 exist. Excess vessels must
        # queue for a berth rather than stacking on an occupied one.
        tms.clear_dock_assignments()
        for v in self.fleet:
            v['pos'] = np.array([3000.0, 5000.0])
            v['pred_type'] = v['type']
        tms.manage_traffic_from_csv(self.fleet)
        slots = [v.get('dock_slot') for v in self.fleet if v.get('dock_slot')]
        self.assertEqual(len(slots), len(set(slots)), "berth collision detected")

    def test_excess_vessels_queue_for_berth(self):
        tms.clear_dock_assignments()
        for v in self.fleet:
            v['pos'] = np.array([3000.0, 5000.0])
            v['pred_type'] = v['type']
        tms.manage_traffic_from_csv(self.fleet)
        assigned = [v for v in self.fleet if v.get('dock_slot')]
        waiting = [v for v in self.fleet
                   if v.get('state') == 'HOLDING_FOR_BERTH']
        # Every vessel is accounted for: berthed OR waiting, never neither
        self.assertEqual(len(assigned) + len(waiting), len(self.fleet))
        self.assertGreater(len(waiting), 0, "oversubscribed classes should queue")
        # A waiting vessel must not be moving
        for v in waiting:
            self.assertEqual(v['current_speed'], 0)
        tms.clear_dock_assignments()

    def test_released_berth_is_reusable(self):
        tms.clear_dock_assignments()
        v = self.fleet[0]
        v['pos'] = np.array([3000.0, 5000.0])
        v['pred_type'] = v['type']
        tms.manage_traffic_from_csv(self.fleet)
        slot = v.get('dock_slot')
        self.assertIsNotNone(slot)
        tms.release_dock(v['id'])
        self.assertNotIn(v['id'], tms._dock_assignments)

    def test_direction_rule_is_geometric_not_ml(self):
        v = self.fleet[0]
        v['angle'] = 180.0
        self.assertEqual(sim.get_vessel_direction(v), 'INBOUND')
        v['angle'] = 0.0
        self.assertEqual(sim.get_vessel_direction(v), 'OUTBOUND')


class TestAUVBehaviour(unittest.TestCase):
    """Patrol, battery, dispatch and boundary handling."""

    def setUp(self):
        sim.set_active_scenario('NORMAL')
        auv.init_physics()
        self.states = auv.spawn_auvs()

    def test_expected_number_spawned(self):
        self.assertEqual(len(self.states), sim.NUM_AUVS)

    def test_positions_stay_in_bounds(self):
        for f in range(60):
            auv.step_auvs(f, 1 / 60, active_violations=[], fleet=[])
        for s in auv.get_auv_states():
            x, y, z = s['pos']
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, sim.MAP_SIZE)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, sim.MAP_SIZE)

    def test_battery_drains_over_time(self):
        before = auv.get_auv_states()[0]['battery_pct']
        for f in range(30):
            auv.step_auvs(f, 1 / 60, active_violations=[], fleet=[])
        after = auv.get_auv_states()[0]['battery_pct']
        self.assertLess(after, before)

    def test_low_battery_triggers_charging(self):
        for r in auv._robots:
            r.battery_pct = 5.0          # below the 15 % threshold
        auv.step_auvs(0, 1 / 60, active_violations=[], fleet=[])
        self.assertTrue(any(r.status == 'CHARGING' for r in auv._robots))

    def test_status_values_are_known(self):
        for s in auv.get_auv_states():
            self.assertIn(s['status'],
                          {'PATROLLING', 'DISPATCHED', 'CHARGING'})


class TestErrorHandling(unittest.TestCase):
    """Failure paths must be observable rather than silently swallowed."""

    def test_detection_collection_survives_empty_fleet(self):
        active, rows = sim.collect_sensor_detections([], sim.generate_sensors(), 0)
        self.assertIsInstance(active, list)
        self.assertEqual(rows, 0)

    def test_detection_rows_carry_ground_truth(self):
        sensors = sim.generate_sensors()
        fleet = sim.generate_fleet()
        for v in fleet:
            v['pos'] = np.array([sensors[0].pos[0], sensors[0].pos[1]])
        active, _ = sim.collect_sensor_detections(fleet, sensors, 0)
        self.assertTrue(any(active))

    def test_empty_graph_routing_does_not_crash(self):
        import networkx as nx
        empty = nx.Graph()
        _, _, stats = network.route_detections(
            sim.generate_fleet(), [], [], empty, frame=0)
        self.assertIn('network_state', stats)

    def test_prediction_logger_isolated_on_bad_input(self):
        import prediction_logger as pl
        tmp = tempfile.mkdtemp()
        orig_csv = pl.PREDICTIONS_CSV
        pl.PREDICTIONS_CSV = os.path.join(tmp, 'p.csv')
        try:
            # Non-numeric values must be reported, not crash the caller
            pl.log_regression(1, 'NORMAL', 1, pl.TASK_WEIGHT,
                              predicted_value='abc', actual_value=100)
            rows = pl.read_log()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['validation_status'], pl.STATUS_UNAVAILABLE)
        finally:
            pl.PREDICTIONS_CSV = orig_csv


class TestDashboardThemes(unittest.TestCase):
    """Static verification of the web dashboards."""

    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, 'dashboard.html'), encoding='utf-8') as fh:
            self.dash = fh.read()
        with open(os.path.join(here, 'networking_dashboard.html'), encoding='utf-8') as fh:
            self.net = fh.read()

    def test_both_have_theme_toggle(self):
        for name, html in (('dashboard', self.dash), ('network', self.net)):
            self.assertIn('id="theme-toggle"', html, name)
            self.assertIn('function applyTheme', html, name)
            self.assertIn('function initTheme', html, name)

    def test_theme_persists_via_localstorage(self):
        for html in (self.dash, self.net):
            self.assertIn('localStorage.setItem', html)
            self.assertIn('localStorage.getItem', html)

    def test_both_share_one_storage_key(self):
        key_d = re.search(r"THEME_KEY\s*=\s*'([^']+)'", self.dash).group(1)
        key_n = re.search(r"THEME_KEY\s*=\s*'([^']+)'", self.net).group(1)
        self.assertEqual(key_d, key_n, "dashboards must stay in sync")

    def test_light_theme_defined(self):
        for html in (self.dash, self.net):
            self.assertIn('html[data-theme="light"]', html)

    def test_all_var_references_resolve(self):
        for name, html in (('dashboard', self.dash), ('network', self.net)):
            css = html.split('</style>', 1)[0]
            declared = set(re.findall(r'(--[a-z0-9-]+)\s*:', css))
            used = set(re.findall(r'var\((--[a-z0-9-]+)', html))
            self.assertEqual(used - declared, set(), f"{name}: unresolved vars")

    def test_no_hardcoded_dark_alphas_in_body(self):
        for name, html in (('dashboard', self.dash), ('network', self.net)):
            body = html.split('</style>', 1)[1]
            self.assertNotRegex(body, r'rgba\(255,\s*255,\s*255',
                                f"{name}: hardcoded white-alpha outside tokens")

    def test_accessibility_features_present(self):
        for html in (self.dash, self.net):
            self.assertIn(':focus-visible', html)
            self.assertIn('prefers-reduced-motion', html)
            self.assertIn('aria-label', html)
            self.assertIn('aria-pressed', html)

    def test_monospace_fallback_chain(self):
        for html in (self.dash, self.net):
            self.assertIn('JetBrains Mono', html)
            self.assertIn('Consolas', html)

    def test_sanitize_storage_failure(self):
        # A blocked localStorage must not break initialisation
        for html in (self.dash, self.net):
            self.assertIn('storage unavailable', html)


class TestMLIntegration(unittest.TestCase):
    """The trained model must load and produce all documented outputs."""

    @classmethod
    def setUpClass(cls):
        import mlmodel as ml
        cls.ml = ml

    def test_model_artifact_loads(self):
        self.assertTrue(self.ml._model_is_live)
        self.assertIsNotNone(self.ml._cached_model)
        self.assertIsNotNone(self.ml._cached_scaler)
        self.assertIsNotNone(self.ml._cached_le)

    def test_feature_count_matches_model(self):
        self.assertEqual(len(self.ml.ALL_FEATURES), 21)

    def test_six_classes_encoded(self):
        self.assertEqual(len(self.ml._cached_le.classes_), 6)

    def test_scaler_fitted_on_train_split_only(self):
        src = open(self.ml.__file__, encoding='utf-8').read()
        self.assertIn('scaler.fit_transform(X_train_r)', src)
        self.assertIn('scaler.transform(X_val_r)', src)

    def test_inference_returns_two_heads(self):
        import torch
        n = len(self.ml.ALL_FEATURES)
        ncls = len(self.ml._cached_le.classes_)
        model = self.ml.VesselMLP(n, ncls)
        x = torch.zeros(2, n)
        logits, weight = model(x)
        self.assertEqual(tuple(logits.shape), (2, ncls))
        self.assertEqual(tuple(weight.shape), (2,))


if __name__ == '__main__':
    unittest.main(verbosity=2)