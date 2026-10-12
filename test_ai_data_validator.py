"""
Unit tests for Groq AI Data Validator module.
Verifies telemetry payload generation, local physics heuristics,
and schema integrity without requiring an active Groq API key.
"""

import os
import unittest
from ai_data_validator import GroqDataValidator, _local_heuristic_validation, SYSTEM_PROMPT


class TestGroqDataValidator(unittest.TestCase):
    def setUp(self):
        self.validator = GroqDataValidator(api_key="test_dummy_key")

    def test_system_prompt_contains_core_domains(self):
        """System prompt must explicitly mention currents, weights, acoustics, and schema."""
        self.assertIn("Ocean Current Dynamics", SYSTEM_PROMPT)
        self.assertIn("Vessel Architecture & Displacement Weights", SYSTEM_PROMPT)
        self.assertIn("Underwater Acoustic Propagation", SYSTEM_PROMPT)
        self.assertIn("real_world_plausibility_pct", SYSTEM_PROMPT)
        self.assertIn("Mackenzie", SYSTEM_PROMPT)

    def test_local_heuristic_nominal_data(self):
        """Nominal synthetic data should receive high plausibility score."""
        currents = {
            "intensity_mps": 0.35,
            "surge_level": 0.1,
            "wave_height": 0.4
        }
        vessels = [
            {"type": "Cargo Ship", "true_weight": 90000, "speed_knots": 8.5},
            {"type": "Speedboat", "true_weight": 4500, "speed_knots": 28.0}
        ]
        acoustics = [
            {
                "Sound_Speed_mps": 1505.0,
                "Received_Level_dB": 115.0,
                "Source_Level_dB": 175.0,
                "Transmission_Loss_dB": 60.0,
                "SNR_dB": 50.0,
                "Ambient_Noise_dB": 65.0,
                "Frequency_Hz": 300.0,
                "Doppler_Shift_Hz": 1.2
            }
        ]

        result = _local_heuristic_validation(currents, vessels, acoustics)
        self.assertIn("real_world_plausibility_pct", result)
        self.assertGreaterEqual(result["real_world_plausibility_pct"], 80.0)
        self.assertEqual(result["verdict"], "HIGHLY_REALISTIC")
        self.assertIn("currents_plausibility_pct", result["category_scores"])
        self.assertIn("vessel_weights_plausibility_pct", result["category_scores"])
        self.assertIn("acoustics_plausibility_pct", result["category_scores"])

    def test_local_heuristic_catches_unphysical_sound_speed(self):
        """Impossible sound speeds (< 1400 or > 1600 m/s) must be flagged as violation."""
        acoustics = [
            {
                "Sound_Speed_mps": 2800.0,  # Unphysical in seawater!
                "Received_Level_dB": 100.0,
                "Source_Level_dB": 160.0,
                "Transmission_Loss_dB": 60.0,
                "SNR_dB": 40.0,
                "Ambient_Noise_dB": 60.0,
                "Frequency_Hz": 500.0,
                "Doppler_Shift_Hz": 0.0
            }
        ]
        result = _local_heuristic_validation(None, None, acoustics)
        violations = result["critical_violations"]
        param_names = [v["parameter"] for v in violations]
        self.assertIn("Sound_Speed_mps", param_names)

    def test_local_heuristic_catches_excessive_current(self):
        """Currents over 7 m/s (~14 knots) inside a harbor must trigger violation."""
        currents = {
            "intensity_mps": 14.5,  # Unphysical harbor tsunami/extreme
            "surge_level": 12.0,
            "wave_height": 18.0
        }
        result = _local_heuristic_validation(currents, None, None)
        param_names = [v["parameter"] for v in result["critical_violations"]]
        self.assertIn("Current_Intensity_mps", param_names)

    def test_fallback_when_no_active_api(self):
        """When client is None, validate_telemetry_payload should cleanly return heuristic report."""
        v = GroqDataValidator(api_key=None)
        v.client = None  # Force offline simulation
        res = v.validate_telemetry_payload()
        self.assertIn("real_world_plausibility_pct", res)
        self.assertEqual(res["_meta"]["engine"], "local_physics_rule_engine")


if __name__ == "__main__":
    unittest.main()
