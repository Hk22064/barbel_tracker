import sys
import os
import unittest

# Add parent directory to path to import vbt_analyzer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vbt_analyzer import VBTAnalyzer

class TestVBTAnalyzerFatigue(unittest.TestCase):
    def setUp(self):
        self.analyzer = VBTAnalyzer()

    def test_fatigue_calculation_with_slow_first_rep(self):
        """
        Test that a slow first rep (warm-up) does not cause negative drop-off
        when subsequent reps are faster.
        """
        # Scenario:
        # Rep 1: 0.2 m/s (Warm up / Slow)
        # Rep 2: 1.0 m/s (Peak performance)
        # Rep 3: 0.9 m/s (Slight fatigue)
        
        # Simulate Rep 1
        self.analyzer.current_state = self.analyzer.STATE_CONCENTRIC
        self.analyzer.current_max_velocity = 0.2
        self.analyzer.commit_rep()
        
        # Check Fatigue after Rep 1 (Should be 0% as it's the only rep/best rep)
        fatigue_1 = self.analyzer.get_fatigue_status()
        print(f"Rep 1 (0.2m/s) Fatigue: {fatigue_1}%")
        # Current logic might give 0.0 here which is fine.

        # Simulate Rep 2
        self.analyzer.current_state = self.analyzer.STATE_CONCENTRIC
        self.analyzer.current_max_velocity = 1.0
        self.analyzer.commit_rep()
        
        # Check Fatigue after Rep 2
        # EXPECTED with FIX: 0.0% (because 1.0 is the new max)
        # CURRENT BUGGY BEHAVIOR: 1.0 > 0.2, so drop off is negative (1 - 1.0/0.2 = -4.0 -> -400% or clamped to 0 depending on implementation)
        # The current code has `max(0.0, drop_off * 100)`, so it might report 0.0 but internally the calculation is wrong if we want it to be the baseline.
        # Wait, if `max(0.0, ...)` is there, it reports 0%. But the *Interpretation* is wrong.
        # However, let's look at the report summary logic in vbt_analyzer.py line 205:
        # `drop_off = (1.0 - (vel / first_rep_max)) * 100` -> This does NOT have max(0, ...).
        
        fatigue_2 = self.analyzer.get_fatigue_status()
        print(f"Rep 2 (1.0m/s) Fatigue: {fatigue_2}%")

        # Simulate Rep 3
        self.analyzer.current_state = self.analyzer.STATE_CONCENTRIC
        self.analyzer.current_max_velocity = 0.9
        self.analyzer.commit_rep()
        
        # Check Fatigue after Rep 3
        # EXPECTED with FIX: Baseline is 1.0. 0.9 is 10% drop off.
        # CURRENT BUGGY BEHAVIOR: Baseline is 0.2. 0.9 > 0.2, so returns 0% (clamped) or negative.
        fatigue_3 = self.analyzer.get_fatigue_status()
        print(f"Rep 3 (0.9m/s) Fatigue: {fatigue_3}%")
        
        # Assertion for Rep 3
        # If we fix the logic, max_velocity should be 1.0.
        # Drop off = (1 - 0.9/1.0) * 100 = 10.0
        
        # This assertion will FAIL on the current code if the logic is broken as described.
        # Current code likely uses 0.2 as baseline -> 0.9 > 0.2 -> Drop off is negative -> clamped to 0.0.
        # So it will report 0.0 instead of 10.0.
        self.assertAlmostEqual(fatigue_3, 10.0, delta=0.1, msg="Fatigue should be 10% (0.9 vs 1.0 baseline)")

    def test_report_summary_negative_dropoff(self):
        """
        Test that the final report does not show negative percentages.
        """
        self.analyzer.current_max_velocity = 0.1
        self.analyzer.commit_rep() # Rep 1: 0.1
        
        self.analyzer.current_max_velocity = 1.0
        self.analyzer.commit_rep() # Rep 2: 1.0
        
        summary = self.analyzer.get_results_summary()
        for line in summary:
            print(line)
            if "Rep 2:" in line:
                # OLD Behavior: Drop-off: -900.0% or something similar
                # NEW Behavior: Drop-off: 0.0% (because 1.0 is the peak)
                if "Drop-off:" in line:
                    # Extract the percentage value
                    parts = line.split("Drop-off:")
                    percentage_str = parts[1].strip().replace("%", "")
                    drop_off_val = float(percentage_str)
                    self.assertGreaterEqual(drop_off_val, 0.0, f"Drop-off should be non-negative, got {drop_off_val}")

if __name__ == '__main__':
    unittest.main()
