import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vbt_analyzer import VBTAnalyzer

class TestVBTAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = VBTAnalyzer(smoothing_window=1) # Disable smoothing for easy math check

    def test_calibration(self):
        # 100 pixels = 0.81 meters
        success = self.analyzer.calibrate([0, 0], [0, 100], real_distance_m=0.81)
        self.assertTrue(success)
        self.assertAlmostEqual(self.analyzer.calibration_factor, 0.0081)

    def test_rep_logic_state_machine(self):
        self.analyzer.calibration_factor = 1.0 # 1px = 1m
        self.analyzer.smoothing_window = 1
        
        # 1. Start (WAITING)
        self.assertEqual(self.analyzer.current_state, "WAITING")
        self.assertEqual(self.analyzer.rep_count, 0)
        
        # 2. Start Lowering (ECCENTRIC) (-0.1 m/s)
        self.analyzer.process_rep(-0.1)
        self.assertEqual(self.analyzer.current_state, "ECCENTRIC")
        
        # 3. Bottom (Transition to CONCENTRIC)
        # Velocity becomes positive (+0.1 m/s)
        self.analyzer.process_rep(0.1)
        self.assertEqual(self.analyzer.current_state, "CONCENTRIC")
        
        # 4. Lifting Phase (Peak)
        self.analyzer.process_rep(1.0) # 1.0 m/s
        self.assertEqual(self.analyzer.current_max_velocity, 1.0)
        
        self.analyzer.process_rep(0.5) # Slowing down
        self.assertEqual(self.analyzer.current_max_velocity, 1.0) # Peak hold
        
        # 5. Top (Finish Rep)
        # Velocity drops below threshold (< 0.05)
        self.analyzer.process_rep(0.0)
        self.assertEqual(self.analyzer.current_state, "WAITING")
        self.assertEqual(self.analyzer.rep_count, 1)
        self.assertEqual(self.analyzer.rep_velocities[-1], 1.0)

    def test_fatigue_calc(self):
        self.analyzer.rep_velocities = [1.0, 0.9, 0.5]
        self.analyzer.max_velocity_first_rep = 1.0
        
        # Current status based on last rep (0.5)
        # Drop off = 1.0 - 0.5/1.0 = 0.5 (50%)
        fatigue = self.analyzer.get_fatigue_status()
        self.assertEqual(fatigue, 50.0)

if __name__ == '__main__':
    unittest.main()
