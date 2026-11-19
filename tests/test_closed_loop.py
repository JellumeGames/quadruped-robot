import unittest

from pi.sim_runner import run_simulation


class TestClosedLoopSimulation(unittest.TestCase):
    def test_closed_loop_trot_disturbance_and_watchdog(self):
        summary = run_simulation(
            duration_s=4.0,
            step_freq_hz=1.5,
            vx=0.18,
            inject_disturbance=True,
            inject_dropout=True,
            seed=42,
        )
        self.assertLess(summary.max_ik_fk_err_mm, 1e-3)
        self.assertLess(summary.trot_diag_sync_err_deg, 1.0)
        self.assertLess(summary.trot_anti_phase_err_deg, 1.0)
        self.assertGreater(summary.peak_disturbance_roll_deg, 5.0)
        self.assertLess(summary.post_recovery_roll_rms_deg, 1.0)
        self.assertLess(summary.imu_est_err_rms_deg, 1.0)
        self.assertGreater(summary.crc_rejected_frames, 0)
        self.assertGreater(summary.rx_valid_frames, 150)
        self.assertEqual(summary.ack_count, 1)
        self.assertTrue(summary.watchdog_triggered)
        self.assertTrue(summary.watchdog_recovered)
        self.assertGreater(summary.fsr_resets, 4)


if __name__ == "__main__":
    unittest.main()
