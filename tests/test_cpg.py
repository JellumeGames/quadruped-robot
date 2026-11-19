import math
import unittest

from pi.cpg import CoupledHopfCPG, GaitPlanner
from pi.kinematics import LEGS


def _circ_diff_deg(a: float, b: float) -> float:
    return abs(math.degrees((a - b + math.pi) % (2.0 * math.pi) - math.pi))


class TestCPG(unittest.TestCase):
    def test_coupled_hopf_limit_cycle_and_trot_locking_from_random_phases(self):
        cpg = CoupledHopfCPG(step_freq_hz=1.5, gait="trot", alpha=8.0, coupling=3.0)
        cpg.set_phase("LF", 0.3, radius=0.5)
        cpg.set_phase("RF", 1.9, radius=1.4)
        cpg.set_phase("LR", 4.1, radius=0.8)
        cpg.set_phase("RR", 5.5, radius=1.2)

        for _ in range(400):
            cpg.step(0.01)

        radii = cpg.radii()
        phases = cpg.phases()
        for leg in LEGS:
            self.assertAlmostEqual(radii[leg], 1.0, delta=1e-3)

        self.assertLess(_circ_diff_deg(phases["LF"], phases["RR"]), 1.0)
        self.assertLess(_circ_diff_deg(phases["RF"], phases["LR"]), 1.0)
        self.assertLess(_circ_diff_deg(phases["LF"] - phases["RF"], math.pi), 1.0)

    def test_walk_and_bound_phase_offsets(self):
        for gait, expected_offsets in (
            ("walk", {"LF": 0.0, "RR": 0.5 * math.pi, "RF": math.pi, "LR": 1.5 * math.pi}),
            ("bound", {"LF": 0.0, "RF": 0.0, "LR": math.pi, "RR": math.pi}),
        ):
            cpg = CoupledHopfCPG(step_freq_hz=1.2, gait=gait, coupling=3.0)
            cpg.set_phase("LF", 0.4)
            cpg.set_phase("RF", 2.2)
            cpg.set_phase("LR", 3.1)
            cpg.set_phase("RR", 0.9)
            for _ in range(500):
                cpg.step(0.01)
            phases = cpg.phases()
            for leg in LEGS:
                self.assertLess(_circ_diff_deg(phases[leg] - phases["LF"], expected_offsets[leg]), 1.0)

    def test_fsr_contact_reset_and_smooth_swing_landing(self):
        planner = GaitPlanner(step_freq_hz=1.5, step_height=0.032)
        planner.set_velocity(vx=0.20, vy=0.0, yaw_rate=0.0)
        planner.cpg.set_phase("LF", 1.85 * math.pi)
        self.assertTrue(planner.on_foot_contact("LF"))
        self.assertAlmostEqual(planner.cpg.phase("LF"), 0.0, delta=1e-6)

        # Check zero vertical slope at swing liftoff (pi) and touchdown (2*pi)
        z_at_pi = planner._foot_world_target("LF", math.pi + 1e-4)[2]
        z_base = planner._foot_world_target("LF", 0.5)[2]
        z_before_land = planner._foot_world_target("LF", 2.0 * math.pi - 1e-4)[2]
        self.assertAlmostEqual(z_at_pi, z_base, delta=1e-6)
        self.assertAlmostEqual(z_before_land, z_base, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
