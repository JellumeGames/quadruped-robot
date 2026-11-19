import math
import unittest

from pi.kinematics import (
    HIP_OFFSETS,
    L0,
    LEG_SIDE,
    LEGS,
    flatten_angles,
    leg_fk,
    leg_ik,
    leg_jacobian,
    nominal_stance_feet_world,
    rot_zyx,
    whole_body_ik,
)


class TestKinematics(unittest.TestCase):
    def test_ik_fk_roundtrip_dense_grid(self):
        max_err = 0.0
        for leg in LEGS:
            side = LEG_SIDE[leg]
            for x in (-0.07, -0.03, 0.0, 0.03, 0.07):
                for dy in (-0.035, 0.0, 0.035):
                    y = side * L0 + dy
                    for z in (-0.175, -0.155, -0.125, -0.100):
                        ik = leg_ik(x, y, z, side=side)
                        self.assertFalse(ik.clamped, f"Unexpected clamp at {(leg, x, y, z)}")
                        fx, fy, fz = leg_fk(ik.hip_roll, ik.hip_pitch, ik.knee, side=side)
                        err = math.sqrt((fx - x) ** 2 + (fy - y) ** 2 + (fz - z) ** 2)
                        max_err = max(max_err, err)
        self.assertLess(max_err, 1e-9)

    def test_out_of_workspace_clamps_safely_without_nan(self):
        for side in (1.0, -1.0):
            for target in ((0.0, 0.0, 0.0), (0.5, 0.5, -0.5), (-0.4, -0.3, 0.2), (0.0, 0.0, -0.01)):
                ik = leg_ik(*target, side=side, clamp_joints=True)
                self.assertTrue(ik.clamped)
                for val in ik.as_tuple():
                    self.assertTrue(math.isfinite(val))
                fk = leg_fk(ik.hip_roll, ik.hip_pitch, ik.knee, side=side)
                for val in fk:
                    self.assertTrue(math.isfinite(val))

    def test_analytical_jacobian_matches_finite_difference(self):
        eps = 1e-6
        test_poses = [
            (0.0, 0.85, -1.40, 1.0),
            (0.25, 0.55, -1.10, -1.0),
            (-0.30, 1.05, -1.80, 1.0),
        ]
        for q0, q1, q2, side in test_poses:
            j_ana = leg_jacobian(q0, q1, q2, side)
            q = [q0, q1, q2]
            for col in range(3):
                qp = list(q)
                qm = list(q)
                qp[col] += eps
                qm[col] -= eps
                fp = leg_fk(*qp, side)
                fm = leg_fk(*qm, side)
                for row in range(3):
                    num = (fp[row] - fm[row]) / (2.0 * eps)
                    self.assertAlmostEqual(j_ana[row][col], num, places=6)

    def test_whole_body_ik_with_pose_and_rpy(self):
        feet_world = nominal_stance_feet_world(0.155)
        body_pos = (0.01, -0.005, 0.015)
        body_rpy = (0.12, -0.09, 0.08)
        r_bw = rot_zyx(*body_rpy)
        sol = whole_body_ik(feet_world, body_pos=body_pos, body_rpy=body_rpy)

        for leg in LEGS:
            ik = sol[leg]
            self.assertFalse(ik.clamped)
            p_hip = leg_fk(*ik.as_tuple(), LEG_SIDE[leg])
            hx, hy, hz = HIP_OFFSETS[leg]
            p_body = (p_hip[0] + hx, p_hip[1] + hy, p_hip[2] + hz)
            p_world = (
                body_pos[0] + r_bw[0][0] * p_body[0] + r_bw[0][1] * p_body[1] + r_bw[0][2] * p_body[2],
                body_pos[1] + r_bw[1][0] * p_body[0] + r_bw[1][1] * p_body[1] + r_bw[1][2] * p_body[2],
                body_pos[2] + r_bw[2][0] * p_body[0] + r_bw[2][1] * p_body[1] + r_bw[2][2] * p_body[2],
            )
            for idx in range(3):
                self.assertAlmostEqual(p_world[idx], feet_world[leg][idx], places=9)

        flat = flatten_angles({leg: sol[leg].as_tuple() for leg in LEGS})
        self.assertEqual(len(flat), 12)


if __name__ == "__main__":
    unittest.main()
