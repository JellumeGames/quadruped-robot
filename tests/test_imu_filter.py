import math
import random
import unittest

from pi.imu_filter import (
    GyroCalibrator,
    MahonyAHRS,
    decode_mpu6050_frame,
    encode_mpu6050_frame,
)


class TestImuFilter(unittest.TestCase):
    def test_mpu6050_14byte_roundtrip_with_negative_values(self):
        raw = encode_mpu6050_frame(
            ax_g=-1.25,
            ay_g=0.45,
            az_g=-0.80,
            gx_rads=-1.80,
            gy_rads=0.95,
            gz_rads=-2.50,
            temp_c=-4.5,
        )
        self.assertEqual(len(raw), 14)
        s = decode_mpu6050_frame(raw)
        self.assertAlmostEqual(s.ax_g, -1.25, places=3)
        self.assertAlmostEqual(s.ay_g, 0.45, places=3)
        self.assertAlmostEqual(s.az_g, -0.80, places=3)
        self.assertAlmostEqual(s.gx_rads, -1.80, places=3)
        self.assertAlmostEqual(s.gy_rads, 0.95, places=3)
        self.assertAlmostEqual(s.gz_rads, -2.50, places=3)
        self.assertAlmostEqual(s.temp_c, -4.5, places=1)

    def test_gyro_calibrator_rejects_motion_then_calibrates(self):
        cal = GyroCalibrator(target_samples=30, max_accel_std_g=0.05)
        # Noisy shaking prevents calibration
        for k in range(30):
            s = decode_mpu6050_frame(
                encode_mpu6050_frame(0.0, 0.0, 1.0 + (0.3 if k % 2 == 0 else -0.3), 0.1, -0.2, 0.05)
            )
            self.assertFalse(cal.add_sample(s))
        self.assertFalse(cal.calibrated)

        # Stationary samples succeed
        for _ in range(30):
            s = decode_mpu6050_frame(encode_mpu6050_frame(0.0, 0.0, 1.0, 0.021, -0.015, 0.008))
            cal.add_sample(s)
        self.assertTrue(cal.calibrated)
        self.assertAlmostEqual(cal.bias[0], 0.021, places=3)
        self.assertAlmostEqual(cal.bias[1], -0.015, places=3)
        self.assertAlmostEqual(cal.bias[2], 0.008, places=3)

    def test_stationary_tilt_convergence(self):
        for target_roll_deg, target_pitch_deg in ((18.0, -12.0), (-15.0, 10.0)):
            r = math.radians(target_roll_deg)
            p = math.radians(target_pitch_deg)
            # Body-frame gravity measurement for ZYX (roll, pitch)
            ax = -math.sin(p)
            ay = math.cos(p) * math.sin(r)
            az = math.cos(p) * math.cos(r)

            ahrs = MahonyAHRS(kp=3.0, ki=0.05)
            for _ in range(500):
                roll_est, pitch_est, _ = ahrs.update(ax, ay, az, 0.0, 0.0, 0.0, 0.01)

            self.assertAlmostEqual(math.degrees(roll_est), target_roll_deg, delta=0.1)
            self.assertAlmostEqual(math.degrees(pitch_est), target_pitch_deg, delta=0.1)

    def test_dynamic_tracking_and_impact_shock_gating(self):
        rng = random.Random(7)
        ahrs = MahonyAHRS(kp=2.2, ki=0.05, accel_gate_g=0.25)
        dt = 0.01
        err_sq = 0.0
        n = 300
        for k in range(n):
            t = k * dt
            roll = 0.20 * math.sin(2.0 * math.pi * 0.5 * t)
            pitch = 0.15 * math.cos(2.0 * math.pi * 0.4 * t)
            d_roll = 0.20 * math.pi * math.cos(2.0 * math.pi * 0.5 * t)
            d_pitch = -0.15 * 0.8 * math.pi * math.sin(2.0 * math.pi * 0.4 * t)
            # Body angular rates from ZYX Euler rates (yaw=0):
            # wx = d_roll, wy = cos(roll)*d_pitch, wz = -sin(roll)*d_pitch
            gx = d_roll
            gy = math.cos(roll) * d_pitch
            gz = -math.sin(roll) * d_pitch

            ax = -math.sin(pitch) + rng.gauss(0.0, 0.004)
            ay = math.cos(pitch) * math.sin(roll) + rng.gauss(0.0, 0.004)
            az = math.cos(pitch) * math.cos(roll) + rng.gauss(0.0, 0.004)

            # Inject 2.5g landing impact spikes at t = 1.0s and t = 2.0s
            if k in range(100, 105) or k in range(200, 205):
                az += 1.6

            if k == 0:
                # Start aligned at t=0
                for _ in range(300):
                    ahrs.update(-math.sin(pitch), math.cos(pitch) * math.sin(roll), math.cos(pitch) * math.cos(roll), 0.0, 0.0, 0.0, dt)

            r_est, p_est, _ = ahrs.update(ax, ay, az, gx, gy, gz, dt)
            err_sq += math.degrees(r_est - roll) ** 2 + math.degrees(p_est - pitch) ** 2

        rms_deg = math.sqrt(err_sq / (2 * n))
        self.assertGreaterEqual(ahrs.accel_gated_count, 10)
        self.assertLess(rms_deg, 0.5)


if __name__ == "__main__":
    unittest.main()
