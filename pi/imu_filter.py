"""MPU6050 14-byte raw register decoding, gyro calibration, and Mahony AHRS.

Matches the ESP32 C++ implementation in `esp32/imu_mpu6050.cpp` bit-for-bit
in register layout, scale factors, shock gating, and ZYX Euler output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable, List, Tuple

MPU6050_I2C_ADDR = 0x68
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B
REG_PWR_MGMT_1 = 0x6B
REG_WHO_AM_I = 0x75

ACCEL_SCALE_4G = 8192.0   # LSB / g for +/- 4g
GYRO_SCALE_500DPS = 65.5  # LSB / (deg/s) for +/- 500 deg/s
DEG2RAD = math.pi / 180.0
GRAVITY = 9.80665


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


@dataclass(frozen=True)
class ImuRawSample:
    ax_g: float
    ay_g: float
    az_g: float
    temp_c: float
    gx_rads: float
    gy_rads: float
    gz_rads: float


def decode_mpu6050_frame(raw14: bytes) -> ImuRawSample:
    """Decode 14 consecutive big-endian bytes starting at ACCEL_XOUT_H (0x3B)."""
    if len(raw14) != 14:
        raise ValueError(f"Expected 14 MPU6050 bytes, got {len(raw14)}")
    ax_raw, ay_raw, az_raw, temp_raw, gx_raw, gy_raw, gz_raw = struct.unpack(">7h", raw14)
    return ImuRawSample(
        ax_g=ax_raw / ACCEL_SCALE_4G,
        ay_g=ay_raw / ACCEL_SCALE_4G,
        az_g=az_raw / ACCEL_SCALE_4G,
        temp_c=(temp_raw / 340.0) + 36.53,
        gx_rads=(gx_raw / GYRO_SCALE_500DPS) * DEG2RAD,
        gy_rads=(gy_raw / GYRO_SCALE_500DPS) * DEG2RAD,
        gz_rads=(gz_raw / GYRO_SCALE_500DPS) * DEG2RAD,
    )


def encode_mpu6050_frame(
    ax_g: float,
    ay_g: float,
    az_g: float,
    gx_rads: float,
    gy_rads: float,
    gz_rads: float,
    temp_c: float = 25.0,
) -> bytes:
    """Encode physical IMU readings into 14 big-endian MPU6050 register bytes."""
    def _i16(val: float) -> int:
        return int(round(_clamp(val, -32768.0, 32767.0)))

    ax_raw = _i16(ax_g * ACCEL_SCALE_4G)
    ay_raw = _i16(ay_g * ACCEL_SCALE_4G)
    az_raw = _i16(az_g * ACCEL_SCALE_4G)
    temp_raw = _i16((temp_c - 36.53) * 340.0)
    gx_raw = _i16((gx_rads / DEG2RAD) * GYRO_SCALE_500DPS)
    gy_raw = _i16((gy_rads / DEG2RAD) * GYRO_SCALE_500DPS)
    gz_raw = _i16((gz_rads / DEG2RAD) * GYRO_SCALE_500DPS)
    return struct.pack(">7h", ax_raw, ay_raw, az_raw, temp_raw, gx_raw, gy_raw, gz_raw)


class GyroCalibrator:
    """Collects stationary IMU samples to estimate 3-axis gyroscope bias."""

    def __init__(self, target_samples: int = 100, max_accel_std_g: float = 0.08):
        self.target_samples = target_samples
        self.max_accel_std_g = max_accel_std_g
        self._gyros: List[Tuple[float, float, float]] = []
        self._accel_norms: List[float] = []
        self.bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.calibrated: bool = False

    def add_sample(self, sample: ImuRawSample) -> bool:
        """Add one stationary sample. Returns True once calibration succeeds."""
        if self.calibrated:
            return True
        a_norm = math.sqrt(sample.ax_g ** 2 + sample.ay_g ** 2 + sample.az_g ** 2)
        self._gyros.append((sample.gx_rads, sample.gy_rads, sample.gz_rads))
        self._accel_norms.append(a_norm)
        if len(self._gyros) < self.target_samples:
            return False

        mean_a = sum(self._accel_norms) / len(self._accel_norms)
        var_a = sum((v - mean_a) ** 2 for v in self._accel_norms) / len(self._accel_norms)
        if math.sqrt(var_a) > self.max_accel_std_g:
            # Robot was disturbed during calibration: reset window and retry.
            self._gyros.clear()
            self._accel_norms.clear()
            return False

        n = float(len(self._gyros))
        bx = sum(g[0] for g in self._gyros) / n
        by = sum(g[1] for g in self._gyros) / n
        bz = sum(g[2] for g in self._gyros) / n
        self.bias = (bx, by, bz)
        self.calibrated = True
        return True

    def calibrate_batch(self, samples: Iterable[ImuRawSample]) -> Tuple[float, float, float]:
        for s in samples:
            if self.add_sample(s):
                break
        return self.bias


class MahonyAHRS:
    """Mahony quaternion AHRS with accelerometer impact/shock gating.

    When ||a|| deviates from 1.0 g by more than `accel_gate_g` (e.g. foot
    landing impact spike), accelerometer correction is skipped and attitude is
    propagated purely by gyroscope integration so roll/pitch do not jerk.
    """

    def __init__(self, kp: float = 2.0, ki: float = 0.05, accel_gate_g: float = 0.25):
        self.kp = kp
        self.ki = ki
        self.accel_gate_g = accel_gate_g
        self.qw: float = 1.0
        self.qx: float = 0.0
        self.qy: float = 0.0
        self.qz: float = 0.0
        self.ex_int: float = 0.0
        self.ey_int: float = 0.0
        self.ez_int: float = 0.0
        self.accel_gated_count: int = 0

    def reset(self) -> None:
        self.qw, self.qx, self.qy, self.qz = 1.0, 0.0, 0.0, 0.0
        self.ex_int, self.ey_int, self.ez_int = 0.0, 0.0, 0.0
        self.accel_gated_count = 0

    def update(
        self,
        ax_g: float,
        ay_g: float,
        az_g: float,
        gx_rads: float,
        gy_rads: float,
        gz_rads: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        """Advance filter by `dt` seconds and return (roll, pitch, yaw) in rad."""
        if dt <= 0.0:
            return self.euler()

        gx, gy, gz = gx_rads, gy_rads, gz_rads
        a_norm = math.sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g)

        if a_norm > 1e-6 and abs(a_norm - 1.0) <= self.accel_gate_g:
            ax = ax_g / a_norm
            ay = ay_g / a_norm
            az = az_g / a_norm

            # Estimated gravity direction in body frame from unit quaternion.
            vx = 2.0 * (self.qx * self.qz - self.qw * self.qy)
            vy = 2.0 * (self.qw * self.qx + self.qy * self.qz)
            vz = 1.0 - 2.0 * (self.qx * self.qx + self.qy * self.qy)

            # Error is cross product between measured and estimated gravity.
            ex = ay * vz - az * vy
            ey = az * vx - ax * vz
            ez = ax * vy - ay * vx

            if self.ki > 0.0:
                self.ex_int += ex * dt
                self.ey_int += ey * dt
                self.ez_int += ez * dt
                gx += self.ki * self.ex_int
                gy += self.ki * self.ey_int
                gz += self.ki * self.ez_int

            gx += self.kp * ex
            gy += self.kp * ey
            gz += self.kp * ez
        else:
            self.accel_gated_count += 1

        # Quaternion rate of change: q_dot = 0.5 * q (*) [0, gx, gy, gz]
        qw, qx, qy, qz = self.qw, self.qx, self.qy, self.qz
        half_dt = 0.5 * dt
        dqw = (-qx * gx - qy * gy - qz * gz) * half_dt
        dqx = (qw * gx + qy * gz - qz * gy) * half_dt
        dqy = (qw * gy - qx * gz + qz * gx) * half_dt
        dqz = (qw * gz + qx * gy - qy * gx) * half_dt

        qw += dqw
        qx += dqx
        qy += dqy
        qz += dqz

        q_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if q_norm > 1e-9:
            inv = 1.0 / q_norm
            self.qw = qw * inv
            self.qx = qx * inv
            self.qy = qy * inv
            self.qz = qz * inv

        return self.euler()

    def euler(self) -> Tuple[float, float, float]:
        """Return (roll, pitch, yaw) in radians using ZYX aerospace convention."""
        qw, qx, qy, qz = self.qw, self.qx, self.qy, self.qz
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = _clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (roll, pitch, yaw)
