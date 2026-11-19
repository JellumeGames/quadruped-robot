"""Coupled Hopf CPG trot/walk/bound gait generator and 3D foot planner.

Four nonlinear Hopf oscillators coupled via relative phase offsets, integrated
with 4th-order Runge-Kutta (RK4) for unconditional limit-cycle stability.
Foot-sole FSR touchdown events reset the corresponding leg oscillator phase so
stance/swing transitions stay locked to physical ground contact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

from pi.kinematics import (
    HIP_OFFSETS,
    L0,
    LEG_SIDE,
    LEGS,
    NOMINAL_STAND_Z,
    JointAngles,
    Vec3,
    leg_ik,
    whole_body_ik,
)

GAIT_PHASE_OFFSETS: Dict[str, Dict[str, float]] = {
    "trot": {"LF": 0.0, "RR": 0.0, "RF": math.pi, "LR": math.pi},
    "walk": {"LF": 0.0, "RR": 0.5 * math.pi, "RF": math.pi, "LR": 1.5 * math.pi},
    "bound": {"LF": 0.0, "RF": 0.0, "LR": math.pi, "RR": math.pi},
    "stand": {"LF": 0.0, "RF": 0.0, "LR": 0.0, "RR": 0.0},
}

# Preserve compatibility constants from the initial module interface.
PHASE_OFFSET = GAIT_PHASE_OFFSETS["trot"]


def _wrap_2pi(angle: float) -> float:
    return angle % (2.0 * math.pi)


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def analytic_ik(x: float, z: float, side: float = 1.0) -> Tuple[float, float]:
    """Compatibility sagittal IK helper delegating to 3-DOF `pi.kinematics.leg_ik`."""
    res = leg_ik(x, side * L0, z, side=side)
    return res.hip_pitch, res.knee


class CoupledHopfCPG:
    """4-oscillator coupled Hopf CPG with RK4 integration."""

    def __init__(
        self,
        step_freq_hz: float = 1.5,
        gait: str = "trot",
        alpha: float = 8.0,
        coupling: float = 2.5,
        mu: float = 1.0,
    ):
        if gait not in GAIT_PHASE_OFFSETS:
            raise ValueError(f"Unknown gait '{gait}', expected one of {tuple(GAIT_PHASE_OFFSETS)}")
        self.step_freq_hz = step_freq_hz
        self.omega = 2.0 * math.pi * step_freq_hz
        self.gait = gait
        self.alpha = alpha
        self.coupling = coupling
        self.mu = mu
        self.offsets: Dict[str, float] = dict(GAIT_PHASE_OFFSETS[gait])
        self.state: Dict[str, Tuple[float, float]] = {
            leg: (math.cos(self.offsets[leg]), math.sin(self.offsets[leg]))
            for leg in LEGS
        }

    def set_gait(self, gait: str) -> None:
        if gait not in GAIT_PHASE_OFFSETS:
            raise ValueError(f"Unknown gait '{gait}'")
        self.gait = gait
        self.offsets = dict(GAIT_PHASE_OFFSETS[gait])

    def set_frequency(self, step_freq_hz: float) -> None:
        self.step_freq_hz = max(0.0, step_freq_hz)
        self.omega = 2.0 * math.pi * self.step_freq_hz

    def set_phase(self, leg: str, phase_rad: float, radius: float = 1.0) -> None:
        self.state[leg] = (radius * math.cos(phase_rad), radius * math.sin(phase_rad))

    def on_foot_contact(self, leg: str, force: bool = False) -> bool:
        """Snap `leg` oscillator to stance start on late-swing FSR touchdown."""
        phi = self.phase(leg)
        # Accept contact snap during late swing [1.35*pi, 2*pi) or early stance [0, 0.25*pi),
        # or always when force=True.
        if force or phi >= 1.35 * math.pi or phi <= 0.25 * math.pi:
            ref_phase = self.phase("LF") - self.offsets["LF"] + self.offsets[leg]
            # Snap to stance start (local phase = 0.0 -> global phase = offset)
            # when LF is used as reference? No: stance start for `leg` is local phase 0!
            # Keep coupling intact by resetting to current swing-end -> stance-start boundary:
            target = (
                self.offsets[leg]
                if leg == "LF"
                else _wrap_2pi(round((self.phase(leg) - 0.0) / (2.0 * math.pi)) * 2.0 * math.pi)
            )
            # In this coupled formulation, local stance start for `leg` occurs at global
            # phase == 0 mod 2*pi? Wait: let's check!
            # Each oscillator rotates at `omega`, and `phi_i - phi_j = offset[i] - offset[j]`.
            # So leg `i`'s stance starts when `phi_i == 0` (mod 2*pi)!
            _ = ref_phase
            _ = target
            self.set_phase(leg, 0.0, radius=math.sqrt(self.mu))
            return True
        return False

    def phase(self, leg: str) -> float:
        x, y = self.state[leg]
        return _wrap_2pi(math.atan2(y, x))

    def phases(self) -> Dict[str, float]:
        return {leg: self.phase(leg) for leg in LEGS}

    def radii(self) -> Dict[str, float]:
        return {leg: math.hypot(*self.state[leg]) for leg in LEGS}

    def is_swing(self, leg: str) -> bool:
        return self.phase(leg) >= math.pi

    def _deriv(self, state: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        d: Dict[str, Tuple[float, float]] = {}
        k = self.coupling
        for leg_i in LEGS:
            xi, yi = state[leg_i]
            r2 = xi * xi + yi * yi
            dxi = self.alpha * (self.mu - r2) * xi - self.omega * yi
            dyi = self.alpha * (self.mu - r2) * yi + self.omega * xi

            if k != 0.0:
                off_i = self.offsets[leg_i]
                for leg_j in LEGS:
                    if leg_j == leg_i:
                        continue
                    xj, yj = state[leg_j]
                    dphi = off_i - self.offsets[leg_j]
                    cd, sd = math.cos(dphi), math.sin(dphi)
                    target_x = xj * cd - yj * sd
                    target_y = xj * sd + yj * cd
                    dxi += k * (target_x - xi)
                    dyi += k * (target_y - yi)
            d[leg_i] = (dxi, dyi)
        return d

    def _rk4_substep(self, h: float) -> None:
        k1 = self._deriv(self.state)
        s2 = {l: (self.state[l][0] + 0.5 * h * k1[l][0], self.state[l][1] + 0.5 * h * k1[l][1]) for l in LEGS}
        k2 = self._deriv(s2)
        s3 = {l: (self.state[l][0] + 0.5 * h * k2[l][0], self.state[l][1] + 0.5 * h * k2[l][1]) for l in LEGS}
        k3 = self._deriv(s3)
        s4 = {l: (self.state[l][0] + h * k3[l][0], self.state[l][1] + h * k3[l][1]) for l in LEGS}
        k4 = self._deriv(s4)

        self.state = {
            l: (
                self.state[l][0] + (h / 6.0) * (k1[l][0] + 2.0 * k2[l][0] + 2.0 * k3[l][0] + k4[l][0]),
                self.state[l][1] + (h / 6.0) * (k1[l][1] + 2.0 * k2[l][1] + 2.0 * k3[l][1] + k4[l][1]),
            )
            for l in LEGS
        }

    def step(self, dt: float) -> Dict[str, float]:
        """Advance coupled CPG by `dt` seconds using sub-stepped RK4."""
        if dt <= 0.0:
            return self.phases()
        max_sub = 0.005
        n_steps = max(1, int(math.ceil(dt / max_sub)))
        h = dt / n_steps
        for _ in range(n_steps):
            self._rk4_substep(h)
        return self.phases()


@dataclass(frozen=True)
class GaitOutput:
    angles: Dict[str, JointAngles]
    feet_world: Dict[str, Vec3]
    feet_hip: Dict[str, Vec3]
    swing_mask: Dict[str, bool]
    phases: Dict[str, float]


class GaitPlanner:
    """High-level velocity-commanded 3D gait planner + IMU attitude leveling."""

    def __init__(
        self,
        step_freq_hz: float = 1.5,
        step_height: float = 0.032,
        stand_height: float = abs(NOMINAL_STAND_Z),
        gait: str = "trot",
        max_stride: float = 0.09,
        kp_att: float = 0.85,
        kd_att: float = 0.12,
    ):
        self.cpg = CoupledHopfCPG(step_freq_hz=step_freq_hz, gait=gait)
        self.step_height = step_height
        self.stand_height = stand_height
        self.max_stride = max_stride
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

    def set_velocity(self, vx: float, vy: float = 0.0, yaw_rate: float = 0.0) -> None:
        self.vx = vx
        self.vy = vy
        self.yaw_rate = yaw_rate

    def set_gait(self, gait: str) -> None:
        self.cpg.set_gait(gait)

    def on_foot_contact(self, leg: str, force: bool = False) -> bool:
        return self.cpg.on_foot_contact(leg, force=force)

    def _foot_world_target(self, leg: str, phi: float) -> Vec3:
        hx, hy, _ = HIP_OFFSETS[leg]
        base_x = hx
        base_y = hy + LEG_SIDE[leg] * L0
        base_z = -self.stand_height

        if self.cpg.gait == "stand" or self.cpg.step_freq_hz <= 1e-6:
            return (base_x, base_y, base_z)

        t_stance = 0.5 / self.cpg.step_freq_hz
        vx_leg = self.vx - self.yaw_rate * base_y
        vy_leg = self.vy + self.yaw_rate * base_x

        stride_x = _clamp(vx_leg * t_stance, -self.max_stride, self.max_stride)
        stride_y = _clamp(vy_leg * t_stance, -0.5 * self.max_stride, 0.5 * self.max_stride)

        if phi < math.pi:
            # Stance: linear sweep from +0.5*stride to -0.5*stride on the ground.
            u = phi / math.pi
            s = 0.5 - u
            dz = 0.0
        else:
            # Swing: raised-cosine vertical lift (zero velocity at liftoff & landing)
            # and cosine-eased horizontal swing from -0.5*stride to +0.5*stride.
            u = (phi - math.pi) / math.pi
            s = -0.5 * math.cos(math.pi * u)
            dz = self.step_height * 0.5 * (1.0 - math.cos(2.0 * math.pi * u))

        return (base_x + s * stride_x, base_y + s * stride_y, base_z + dz)

    def step(
        self,
        dt: float,
        imu_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        imu_gyro: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> GaitOutput:
        phases = self.cpg.step(dt)
        feet_world = {leg: self._foot_world_target(leg, phases[leg]) for leg in LEGS}

        # Restoring body attitude command: tilt opposite to measured roll/pitch & gyro rate.
        comp_roll = _clamp(-(self.kp_att * imu_rpy[0] + self.kd_att * imu_gyro[0]), -0.30, 0.30)
        comp_pitch = _clamp(-(self.kp_att * imu_rpy[1] + self.kd_att * imu_gyro[1]), -0.30, 0.30)

        ik_out = whole_body_ik(
            feet_world=feet_world,
            body_pos=(0.0, 0.0, 0.0),
            body_rpy=(comp_roll, comp_pitch, 0.0),
            clamp_joints=True,
        )
        angles = {leg: ik_out[leg].as_tuple() for leg in LEGS}
        feet_hip = {
            leg: (
                feet_world[leg][0] - HIP_OFFSETS[leg][0],
                feet_world[leg][1] - HIP_OFFSETS[leg][1],
                feet_world[leg][2] - HIP_OFFSETS[leg][2],
            )
            for leg in LEGS
        }
        swing_mask = {leg: self.cpg.is_swing(leg) for leg in LEGS}
        return GaitOutput(
            angles=angles,
            feet_world=feet_world,
            feet_hip=feet_hip,
            swing_mask=swing_mask,
            phases=phases,
        )
