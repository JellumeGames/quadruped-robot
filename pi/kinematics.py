"""3-DOF leg kinematics, analytical Jacobian, and whole-body IK.

Coordinate convention (right-handed, body frame):
    +X: forward (nose)
    +Y: left
    +Z: up

Leg ordering:
    LF (front-left,  side = +1)
    RF (front-right, side = -1)
    LR (rear-left,   side = +1)
    RR (rear-right,  side = -1)

Joint ordering per leg:
    (hip_roll, hip_pitch, knee) in radians.

Must stay numerically aligned with `esp32/kinematics.h`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

L0: float = 0.045  # Coxa / hip-roll lateral offset (m)
L1: float = 0.080  # Thigh length (m)
L2: float = 0.120  # Shin length (m)

BODY_LENGTH: float = 0.240
BODY_WIDTH: float = 0.140
NOMINAL_STAND_Z: float = -0.155

LEGS: Tuple[str, ...] = ("LF", "RF", "LR", "RR")
LEG_SIDE: Dict[str, float] = {"LF": 1.0, "RF": -1.0, "LR": 1.0, "RR": -1.0}
HIP_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "LF": (0.5 * BODY_LENGTH, 0.5 * BODY_WIDTH, 0.0),
    "RF": (0.5 * BODY_LENGTH, -0.5 * BODY_WIDTH, 0.0),
    "LR": (-0.5 * BODY_LENGTH, 0.5 * BODY_WIDTH, 0.0),
    "RR": (-0.5 * BODY_LENGTH, -0.5 * BODY_WIDTH, 0.0),
}

HIP_ROLL_LIMIT: Tuple[float, float] = (-0.60, 0.60)
HIP_PITCH_LIMIT: Tuple[float, float] = (-1.20, 1.40)
KNEE_LIMIT: Tuple[float, float] = (-2.45, -0.35)

Vec3 = Tuple[float, float, float]
JointAngles = Tuple[float, float, float]


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


@dataclass(frozen=True)
class IKResult:
    hip_roll: float
    hip_pitch: float
    knee: float
    clamped: bool = False

    def as_tuple(self) -> JointAngles:
        return (self.hip_roll, self.hip_pitch, self.knee)


def clamp_leg_joints(hip_roll: float, hip_pitch: float, knee: float) -> Tuple[JointAngles, bool]:
    """Clamp joint angles into mechanical limits; report whether clamping occurred."""
    r = _clamp(hip_roll, HIP_ROLL_LIMIT[0], HIP_ROLL_LIMIT[1])
    p = _clamp(hip_pitch, HIP_PITCH_LIMIT[0], HIP_PITCH_LIMIT[1])
    k = _clamp(knee, KNEE_LIMIT[0], KNEE_LIMIT[1])
    changed = (r != hip_roll) or (p != hip_pitch) or (k != knee)
    return (r, p, k), changed


def leg_fk(hip_roll: float, hip_pitch: float, knee: float, side: float) -> Vec3:
    """Compute foot position (x, y, z) in the hip frame from 3 joint angles."""
    y_s = side * L0
    x_s = L1 * math.sin(hip_pitch) + L2 * math.sin(hip_pitch + knee)
    z_s = -L1 * math.cos(hip_pitch) - L2 * math.cos(hip_pitch + knee)

    c_r = math.cos(hip_roll)
    s_r = math.sin(hip_roll)
    y = y_s * c_r - z_s * s_r
    z = y_s * s_r + z_s * c_r
    return (x_s, y, z)


def leg_ik(
    x: float,
    y: float,
    z: float,
    side: float,
    knee_sign: float = -1.0,
    clamp_joints: bool = False,
) -> IKResult:
    """Closed-form 3-DOF analytical inverse kinematics in the hip frame.

    Gracefully clamps targets outside the reachable workspace so no NaN/domain
    error can occur during extreme dynamic maneuvers.
    """
    y_s = side * L0
    clamped = False

    # 1. Solve hip_roll in the YZ plane, guarding the inner cylinder r_yz <= L0.
    r_yz2 = y * y + z * z
    min_yz2 = L0 * L0 + 1e-6
    if r_yz2 < min_yz2:
        clamped = True
        r_yz = math.sqrt(r_yz2)
        target_r = math.sqrt(min_yz2)
        if r_yz < 1e-9:
            y = y_s
            z = -1e-3
        else:
            scale = target_r / r_yz
            y *= scale
            z *= scale
        r_yz2 = y * y + z * z

    z_s = -math.sqrt(max(0.0, r_yz2 - L0 * L0))
    hip_roll = math.atan2(y_s * z - z_s * y, y_s * y + z_s * z)

    # 2. Solve sagittal (hip_pitch, knee) in the rotated hip plane (x_s, z_s).
    x_s = x
    d_min = abs(L1 - L2) + 1e-4
    d_max = (L1 + L2) - 1e-4
    d = math.hypot(x_s, z_s)
    if d < d_min or d > d_max:
        clamped = True
        d_clamped = _clamp(d, d_min, d_max)
        if d < 1e-9:
            x_s = 0.0
            z_s = -d_clamped
        else:
            scale = d_clamped / d
            x_s *= scale
            z_s *= scale
        d = d_clamped

    cos_k = _clamp((d * d - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0)
    knee = (-1.0 if knee_sign < 0.0 else 1.0) * math.acos(cos_k)

    a = L1 + L2 * math.cos(knee)
    b = L2 * math.sin(knee)
    hip_pitch = math.atan2(a * x_s + b * z_s, b * x_s - a * z_s)

    if clamp_joints:
        (hip_roll, hip_pitch, knee), joint_clamped = clamp_leg_joints(hip_roll, hip_pitch, knee)
        clamped = clamped or joint_clamped

    return IKResult(hip_roll=hip_roll, hip_pitch=hip_pitch, knee=knee, clamped=clamped)


def leg_jacobian(hip_roll: float, hip_pitch: float, knee: float, side: float) -> List[List[float]]:
    """Analytical 3x3 foot Jacobian J = d(x, y, z) / d(hip_roll, hip_pitch, knee)."""
    y_s = side * L0
    c0, s0 = math.cos(hip_roll), math.sin(hip_roll)
    c1, s1 = math.cos(hip_pitch), math.sin(hip_pitch)
    c12, s12 = math.cos(hip_pitch + knee), math.sin(hip_pitch + knee)

    x_s = L1 * s1 + L2 * s12
    z_s = -L1 * c1 - L2 * c12
    y = y_s * c0 - z_s * s0
    z = y_s * s0 + z_s * c0

    dz_dq1 = x_s
    dz_dq2 = L2 * s12

    return [
        [0.0, L1 * c1 + L2 * c12, L2 * c12],
        [-z, -s0 * dz_dq1, -s0 * dz_dq2],
        [y, c0 * dz_dq1, c0 * dz_dq2],
    ]


def rot_zyx(roll: float, pitch: float, yaw: float) -> Tuple[Vec3, Vec3, Vec3]:
    """3x3 body-to-world rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mat_t_vec(mat: Tuple[Vec3, Vec3, Vec3], v: Vec3) -> Vec3:
    return (
        mat[0][0] * v[0] + mat[1][0] * v[1] + mat[2][0] * v[2],
        mat[0][1] * v[0] + mat[1][1] * v[1] + mat[2][1] * v[2],
        mat[0][2] * v[0] + mat[1][2] * v[1] + mat[2][2] * v[2],
    )


def nominal_stance_feet_world(stand_height: float = abs(NOMINAL_STAND_Z)) -> Dict[str, Vec3]:
    """Neutral standing foot positions in world frame directly beneath each hip."""
    return {
        leg: (HIP_OFFSETS[leg][0], HIP_OFFSETS[leg][1] + LEG_SIDE[leg] * L0, -abs(stand_height))
        for leg in LEGS
    }


def whole_body_ik(
    feet_world: Dict[str, Vec3],
    body_pos: Vec3 = (0.0, 0.0, 0.0),
    body_rpy: Vec3 = (0.0, 0.0, 0.0),
    clamp_joints: bool = False,
) -> Dict[str, IKResult]:
    """Solve 12-DOF IK from desired world foot targets and body pose (pos, RPY)."""
    r_bw = rot_zyx(body_rpy[0], body_rpy[1], body_rpy[2])
    out: Dict[str, IKResult] = {}
    for leg in LEGS:
        fw = feet_world[leg]
        rel = (fw[0] - body_pos[0], fw[1] - body_pos[1], fw[2] - body_pos[2])
        p_body = _mat_t_vec(r_bw, rel)
        hx, hy, hz = HIP_OFFSETS[leg]
        p_hip = (p_body[0] - hx, p_body[1] - hy, p_body[2] - hz)
        out[leg] = leg_ik(p_hip[0], p_hip[1], p_hip[2], LEG_SIDE[leg], clamp_joints=clamp_joints)
    return out


def flatten_angles(angles_by_leg: Dict[str, JointAngles], legs: Iterable[str] = LEGS) -> List[float]:
    """Flatten {leg: (hip_roll, hip_pitch, knee)} into the 12-float wire order."""
    flat: List[float] = []
    for leg in legs:
        flat.extend(angles_by_leg[leg])
    return flat
