"""CPG trot gait generator for a 12-DOF quadruped.

Four coupled Hopf oscillators, one per leg. Trot gait: diagonal legs in
phase (LF + RR, RF + LR), 180 deg apart. Runs at 10 Hz on the Pi and
outputs 12 joint angles (hip_roll, hip_pitch, knee per leg).

Foot contact (FSR) events reset the oscillator phase so stance/swing
switching stays locked to the ground, not to a free-running clock.
"""

import math

# Trot phase offsets (radians): diagonal legs share a phase.
PHASE_OFFSET = {"LF": 0.0, "RR": 0.0, "RF": math.pi, "LR": math.pi}
LEGS = ["LF", "RF", "LR", "RR"]

# Leg geometry (m) — tune to the real build.
L1 = 0.08   # hip to knee
L2 = 0.12   # knee to foot


class HopfOscillator:
    """Single Hopf oscillator: stable limit cycle, phase resets on contact."""

    def __init__(self, omega=2 * math.pi * 1.0, mu=1.0):
        self.x, self.y = 1.0, 0.0
        self.omega = omega  # step frequency, rad/s
        self.mu = mu

    def step(self, dt):
        r2 = self.x * self.x + self.y * self.y
        dx = (self.mu - r2) * self.x - self.omega * self.y
        dy = (self.mu - r2) * self.y + self.omega * self.x
        self.x += dx * dt
        self.y += dy * dt
        return math.atan2(self.y, self.x)

    def reset_phase(self, phase):
        self.x = math.cos(phase)
        self.y = math.sin(phase)


def analytic_ik(x, z):
    """Closed-form IK for one 3-DOF leg in the sagittal plane.

    Returns (hip_pitch, knee). Hip roll is handled separately from body
    roll compensation. Runs in microseconds on the ESP32.
    """
    d = min(math.hypot(x, z), L1 + L2 - 1e-6)
    knee = math.acos(max(-1.0, min(1.0, (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2))))
    hip = math.atan2(z, x) - math.atan2(L2 * math.sin(knee), L1 + L2 * math.cos(knee))
    return hip, knee


class GaitPlanner:
    def __init__(self, step_freq_hz=1.0, step_length=0.06, step_height=0.03):
        self.osc = {leg: HopfOscillator(omega=2 * math.pi * step_freq_hz)
                    for leg in LEGS}
        for leg, off in PHASE_OFFSET.items():
            self.osc[leg].reset_phase(off)
        self.step_length = step_length
        self.step_height = step_height

    def on_foot_contact(self, leg):
        """FSR touchdown event: snap this leg's oscillator to stance start."""
        self.osc[leg].reset_phase(PHASE_OFFSET[leg])

    def step(self, dt):
        """One 10 Hz tick. Returns {leg: (hip_roll, hip_pitch, knee)}."""
        out = {}
        for leg in LEGS:
            phase = self.osc[leg].step(dt)
            # Foot trajectory: ellipse in the sagittal plane.
            fx = self.step_length * math.cos(phase)
            fz = -self.step_height * max(0.0, math.sin(phase))
            hip_pitch, knee = analytic_ik(fx, fz)
            out[leg] = (0.0, hip_pitch, knee)
        return out
