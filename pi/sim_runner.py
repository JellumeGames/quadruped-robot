"""Closed-loop 100 Hz quadruped simulation: CPG + IK + Mahony IMU + noisy UART.

Runnable directly:
    python -m pi.sim_runner --duration 4.0 --inject-disturbance --inject-dropout
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random
from typing import List, Tuple

from pi.cpg import GaitPlanner
from pi.imu_filter import (
    GyroCalibrator,
    MahonyAHRS,
    decode_mpu6050_frame,
    encode_mpu6050_frame,
)
from pi.kinematics import (
    BODY_LENGTH,
    BODY_WIDTH,
    HIP_OFFSETS,
    L0,
    LEG_SIDE,
    LEGS,
    NOMINAL_STAND_Z,
    flatten_angles,
    leg_fk,
    leg_ik,
)
from pi.uart_link import (
    CMD_JOINTS,
    CMD_MODE,
    CMD_TELEMETRY,
    MODE_TROT,
    STATE_ACTIVE,
    STATE_TIMEOUT_SAFE,
    FrameParser,
    ReliableCommander,
    Telemetry,
    pack_ack,
    pack_joints,
    pack_telemetry,
    unpack_joints,
    unpack_mode,
    unpack_telemetry,
)


@dataclass(frozen=True)
class Checkpoint:
    t_s: float
    roll_deg: float
    pitch_deg: float
    lf_phase_deg: float
    diag_sync_err_deg: float
    esp_state: str
    fk_ik_err_mm: float


@dataclass(frozen=True)
class SimSummary:
    duration_s: float
    ticks: int
    max_ik_fk_err_mm: float
    mean_ik_fk_err_mm: float
    trot_diag_sync_err_deg: float
    trot_anti_phase_err_deg: float
    peak_disturbance_roll_deg: float
    post_recovery_roll_rms_deg: float
    imu_est_err_rms_deg: float
    tx_frames: int
    rx_valid_frames: int
    crc_rejected_frames: int
    ack_count: int
    watchdog_triggered: bool
    watchdog_recovered: bool
    fsr_resets: int
    checkpoints: List[Checkpoint]


def _angle_diff_deg(a_rad: float, b_rad: float) -> float:
    diff = (a_rad - b_rad + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(diff))


def _state_name(state: int) -> str:
    return {
        0: "IDLE",
        1: "STAND",
        STATE_ACTIVE: "ACTIVE",
        STATE_TIMEOUT_SAFE: "SAFE_STANCE",
        4: "ESTOP",
    }.get(state, f"STATE_{state}")


def run_simulation(
    duration_s: float = 4.0,
    step_freq_hz: float = 1.5,
    vx: float = 0.18,
    vy: float = 0.0,
    yaw_rate: float = 0.0,
    inject_disturbance: bool = True,
    inject_dropout: bool = True,
    seed: int = 42,
) -> SimSummary:
    rng = random.Random(seed)
    dt = 0.01  # 100 Hz ESP32 loop
    ticks = int(round(duration_s / dt))

    planner = GaitPlanner(step_freq_hz=step_freq_hz, step_height=0.032, gait="trot")
    planner.set_velocity(vx=vx, vy=vy, yaw_rate=yaw_rate)

    # Simulated MPU6050 hardware bias & Mahony estimator.
    true_gyro_bias = (0.018, -0.014, 0.009)
    calibrator = GyroCalibrator(target_samples=40)
    for _ in range(40):
        raw14 = encode_mpu6050_frame(
            0.0 + rng.gauss(0.0, 0.005),
            0.0 + rng.gauss(0.0, 0.005),
            1.0 + rng.gauss(0.0, 0.005),
            true_gyro_bias[0] + rng.gauss(0.0, 0.002),
            true_gyro_bias[1] + rng.gauss(0.0, 0.002),
            true_gyro_bias[2] + rng.gauss(0.0, 0.002),
        )
        calibrator.add_sample(decode_mpu6050_frame(raw14))

    ahrs = MahonyAHRS(kp=2.2, ki=0.05, accel_gate_g=0.25)

    # Wire link (Pi <-> ESP32)
    commander = ReliableCommander(retry_interval_s=0.05, max_retries=5)
    esp_parser = FrameParser()
    pi_parser = FrameParser()

    # ESP32 state
    safe_pose = [
        ang
        for leg in LEGS
        for ang in leg_ik(0.0, LEG_SIDE[leg] * L0, NOMINAL_STAND_Z, LEG_SIDE[leg]).as_tuple()
    ]
    esp_target_angles = list(safe_pose)
    esp_current_angles = list(safe_pose)
    esp_state = STATE_ACTIVE
    last_esp_frame_s = 0.0

    # Body rotational plant (roll, pitch)
    roll_true = 0.0
    pitch_true = 0.0
    d_roll = 0.0
    d_pitch = 0.0

    # Initial mode handshake
    _, mode_wire = commander.send_mode(MODE_TROT, now_s=0.0)
    for cmd, seq, payload in esp_parser.feed(mode_wire):
        if cmd == CMD_MODE and unpack_mode(payload) == MODE_TROT:
            ack_wire = pack_ack(CMD_MODE, 0, seq=seq)
            for p_cmd, p_seq, p_pay in pi_parser.feed(ack_wire):
                commander.on_frame(p_cmd, p_seq, p_pay)

    # Metrics
    ik_fk_errors_mm: List[float] = []
    imu_est_errors_deg: List[float] = []
    post_recovery_rolls_deg: List[float] = []
    peak_disturbance_roll_deg = 0.0
    tx_frames = 1
    fsr_resets = 0
    watchdog_triggered = False
    watchdog_recovered = False
    prev_swing = {leg: False for leg in LEGS}
    last_telem = Telemetry(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (0.5, 0.5, 0.5, 0.5), STATE_ACTIVE, 0)
    checkpoints: List[Checkpoint] = []

    trot_diag_err = 0.0
    trot_anti_err = 0.0

    for k in range(ticks):
        t_s = k * dt
        in_dropout = inject_dropout and (3.00 <= t_s < 3.16)

        # 1. External push disturbance + foot landing impact spikes
        dist_roll_acc = 0.0
        dist_pitch_acc = 0.0
        if inject_disturbance and (1.50 <= t_s < 1.65):
            # Strong lateral push producing noticeable ~8-11 deg transient roll
            dist_roll_acc = 45.0
            dist_pitch_acc = -20.0

        # 2. Compute true body restoring moment from executed ESP32 leg lengths
        z_by_leg = {}
        for idx, leg in enumerate(LEGS):
            q = esp_current_angles[idx * 3 : idx * 3 + 3]
            z_by_leg[leg] = leg_fk(q[0], q[1], q[2], LEG_SIDE[leg])[2]

        z_left = 0.5 * (z_by_leg["LF"] + z_by_leg["LR"])
        z_right = 0.5 * (z_by_leg["RF"] + z_by_leg["RR"])
        z_front = 0.5 * (z_by_leg["LF"] + z_by_leg["RF"])
        z_rear = 0.5 * (z_by_leg["LR"] + z_by_leg["RR"])

        # Longer right legs push right side up (-roll); longer front legs push nose up (-pitch).
        roll_from_legs = (z_right - z_left) / BODY_WIDTH
        pitch_from_legs = (z_front - z_rear) / BODY_LENGTH

        wn = 10.0
        zeta = 0.75
        dd_roll = (wn * wn) * (roll_from_legs - roll_true) - 2.0 * zeta * wn * d_roll + dist_roll_acc
        dd_pitch = (wn * wn) * (pitch_from_legs - pitch_true) - 2.0 * zeta * wn * d_pitch + dist_pitch_acc
        d_roll += dd_roll * dt
        d_pitch += dd_pitch * dt
        roll_true += d_roll * dt
        pitch_true += d_pitch * dt

        if abs(math.degrees(roll_true)) > peak_disturbance_roll_deg:
            peak_disturbance_roll_deg = abs(math.degrees(roll_true))
        if inject_disturbance and (2.05 <= t_s <= 2.95):
            post_recovery_rolls_deg.append(math.degrees(roll_true))

        # 3. Synthesize 14-byte MPU6050 register burst from true body kinematics
        impact_spike_g = 0.0
        for leg in LEGS:
            # Landing impact spike on swing -> stance touchdown
            is_sw = planner.cpg.is_swing(leg)
            if prev_swing[leg] and not is_sw:
                impact_spike_g = 1.6  # total |a| ~ 2.6g, triggers Mahony shock gate!
                if planner.on_foot_contact(leg):
                    fsr_resets += 1
            prev_swing[leg] = is_sw

        ax_true = -math.sin(pitch_true)
        ay_true = math.cos(pitch_true) * math.sin(roll_true)
        az_true = math.cos(pitch_true) * math.cos(roll_true) + impact_spike_g

        raw14 = encode_mpu6050_frame(
            ax_true + rng.gauss(0.0, 0.008),
            ay_true + rng.gauss(0.0, 0.008),
            az_true + rng.gauss(0.0, 0.008),
            d_roll + true_gyro_bias[0] + rng.gauss(0.0, 0.002),
            d_pitch + true_gyro_bias[1] + rng.gauss(0.0, 0.002),
            0.0 + true_gyro_bias[2] + rng.gauss(0.0, 0.002),
        )
        sample = decode_mpu6050_frame(raw14)
        gx = sample.gx_rads - calibrator.bias[0]
        gy = sample.gy_rads - calibrator.bias[1]
        gz = sample.gz_rads - calibrator.bias[2]
        est_roll, est_pitch, est_yaw = ahrs.update(
            sample.ax_g, sample.ay_g, sample.az_g, gx, gy, gz, dt
        )
        imu_err = math.degrees(math.hypot(est_roll - roll_true, est_pitch - pitch_true))
        imu_est_errors_deg.append(imu_err)

        # 4. Pi planner + 50 Hz UART TX (even ticks)
        if k % 2 == 0:
            gait_out = planner.step(
                0.02,
                imu_rpy=(last_telem.roll_rad, last_telem.pitch_rad, last_telem.yaw_rad),
                imu_gyro=(last_telem.gx_rads, last_telem.gy_rads, last_telem.gz_rads),
            )

            # Check 3-DOF IK -> FK round-trip fidelity for every leg
            for leg in LEGS:
                fk_hip = leg_fk(*gait_out.angles[leg], LEG_SIDE[leg])
                ik_re = leg_ik(*fk_hip, LEG_SIDE[leg])
                fk_round = leg_fk(*ik_re.as_tuple(), LEG_SIDE[leg])
                err_m = math.sqrt(
                    (fk_hip[0] - fk_round[0]) ** 2
                    + (fk_hip[1] - fk_round[1]) ** 2
                    + (fk_hip[2] - fk_round[2]) ** 2
                )
                ik_fk_errors_mm.append(err_m * 1000.0)

            if not in_dropout:
                wire = pack_joints(flatten_angles(gait_out.angles), seq=k & 0xFF)
                tx_frames += 1

                # Every 15th frame, prepend false header + corrupted CRC junk to stress resync
                if k % 30 == 0 and k > 0:
                    bad_frame = bytearray(pack_joints(flatten_angles(gait_out.angles), seq=99))
                    bad_frame[6] ^= 0xFF  # corrupt payload -> CRC mismatch
                    wire = b"\xAA\x55\x05\x01\x00\xDE\xAD\x0D\x0A" + bytes(bad_frame) + wire

                for cmd, seq, payload in esp_parser.feed(wire):
                    if cmd == CMD_JOINTS:
                        esp_target_angles = unpack_joints(payload)
                        esp_state = STATE_ACTIVE
                        last_esp_frame_s = t_s
                        if watchdog_triggered:
                            watchdog_recovered = True

        # 5. ESP32 100 Hz watchdog & servo tracking
        if esp_state == STATE_ACTIVE and (t_s - last_esp_frame_s) > 0.10:
            esp_state = STATE_TIMEOUT_SAFE
            esp_target_angles = list(safe_pose)
            watchdog_triggered = True

        for i in range(12):
            esp_current_angles[i] += (esp_target_angles[i] - esp_current_angles[i]) * 0.5

        # 6. ESP32 50 Hz telemetry TX back to Pi
        if k % 2 == 1:
            fsr_vals = tuple(0.05 if planner.cpg.is_swing(leg) else 0.72 for leg in LEGS)
            telem_pkt = Telemetry(
                roll_rad=est_roll,
                pitch_rad=est_pitch,
                yaw_rad=est_yaw,
                gx_rads=gx,
                gy_rads=gy,
                gz_rads=gz,
                fsr=fsr_vals,  # type: ignore[arg-type]
                controller_state=esp_state,
                rx_dropped_frames=min(255, esp_parser.dropped_frames),
            )
            for cmd, seq, payload in pi_parser.feed(pack_telemetry(telem_pkt, seq=k & 0xFF)):
                if cmd == CMD_TELEMETRY:
                    last_telem = unpack_telemetry(payload)

        # Record gait phase sync near steady trot
        phases = planner.cpg.phases()
        diag_err = max(
            _angle_diff_deg(phases["LF"], phases["RR"]),
            _angle_diff_deg(phases["RF"], phases["LR"]),
        )
        anti_err = _angle_diff_deg(phases["LF"] - phases["RF"], math.pi)
        if abs(t_s - 1.0) < 0.01:
            trot_diag_err = diag_err
            trot_anti_err = anti_err

        if k in (50, 155, 220, 312, ticks - 1):
            checkpoints.append(
                Checkpoint(
                    t_s=t_s,
                    roll_deg=math.degrees(roll_true),
                    pitch_deg=math.degrees(pitch_true),
                    lf_phase_deg=math.degrees(phases["LF"]),
                    diag_sync_err_deg=diag_err,
                    esp_state=_state_name(esp_state),
                    fk_ik_err_mm=ik_fk_errors_mm[-1] if ik_fk_errors_mm else 0.0,
                )
            )

    _ = HIP_OFFSETS
    return SimSummary(
        duration_s=duration_s,
        ticks=ticks,
        max_ik_fk_err_mm=max(ik_fk_errors_mm) if ik_fk_errors_mm else 0.0,
        mean_ik_fk_err_mm=(sum(ik_fk_errors_mm) / len(ik_fk_errors_mm)) if ik_fk_errors_mm else 0.0,
        trot_diag_sync_err_deg=trot_diag_err,
        trot_anti_phase_err_deg=trot_anti_err,
        peak_disturbance_roll_deg=peak_disturbance_roll_deg,
        post_recovery_roll_rms_deg=(
            math.sqrt(sum(r * r for r in post_recovery_rolls_deg) / len(post_recovery_rolls_deg))
            if post_recovery_rolls_deg
            else 0.0
        ),
        imu_est_err_rms_deg=math.sqrt(sum(e * e for e in imu_est_errors_deg) / len(imu_est_errors_deg)),
        tx_frames=tx_frames,
        rx_valid_frames=esp_parser.valid_frames,
        crc_rejected_frames=esp_parser.dropped_frames,
        ack_count=commander.acked_count,
        watchdog_triggered=watchdog_triggered,
        watchdog_recovered=watchdog_recovered,
        fsr_resets=fsr_resets,
        checkpoints=checkpoints,
    )


def format_report(s: SimSummary) -> str:
    lines = [
        "=== Quadruped Closed-Loop Simulation Report (100 Hz) ===",
        f"Duration           : {s.duration_s:.2f} s ({s.ticks} ticks)",
        f"3-DOF IK/FK Error  : mean={s.mean_ik_fk_err_mm:.6f} mm, max={s.max_ik_fk_err_mm:.6f} mm",
        f"CPG Trot Sync @1.0s: diag_err={s.trot_diag_sync_err_deg:.3f} deg, anti_phase_err={s.trot_anti_phase_err_deg:.3f} deg (FSR resets={s.fsr_resets})",
        f"IMU Mahony AHRS    : RMS_err={s.imu_est_err_rms_deg:.3f} deg, peak_push_roll={s.peak_disturbance_roll_deg:.2f} deg, post_recovery_RMS={s.post_recovery_roll_rms_deg:.3f} deg",
        f"UART CRC16 Link    : tx={s.tx_frames}, rx_valid={s.rx_valid_frames}, crc_rejected={s.crc_rejected_frames}, acked={s.ack_count}",
        f"Watchdog 100ms     : triggered={s.watchdog_triggered}, recovered={s.watchdog_recovered}",
        "",
        "Timepoints:",
        "   t(s) | roll(deg) | pitch(deg) | LF_phi(deg) | diag_err(deg) | esp_state   | ik_fk_err(mm)",
    ]
    for cp in s.checkpoints:
        lines.append(
            f"  {cp.t_s:5.2f} | {cp.roll_deg:+9.3f} | {cp.pitch_deg:+10.3f} | {cp.lf_phase_deg:11.2f} | "
            f"{cp.diag_sync_err_deg:13.3f} | {cp.esp_state:11s} | {cp.fk_ik_err_mm:.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-loop 12-DOF quadruped simulation.")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--freq", type=float, default=1.5)
    parser.add_argument("--vx", type=float, default=0.18)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--inject-disturbance", action="store_true")
    parser.add_argument("--inject-dropout", action="store_true")
    args = parser.parse_args()

    summary = run_simulation(
        duration_s=args.duration,
        step_freq_hz=args.freq,
        vx=args.vx,
        vy=args.vy,
        yaw_rate=args.yaw_rate,
        inject_disturbance=args.inject_disturbance,
        inject_dropout=args.inject_dropout,
    )
    print(format_report(summary))


if __name__ == "__main__":
    main()
