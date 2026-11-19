import math
from pathlib import Path
import struct
import subprocess
import unittest

from pi.kinematics import L0, LEG_SIDE, LEGS, NOMINAL_STAND_Z, leg_fk, leg_ik
from pi.uart_link import (
    CMD_ACK,
    CMD_TELEMETRY,
    MODE_TROT,
    STATE_ACTIVE,
    STATE_TIMEOUT_SAFE,
    FrameParser,
    pack_foot_targets,
    pack_mode,
    unpack_ack,
    unpack_telemetry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_EXE = REPO_ROOT / "build" / "test_esp32_firmware.exe"


def _ensure_cpp_built() -> Path:
    if CPP_EXE.exists():
        return CPP_EXE
    CPP_EXE.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Iesp32",
        "esp32/imu_mpu6050.cpp",
        "esp32/uart_parser.cpp",
        "esp32/joint_controller.cpp",
        "tests/cpp/test_esp32_firmware.cpp",
        "-o",
        str(CPP_EXE),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return CPP_EXE


def _bridge_record(uart_bytes: bytes, roll_deg: float, pitch_deg: float, n_ticks: int) -> bytes:
    return struct.pack("<HffB", len(uart_bytes), roll_deg, pitch_deg, n_ticks) + uart_bytes


class TestCrossLanguageBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = _ensure_cpp_built()

    def test_python_vs_cpp_kinematics_exact_agreement(self):
        test_vectors = []
        for leg in LEGS:
            side = LEG_SIDE[leg]
            for x in (-0.05, 0.0, 0.05):
                for dy in (-0.025, 0.0, 0.025):
                    for z in (-0.165, -0.140, -0.115):
                        test_vectors.append((x, side * L0 + dy, z, side))

        stdin_blob = b"".join(struct.pack("<4f", *v) for v in test_vectors)
        proc = subprocess.run(
            [str(self.exe), "--ik-batch"],
            input=stdin_blob,
            capture_output=True,
            check=True,
        )
        self.assertEqual(len(proc.stdout), len(test_vectors) * 24)

        for idx, (x, y, z, side) in enumerate(test_vectors):
            c_r, c_p, c_k, c_fx, c_fy, c_fz = struct.unpack_from("<6f", proc.stdout, idx * 24)
            py_ik = leg_ik(x, y, z, side)
            py_fk = leg_fk(py_ik.hip_roll, py_ik.hip_pitch, py_ik.knee, side)
            self.assertAlmostEqual(c_r, py_ik.hip_roll, delta=1e-5)
            self.assertAlmostEqual(c_p, py_ik.hip_pitch, delta=1e-5)
            self.assertAlmostEqual(c_k, py_ik.knee, delta=1e-5)
            self.assertAlmostEqual(c_fx, py_fk[0], delta=1e-5)
            self.assertAlmostEqual(c_fy, py_fk[1], delta=1e-5)
            self.assertAlmostEqual(c_fz, py_fk[2], delta=1e-5)

    def test_python_to_cpp_binary_uart_imu_and_watchdog_bridge(self):
        feet_nominal = {leg: (0.015, LEG_SIDE[leg] * L0, NOMINAL_STAND_Z) for leg in LEGS}

        records = bytearray()
        # 1. Send CMD_MODE(TROT) + corrupted frame junk, tilt body +12 deg roll / -6 deg pitch
        bad_mode = bytearray(pack_mode(MODE_TROT, seq=11))
        bad_mode[4] ^= 0xFF
        records += _bridge_record(
            pack_mode(MODE_TROT, seq=12) + bytes(bad_mode),
            roll_deg=12.0,
            pitch_deg=-6.0,
            n_ticks=2,
        )

        # 2. Stream 50 Hz CMD_FOOT_TARGETS for 2.0 seconds (100 batches x 20ms)
        foot_wire = pack_foot_targets(feet_nominal, seq=20)
        for _ in range(100):
            records += _bridge_record(foot_wire, roll_deg=12.0, pitch_deg=-6.0, n_ticks=2)

        # 3. Drop UART link for 160 ms (16 ticks) -> verify C++ watchdog triggers STATE_TIMEOUT_SAFE
        records += _bridge_record(b"", roll_deg=12.0, pitch_deg=-6.0, n_ticks=16)

        proc = subprocess.run(
            [str(self.exe), "--bridge"],
            input=bytes(records),
            capture_output=True,
            check=True,
        )

        parser = FrameParser()
        frames = parser.feed(proc.stdout)
        self.assertEqual(parser.dropped_frames, 0)

        acks = [unpack_ack(p) for cmd, seq, p in frames if cmd == CMD_ACK and seq == 12]
        self.assertEqual(acks, [(2, 0)])  # (CMD_MODE, status=0)

        telems = [unpack_telemetry(p) for cmd, _, p in frames if cmd == CMD_TELEMETRY]
        self.assertGreaterEqual(len(telems), 100)

        # Before dropout, C++ Mahony AHRS should track 12 deg roll / -6 deg pitch
        pre_dropout = telems[95]
        self.assertEqual(pre_dropout.controller_state, STATE_ACTIVE)
        self.assertGreaterEqual(pre_dropout.rx_dropped_frames, 1)
        self.assertAlmostEqual(math.degrees(pre_dropout.roll_rad), 12.0, delta=0.6)
        self.assertAlmostEqual(math.degrees(pre_dropout.pitch_rad), -6.0, delta=0.6)

        # After 160 ms dropout, C++ telemetry reports STATE_TIMEOUT_SAFE
        self.assertEqual(telems[-1].controller_state, STATE_TIMEOUT_SAFE)


if __name__ == "__main__":
    unittest.main()
