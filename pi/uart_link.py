"""Binary UART link between the Raspberry Pi planner and ESP32 controller.

Frame format (115200 bps):
    0xAA 0x55 | len | cmd | seq | payload... | CRC16-CCITT (BE) | 0x0D 0x0A

Where `len = 2 + len(payload)` (bytes of cmd + seq + payload), and CRC16-CCITT
(poly 0x1021, init 0xFFFF) covers `[len, cmd, seq, payload...]`.

Must stay wire-compatible with `esp32/uart_parser.h` and `esp32/uart_parser.cpp`.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Dict, List, Optional, Sequence, Tuple

from pi.kinematics import LEGS, Vec3

HEADER = b"\xAA\x55"
TAIL = b"\x0D\x0A"
MAX_BODY_LEN = 60  # max(cmd + seq + payload)

CMD_JOINTS = 0x01
CMD_MODE = 0x02
CMD_ACK = 0x03
CMD_TELEMETRY = 0x04
CMD_FOOT_TARGETS = 0x05

MODE_IDLE = 0
MODE_STAND = 1
MODE_TROT = 2
MODE_ESTOP = 3

STATE_IDLE = 0
STATE_STAND = 1
STATE_ACTIVE = 2
STATE_TIMEOUT_SAFE = 3
STATE_ESTOP = 4

ANGLE_TO_CENTIDEG = 18000.0 / 3.141592653589793  # 1 rad -> 5729.57795 cdeg
METER_TO_TENTH_MM = 10000.0                       # 1 m -> 10000 * 0.1 mm


@dataclass(frozen=True)
class Frame:
    cmd: int
    seq: int
    payload: bytes


@dataclass(frozen=True)
class Telemetry:
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    gx_rads: float
    gy_rads: float
    gz_rads: float
    fsr: Tuple[float, float, float, float]
    controller_state: int
    rx_dropped_frames: int


def _clamp_i16(val: float) -> int:
    iv = int(round(val))
    return -32768 if iv < -32768 else (32767 if iv > 32767 else iv)


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def pack(cmd: int, payload: bytes = b"", seq: int = 0) -> bytes:
    body = struct.pack("BB", cmd & 0xFF, seq & 0xFF) + payload
    if len(body) > MAX_BODY_LEN:
        raise ValueError(f"Payload too large: body len {len(body)} > {MAX_BODY_LEN}")
    frame = HEADER + struct.pack("B", len(body)) + body
    crc = crc16_ccitt(frame[2:])
    return frame + struct.pack(">H", crc) + TAIL


def pack_joints(angles12: Sequence[float], seq: int = 0) -> bytes:
    """Pack 12 joint angles (rad) as 12 x int16 centi-degrees (0.01 deg)."""
    if len(angles12) != 12:
        raise ValueError(f"Expected 12 joint angles, got {len(angles12)}")
    scaled = [_clamp_i16(a * ANGLE_TO_CENTIDEG) for a in angles12]
    return pack(CMD_JOINTS, struct.pack("<12h", *scaled), seq=seq)


def unpack_joints(payload: bytes) -> List[float]:
    if len(payload) != 24:
        raise ValueError(f"CMD_JOINTS payload must be 24 bytes, got {len(payload)}")
    vals = struct.unpack("<12h", payload)
    return [v / ANGLE_TO_CENTIDEG for v in vals]


def pack_foot_targets(feet_hip: Dict[str, Vec3], seq: int = 0) -> bytes:
    """Pack 4 hip-frame foot targets (m) as 12 x int16 in 0.1 mm units."""
    coords: List[int] = []
    for leg in LEGS:
        x, y, z = feet_hip[leg]
        coords.extend((
            _clamp_i16(x * METER_TO_TENTH_MM),
            _clamp_i16(y * METER_TO_TENTH_MM),
            _clamp_i16(z * METER_TO_TENTH_MM),
        ))
    return pack(CMD_FOOT_TARGETS, struct.pack("<12h", *coords), seq=seq)


def unpack_foot_targets(payload: bytes) -> Dict[str, Vec3]:
    if len(payload) != 24:
        raise ValueError(f"CMD_FOOT_TARGETS payload must be 24 bytes, got {len(payload)}")
    vals = struct.unpack("<12h", payload)
    out: Dict[str, Vec3] = {}
    for idx, leg in enumerate(LEGS):
        base = idx * 3
        out[leg] = (
            vals[base] / METER_TO_TENTH_MM,
            vals[base + 1] / METER_TO_TENTH_MM,
            vals[base + 2] / METER_TO_TENTH_MM,
        )
    return out


def pack_mode(mode: int, seq: int = 1) -> bytes:
    return pack(CMD_MODE, struct.pack("B", mode & 0xFF), seq=seq)


def unpack_mode(payload: bytes) -> int:
    if len(payload) != 1:
        raise ValueError(f"CMD_MODE payload must be 1 byte, got {len(payload)}")
    return payload[0]


def pack_ack(acked_cmd: int, status: int = 0, seq: int = 0) -> bytes:
    return pack(CMD_ACK, struct.pack("BB", acked_cmd & 0xFF, status & 0xFF), seq=seq)


def unpack_ack(payload: bytes) -> Tuple[int, int]:
    if len(payload) != 2:
        raise ValueError(f"CMD_ACK payload must be 2 bytes, got {len(payload)}")
    return payload[0], payload[1]


def pack_telemetry(telem: Telemetry, seq: int = 0) -> bytes:
    """Pack 22-byte ESP32 telemetry frame (<6h4H2B)."""
    r_cdeg = _clamp_i16(telem.roll_rad * ANGLE_TO_CENTIDEG)
    p_cdeg = _clamp_i16(telem.pitch_rad * ANGLE_TO_CENTIDEG)
    y_cdeg = _clamp_i16(telem.yaw_rad * ANGLE_TO_CENTIDEG)
    gx_mr = _clamp_i16(telem.gx_rads * 1000.0)
    gy_mr = _clamp_i16(telem.gy_rads * 1000.0)
    gz_mr = _clamp_i16(telem.gz_rads * 1000.0)
    fsr_u16 = [max(0, min(4095, int(round(f * 4095.0)))) for f in telem.fsr]
    payload = struct.pack(
        "<6h4H2B",
        r_cdeg,
        p_cdeg,
        y_cdeg,
        gx_mr,
        gy_mr,
        gz_mr,
        fsr_u16[0],
        fsr_u16[1],
        fsr_u16[2],
        fsr_u16[3],
        telem.controller_state & 0xFF,
        telem.rx_dropped_frames & 0xFF,
    )
    return pack(CMD_TELEMETRY, payload, seq=seq)


def unpack_telemetry(payload: bytes) -> Telemetry:
    if len(payload) != 22:
        raise ValueError(f"CMD_TELEMETRY payload must be 22 bytes, got {len(payload)}")
    r, p, y, gx, gy, gz, f0, f1, f2, f3, state, dropped = struct.unpack("<6h4H2B", payload)
    return Telemetry(
        roll_rad=r / ANGLE_TO_CENTIDEG,
        pitch_rad=p / ANGLE_TO_CENTIDEG,
        yaw_rad=y / ANGLE_TO_CENTIDEG,
        gx_rads=gx / 1000.0,
        gy_rads=gy / 1000.0,
        gz_rads=gz / 1000.0,
        fsr=(f0 / 4095.0, f1 / 4095.0, f2 / 4095.0, f3 / 4095.0),
        controller_state=state,
        rx_dropped_frames=dropped,
    )


class FrameParser:
    """Streaming frame parser with byte-level resynchronization on bad frames."""

    def __init__(self):
        self.buf = bytearray()
        self.valid_frames: int = 0
        self.dropped_frames: int = 0

    def feed(self, data: bytes) -> List[Tuple[int, int, bytes]]:
        self.buf += data
        frames: List[Tuple[int, int, bytes]] = []
        while True:
            i = self.buf.find(HEADER)
            if i < 0:
                # Keep trailing 0xAA in case header spans read chunks.
                if self.buf and self.buf[-1] == HEADER[0]:
                    del self.buf[:-1]
                else:
                    self.buf.clear()
                return frames
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < 5:
                return frames

            length = self.buf[2]
            if length < 2 or length > MAX_BODY_LEN:
                # Invalid length byte: advance past false header and resync.
                self.dropped_frames += 1
                del self.buf[:2]
                continue

            total = 3 + length + 2 + 2
            if len(self.buf) < total:
                return frames

            candidate = bytes(self.buf[:total])
            tail_ok = candidate[-2:] == TAIL
            want_crc = struct.unpack(">H", candidate[3 + length : 5 + length])[0]
            got_crc = crc16_ccitt(candidate[2 : 3 + length])

            if not tail_ok or got_crc != want_crc:
                # Resync 2 bytes past the false header so an overlapping valid
                # frame starting inside `candidate` is never discarded.
                self.dropped_frames += 1
                del self.buf[:2]
                continue

            del self.buf[:total]
            self.valid_frames += 1
            frames.append((candidate[3], candidate[4], candidate[5 : 3 + length]))
        return frames


@dataclass
class _PendingCommand:
    cmd: int
    seq: int
    wire_bytes: bytes
    last_sent_s: float
    retries: int = 0


class ReliableCommander:
    """Tracks sequence-numbered critical commands (e.g. CMD_MODE) until ACKed."""

    def __init__(self, retry_interval_s: float = 0.05, max_retries: int = 5):
        self.retry_interval_s = retry_interval_s
        self.max_retries = max_retries
        self._next_seq = 1
        self._pending: Dict[int, _PendingCommand] = {}
        self.acked_count = 0
        self.timed_out_count = 0

    def _alloc_seq(self) -> int:
        seq = self._next_seq
        self._next_seq = (self._next_seq % 255) + 1
        return seq

    def send_mode(self, mode: int, now_s: float) -> Tuple[int, bytes]:
        seq = self._alloc_seq()
        raw = pack_mode(mode, seq=seq)
        self._pending[seq] = _PendingCommand(cmd=CMD_MODE, seq=seq, wire_bytes=raw, last_sent_s=now_s)
        return seq, raw

    def on_frame(self, cmd: int, seq: int, payload: bytes) -> Optional[Tuple[int, int]]:
        if cmd != CMD_ACK or len(payload) < 2:
            return None
        acked_cmd, status = unpack_ack(payload[:2])
        pending = self._pending.get(seq)
        if pending is not None and pending.cmd == acked_cmd:
            del self._pending[seq]
            self.acked_count += 1
            return (seq, status)
        return None

    def poll_retries(self, now_s: float) -> List[bytes]:
        retransmits: List[bytes] = []
        expired: List[int] = []
        for seq, item in self._pending.items():
            if now_s - item.last_sent_s >= self.retry_interval_s:
                if item.retries >= self.max_retries:
                    expired.append(seq)
                    self.timed_out_count += 1
                else:
                    item.retries += 1
                    item.last_sent_s = now_s
                    retransmits.append(item.wire_bytes)
        for seq in expired:
            del self._pending[seq]
        return retransmits

    @property
    def pending_count(self) -> int:
        return len(self._pending)
