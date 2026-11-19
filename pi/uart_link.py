"""UART link between the Pi (planner) and the ESP32 (joint controller).

Frame: 0xAA 0x55 | len | cmd | payload | CRC16-CCITT | 0x0D 0x0A
- Joint angles stream at 50 Hz: lost frames are NOT retransmitted, the
  next frame arrives 10 ms later.
- Important commands carry a sequence number and require an ACK.
- Watchdog: no frame for 100 ms -> the ESP32 holds a safe stance.
"""

import struct

HEADER = b"\xAA\x55"
TAIL = b"\x0D\x0A"
CMD_JOINTS = 0x01
CMD_MODE = 0x02
CMD_ACK = 0x03


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def pack(cmd: int, payload: bytes, seq: int = 0) -> bytes:
    body = struct.pack("BB", cmd, seq) + payload
    frame = HEADER + struct.pack("B", len(body)) + body
    return frame + struct.pack(">H", crc16_ccitt(frame[2:])) + TAIL


def pack_joints(angles12: list[float]) -> bytes:
    """12 joint angles (radians) as 12 x int16 centi-degrees."""
    payload = struct.pack("<12h", *[int(a * 5729.57795) for a in angles12])
    return pack(CMD_JOINTS, payload)


class FrameParser:
    """Byte-wise state machine: sync on header, check length + CRC,
    drop bad frames silently."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf += data
        frames = []
        while True:
            i = self.buf.find(HEADER)
            if i < 0:
                self.buf.clear()
                return frames
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < 5:
                return frames
            length = self.buf[2]
            total = 3 + length + 2 + 2
            if len(self.buf) < total:
                return frames
            frame = bytes(self.buf[:total])
            del self.buf[:total]
            if frame[-2:] != TAIL:
                continue
            if crc16_ccitt(frame[2:3 + length]) != struct.unpack(">H", frame[3 + length:5 + length])[0]:
                continue
            frames.append((frame[3], frame[4], frame[5:3 + length]))
        return frames
