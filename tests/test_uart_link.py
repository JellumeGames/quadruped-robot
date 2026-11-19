import unittest

from pi.kinematics import L0, LEG_SIDE, LEGS, NOMINAL_STAND_Z
from pi.uart_link import (
    CMD_ACK,
    CMD_FOOT_TARGETS,
    CMD_JOINTS,
    CMD_MODE,
    CMD_TELEMETRY,
    MODE_TROT,
    STATE_ACTIVE,
    FrameParser,
    ReliableCommander,
    Telemetry,
    pack_ack,
    pack_foot_targets,
    pack_joints,
    pack_mode,
    pack_telemetry,
    unpack_ack,
    unpack_foot_targets,
    unpack_joints,
    unpack_mode,
    unpack_telemetry,
)


class TestUartLink(unittest.TestCase):
    def test_all_message_types_roundtrip(self):
        parser = FrameParser()
        angles = [0.05 * (i - 6) for i in range(12)]
        feet = {leg: (0.02, LEG_SIDE[leg] * L0, NOMINAL_STAND_Z + 0.01) for leg in LEGS}
        telem = Telemetry(
            roll_rad=0.11,
            pitch_rad=-0.08,
            yaw_rad=0.45,
            gx_rads=0.12,
            gy_rads=-0.34,
            gz_rads=0.56,
            fsr=(0.1, 0.25, 0.75, 0.99),
            controller_state=STATE_ACTIVE,
            rx_dropped_frames=3,
        )
        wire = (
            pack_joints(angles, seq=7)
            + pack_foot_targets(feet, seq=8)
            + pack_mode(MODE_TROT, seq=9)
            + pack_ack(CMD_MODE, 0, seq=9)
            + pack_telemetry(telem, seq=10)
        )
        frames = parser.feed(wire)
        self.assertEqual(len(frames), 5)
        self.assertEqual(parser.dropped_frames, 0)

        self.assertEqual(frames[0][0], CMD_JOINTS)
        for a, b in zip(unpack_joints(frames[0][2]), angles):
            self.assertAlmostEqual(a, b, places=3)

        self.assertEqual(frames[1][0], CMD_FOOT_TARGETS)
        feet_out = unpack_foot_targets(frames[1][2])
        for leg in LEGS:
            for a, b in zip(feet_out[leg], feet[leg]):
                self.assertAlmostEqual(a, b, places=3)

        self.assertEqual(frames[2][0], CMD_MODE)
        self.assertEqual(unpack_mode(frames[2][2]), MODE_TROT)

        self.assertEqual(frames[3][0], CMD_ACK)
        self.assertEqual(unpack_ack(frames[3][2]), (CMD_MODE, 0))

        self.assertEqual(frames[4][0], CMD_TELEMETRY)
        telem_out = unpack_telemetry(frames[4][2])
        self.assertAlmostEqual(telem_out.roll_rad, telem.roll_rad, places=3)
        self.assertAlmostEqual(telem_out.pitch_rad, telem.pitch_rad, places=3)
        self.assertAlmostEqual(telem_out.gx_rads, telem.gx_rads, places=3)
        self.assertEqual(telem_out.controller_state, STATE_ACTIVE)
        self.assertEqual(telem_out.rx_dropped_frames, 3)

    def test_crc_corruption_false_header_resync_and_fragmentation(self):
        parser = FrameParser()
        good = pack_mode(MODE_TROT, seq=42)

        # Corrupted CRC frame
        bad = bytearray(good)
        bad[4] ^= 0x01
        self.assertEqual(parser.feed(bytes(bad)), [])
        self.assertEqual(parser.dropped_frames, 1)

        # False 0xAA 0x55 header immediately overlapping a real frame!
        # Byte stream: [0xAA, 0x55, 0x03] + good
        # Notice false frame claims length=3 (total=10), which overlaps 7 bytes of `good`;
        # a naive parser dropping `total` bytes loses `good`, whereas sliding by 2 recovers it.
        parser2 = FrameParser()
        tricky = b"\xAA\x55\x03" + good
        parsed = []
        for byte in tricky:
            parsed.extend(parser2.feed(bytes([byte])))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0][0], CMD_MODE)
        self.assertEqual(parsed[0][1], 42)
        self.assertGreaterEqual(parser2.dropped_frames, 1)

        # Longer false header also recovers once subsequent stream bytes complete its window
        parser2b = FrameParser()
        good2 = pack_mode(MODE_TROT, seq=43)
        parsed2b = parser2b.feed(b"\xAA\x55\x10" + good + good2)
        self.assertEqual([(c, s) for c, s, _ in parsed2b], [(CMD_MODE, 42), (CMD_MODE, 43)])

        # Oversized length rejected cleanly
        parser3 = FrameParser()
        oversized = b"\xAA\x55\xFF\x01\x02" + good
        self.assertEqual(len(parser3.feed(oversized)), 1)
        self.assertEqual(parser3.dropped_frames, 1)

    def test_reliable_commander_retry_and_ack(self):
        cmdr = ReliableCommander(retry_interval_s=0.05, max_retries=3)
        seq, _ = cmdr.send_mode(MODE_TROT, now_s=0.0)
        self.assertEqual(cmdr.pending_count, 1)

        self.assertEqual(cmdr.poll_retries(0.02), [])
        retries = cmdr.poll_retries(0.06)
        self.assertEqual(len(retries), 1)

        ack_info = cmdr.on_frame(CMD_ACK, seq, bytes([CMD_MODE, 0]))
        self.assertEqual(ack_info, (seq, 0))
        self.assertEqual(cmdr.pending_count, 0)
        self.assertEqual(cmdr.acked_count, 1)


if __name__ == "__main__":
    unittest.main()
