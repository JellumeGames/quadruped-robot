#include "uart_parser.h"

#include <cmath>
#include <cstring>

namespace quad {

namespace {

inline int16_t read_i16_le(const uint8_t *p) {
  uint16_t u = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
  return static_cast<int16_t>(u);
}

inline void write_i16_le(uint8_t *p, int16_t v) {
  uint16_t u = static_cast<uint16_t>(v);
  p[0] = static_cast<uint8_t>(u & 0xFF);
  p[1] = static_cast<uint8_t>((u >> 8) & 0xFF);
}

inline void write_u16_le(uint8_t *p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
}

inline int16_t clamp_i16(float val) {
  long iv = std::lround(val);
  if (iv < -32768L) return -32768;
  if (iv > 32767L) return 32767;
  return static_cast<int16_t>(iv);
}

}  // namespace

uint16_t crc16_ccitt(const uint8_t *data, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (uint8_t b = 0; b < 8; ++b) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                           : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

size_t pack_frame(uint8_t cmd, uint8_t seq, const uint8_t *payload,
                  uint8_t payload_len, uint8_t *out_buf, size_t max_out) {
  const uint8_t body_len = static_cast<uint8_t>(2 + payload_len);
  if (body_len > kMaxBodyLen) return 0;
  const size_t total = static_cast<size_t>(3 + body_len + 4);
  if (max_out < total) return 0;

  out_buf[0] = kHdr0;
  out_buf[1] = kHdr1;
  out_buf[2] = body_len;
  out_buf[3] = cmd;
  out_buf[4] = seq;
  if (payload_len > 0 && payload != nullptr) {
    std::memcpy(out_buf + 5, payload, payload_len);
  }
  const uint16_t crc = crc16_ccitt(out_buf + 2, static_cast<size_t>(1 + body_len));
  out_buf[3 + body_len] = static_cast<uint8_t>((crc >> 8) & 0xFF);
  out_buf[4 + body_len] = static_cast<uint8_t>(crc & 0xFF);
  out_buf[5 + body_len] = kTail0;
  out_buf[6 + body_len] = kTail1;
  return total;
}

size_t pack_ack(uint8_t seq, uint8_t acked_cmd, uint8_t status,
                uint8_t *out_buf, size_t max_out) {
  const uint8_t payload[2] = {acked_cmd, status};
  return pack_frame(kCmdAck, seq, payload, 2, out_buf, max_out);
}

size_t pack_telemetry(uint8_t seq, const TelemetryPacket &t,
                      uint8_t *out_buf, size_t max_out) {
  uint8_t payload[22]{};
  write_i16_le(payload + 0, clamp_i16(t.roll_rad * kAngleToCentideg));
  write_i16_le(payload + 2, clamp_i16(t.pitch_rad * kAngleToCentideg));
  write_i16_le(payload + 4, clamp_i16(t.yaw_rad * kAngleToCentideg));
  write_i16_le(payload + 6, clamp_i16(t.gx_rads * 1000.0f));
  write_i16_le(payload + 8, clamp_i16(t.gy_rads * 1000.0f));
  write_i16_le(payload + 10, clamp_i16(t.gz_rads * 1000.0f));
  for (int i = 0; i < 4; ++i) {
    float f = t.fsr[i];
    if (f < 0.0f) f = 0.0f;
    if (f > 1.0f) f = 1.0f;
    uint16_t raw = static_cast<uint16_t>(std::lround(f * 4095.0f));
    write_u16_le(payload + 12 + i * 2, raw);
  }
  payload[20] = t.controller_state;
  payload[21] = t.rx_dropped_frames;
  return pack_frame(kCmdTelemetry, seq, payload, sizeof(payload), out_buf, max_out);
}

bool decode_joints_payload(const uint8_t *payload, uint8_t payload_len, float out_angles[12]) {
  if (payload == nullptr || payload_len != 24) return false;
  for (int i = 0; i < 12; ++i) {
    out_angles[i] = static_cast<float>(read_i16_le(payload + i * 2)) / kAngleToCentideg;
  }
  return true;
}

bool decode_foot_targets_payload(const uint8_t *payload, uint8_t payload_len, float out_xyz[12]) {
  if (payload == nullptr || payload_len != 24) return false;
  for (int i = 0; i < 12; ++i) {
    out_xyz[i] = static_cast<float>(read_i16_le(payload + i * 2)) / kMeterToTenthMm;
  }
  return true;
}

void UartParser::reset() {
  state_ = WAIT_HDR0;
  len_ = 0;
  idx_ = 0;
}

bool UartParser::feed_byte(uint8_t b, ParsedFrame &out) {
  switch (state_) {
    case WAIT_HDR0:
      if (b == kHdr0) state_ = WAIT_HDR1;
      return false;

    case WAIT_HDR1:
      if (b == kHdr1) {
        state_ = WAIT_LEN;
      } else if (b == kHdr0) {
        state_ = WAIT_HDR1;  // Keep syncing on repeated 0xAA bytes.
      } else {
        state_ = WAIT_HDR0;
      }
      return false;

    case WAIT_LEN:
      if (b < 2 || b > kMaxBodyLen) {
        ++dropped_frames_;
        state_ = (b == kHdr0) ? WAIT_HDR1 : WAIT_HDR0;
        return false;
      }
      len_ = b;
      idx_ = 0;
      state_ = WAIT_BODY;
      return false;

    case WAIT_BODY: {
      buf_[idx_++] = b;
      const uint8_t expected = static_cast<uint8_t>(len_ + 4);
      if (idx_ < expected) return false;

      state_ = WAIT_HDR0;
      if (buf_[len_ + 2] != kTail0 || buf_[len_ + 3] != kTail1) {
        ++dropped_frames_;
        return false;
      }

      uint8_t crc_buf[kMaxBodyLen + 1]{};
      crc_buf[0] = len_;
      std::memcpy(crc_buf + 1, buf_, len_);
      const uint16_t got_crc = crc16_ccitt(crc_buf, static_cast<size_t>(len_ + 1));
      const uint16_t want_crc =
          static_cast<uint16_t>((static_cast<uint16_t>(buf_[len_]) << 8) | buf_[len_ + 1]);
      if (got_crc != want_crc) {
        ++dropped_frames_;
        return false;
      }

      out.cmd = buf_[0];
      out.seq = buf_[1];
      out.payload_len = static_cast<uint8_t>(len_ - 2);
      if (out.payload_len > 0) {
        std::memcpy(out.payload, buf_ + 2, out.payload_len);
      }
      ++valid_frames_;
      return true;
    }
  }
  return false;
}

}  // namespace quad
