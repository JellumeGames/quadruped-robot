// UART binary frame parser and serializer for the ESP32 side.
// Frame: 0xAA 0x55 | len | cmd | seq | payload... | CRC16-CCITT (BE) | 0x0D 0x0A
// Mirrors `pi/uart_link.py` byte-for-byte.

#pragma once

#include <cstddef>
#include <cstdint>

namespace quad {

constexpr uint8_t kHdr0 = 0xAA;
constexpr uint8_t kHdr1 = 0x55;
constexpr uint8_t kTail0 = 0x0D;
constexpr uint8_t kTail1 = 0x0A;
constexpr uint8_t kMaxBodyLen = 60;

constexpr uint8_t kCmdJoints = 0x01;
constexpr uint8_t kCmdMode = 0x02;
constexpr uint8_t kCmdAck = 0x03;
constexpr uint8_t kCmdTelemetry = 0x04;
constexpr uint8_t kCmdFootTargets = 0x05;

constexpr uint8_t kModeIdle = 0;
constexpr uint8_t kModeStand = 1;
constexpr uint8_t kModeTrot = 2;
constexpr uint8_t kModeEstop = 3;

constexpr uint8_t kStateIdle = 0;
constexpr uint8_t kStateStand = 1;
constexpr uint8_t kStateActive = 2;
constexpr uint8_t kStateTimeoutSafe = 3;
constexpr uint8_t kStateEstop = 4;

constexpr float kAngleToCentideg = 18000.0f / 3.14159265358979323846f;
constexpr float kMeterToTenthMm = 10000.0f;

struct ParsedFrame {
  uint8_t cmd = 0;
  uint8_t seq = 0;
  uint8_t payload_len = 0;
  uint8_t payload[kMaxBodyLen]{};
};

struct TelemetryPacket {
  float roll_rad = 0.0f;
  float pitch_rad = 0.0f;
  float yaw_rad = 0.0f;
  float gx_rads = 0.0f;
  float gy_rads = 0.0f;
  float gz_rads = 0.0f;
  float fsr[4]{0.0f, 0.0f, 0.0f, 0.0f};
  uint8_t controller_state = kStateIdle;
  uint8_t rx_dropped_frames = 0;
};

uint16_t crc16_ccitt(const uint8_t *data, size_t n);

size_t pack_frame(uint8_t cmd, uint8_t seq, const uint8_t *payload,
                  uint8_t payload_len, uint8_t *out_buf, size_t max_out);

size_t pack_ack(uint8_t seq, uint8_t acked_cmd, uint8_t status,
                uint8_t *out_buf, size_t max_out);

size_t pack_telemetry(uint8_t seq, const TelemetryPacket &t,
                      uint8_t *out_buf, size_t max_out);

bool decode_joints_payload(const uint8_t *payload, uint8_t payload_len, float out_angles[12]);
bool decode_foot_targets_payload(const uint8_t *payload, uint8_t payload_len, float out_xyz[12]);

class UartParser {
 public:
  UartParser() = default;

  void reset();
  bool feed_byte(uint8_t b, ParsedFrame &out);

  uint32_t valid_frames() const { return valid_frames_; }
  uint32_t dropped_frames() const { return dropped_frames_; }

 private:
  enum ParseState { WAIT_HDR0, WAIT_HDR1, WAIT_LEN, WAIT_BODY };

  ParseState state_ = WAIT_HDR0;
  uint8_t buf_[kMaxBodyLen + 4]{};  // cmd + seq + payload + CRC(2) + tail(2)
  uint8_t len_ = 0;
  uint8_t idx_ = 0;
  uint32_t valid_frames_ = 0;
  uint32_t dropped_frames_ = 0;
};

}  // namespace quad
