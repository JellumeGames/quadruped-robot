#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "hal.h"
#include "imu_mpu6050.h"
#include "joint_controller.h"
#include "kinematics.h"
#include "uart_parser.h"

namespace {

using namespace quad;

int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg); \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

void set_mock_mpu6050_registers(float ax_g, float ay_g, float az_g,
                                float gx_rads, float gy_rads, float gz_rads,
                                float temp_c = 25.0f) {
  auto encode_i16 = [](uint8_t reg, float val) {
    long iv = std::lround(val);
    if (iv < -32768L) iv = -32768L;
    if (iv > 32767L) iv = 32767L;
    uint16_t u = static_cast<uint16_t>(static_cast<int16_t>(iv));
    mock_hal().i2c_regs[reg] = static_cast<uint8_t>((u >> 8) & 0xFF);
    mock_hal().i2c_regs[reg + 1] = static_cast<uint8_t>(u & 0xFF);
  };
  constexpr float kRadToDeg = 180.0f / kPi;
  encode_i16(kRegAccelXoutH + 0, ax_g * kAccelScale4g);
  encode_i16(kRegAccelXoutH + 2, ay_g * kAccelScale4g);
  encode_i16(kRegAccelXoutH + 4, az_g * kAccelScale4g);
  encode_i16(kRegAccelXoutH + 6, (temp_c - 36.53f) * 340.0f);
  encode_i16(kRegAccelXoutH + 8, gx_rads * kRadToDeg * kGyroScale500dps);
  encode_i16(kRegAccelXoutH + 10, gy_rads * kRadToDeg * kGyroScale500dps);
  encode_i16(kRegAccelXoutH + 12, gz_rads * kRadToDeg * kGyroScale500dps);
}

void set_mock_mpu6050_tilt_deg(float roll_deg, float pitch_deg,
                               float gx_bias = 0.0f, float gy_bias = 0.0f, float gz_bias = 0.0f,
                               float az_spike_g = 0.0f) {
  const float r = roll_deg * (kPi / 180.0f);
  const float p = pitch_deg * (kPi / 180.0f);
  const float ax = -std::sin(p);
  const float ay = std::cos(p) * std::sin(r);
  const float az = std::cos(p) * std::cos(r) + az_spike_g;
  set_mock_mpu6050_registers(ax, ay, az, gx_bias, gy_bias, gz_bias);
}

void test_cpp_kinematics() {
  float max_err = 0.0f;
  for (uint8_t leg = 0; leg < 4; ++leg) {
    const float side = kLegSide[leg];
    for (float x = -0.06f; x <= 0.061f; x += 0.03f) {
      for (float dy = -0.03f; dy <= 0.031f; dy += 0.03f) {
        const float y = side * kL0 + dy;
        for (float z = -0.175f; z <= -0.105f; z += 0.035f) {
          const LegAngles ik = leg_ik(x, y, z, side);
          CHECK(!ik.clamped, "Workspace grid point should not clamp");
          const Vec3 fk = leg_fk(ik.hip_roll, ik.hip_pitch, ik.knee, side);
          const float err = std::sqrt((fk.x - x) * (fk.x - x) +
                                      (fk.y - y) * (fk.y - y) +
                                      (fk.z - z) * (fk.z - z));
          if (err > max_err) max_err = err;
        }
      }
    }
  }
  CHECK(max_err < 1e-5f, "C++ 3-DOF IK/FK round-trip error < 1e-5 m");

  const LegAngles oob = leg_ik(0.5f, 0.0f, 0.0f, 1.0f, -1.0f, true);
  CHECK(oob.clamped, "Out-of-workspace target should report clamped");
  CHECK(std::isfinite(oob.hip_roll) && std::isfinite(oob.hip_pitch) && std::isfinite(oob.knee),
        "Clamped IK angles must remain finite");
  std::printf("[PASS] test_cpp_kinematics (max_err=%.2e m)\n", max_err);
}

void test_cpp_mpu6050_and_mahony() {
  mock_hal().reset();
  set_mock_mpu6050_tilt_deg(0.0f, 0.0f, 0.02f, -0.015f, 0.01f);

  ImuMpu6050 imu;
  CHECK(imu.init(), "MPU6050 init succeeds with WHO_AM_I=0x68");
  CHECK(mock_hal().i2c_regs[kRegPwrMgmt1] == 0x01, "PWR_MGMT_1 woken");
  CHECK(mock_hal().i2c_regs[kRegGyroConfig] == 0x08, "GYRO_CONFIG +/-500dps");
  CHECK(mock_hal().i2c_regs[kRegAccelConfig] == 0x08, "ACCEL_CONFIG +/-4g");

  CHECK(imu.calibrate_gyro(64), "Gyro calibration succeeds on stationary data");
  float bx = 0.0f, by = 0.0f, bz = 0.0f;
  imu.gyro_bias(bx, by, bz);
  CHECK(std::fabs(bx - 0.02f) < 0.002f, "Gyro X bias calibrated");
  CHECK(std::fabs(by + 0.015f) < 0.002f, "Gyro Y bias calibrated");

  set_mock_mpu6050_tilt_deg(15.0f, -10.0f, 0.02f, -0.015f, 0.01f);
  for (int i = 0; i < 450; ++i) {
    CHECK(imu.update(0.01f), "IMU update reads registers");
  }
  const EulerAngles att = imu.attitude();
  const float roll_deg = att.roll * (180.0f / kPi);
  const float pitch_deg = att.pitch * (180.0f / kPi);
  CHECK(std::fabs(roll_deg - 15.0f) < 0.25f, "Mahony converges to 15 deg roll");
  CHECK(std::fabs(pitch_deg + 10.0f) < 0.25f, "Mahony converges to -10 deg pitch");

  // Inject 2.5g impact spike and confirm shock gate suppresses distortion
  set_mock_mpu6050_tilt_deg(15.0f, -10.0f, 0.02f, -0.015f, 0.01f, 1.6f);
  for (int i = 0; i < 10; ++i) {
    imu.update(0.01f);
  }
  CHECK(imu.accel_gated_count() >= 10, "Mahony shock gate activates on 2.5g spike");
  const float roll_after = imu.attitude().roll * (180.0f / kPi);
  CHECK(std::fabs(roll_after - 15.0f) < 0.35f, "Attitude holds steady through impact spike");
  std::printf("[PASS] test_cpp_mpu6050_and_mahony (roll=%.2f deg, pitch=%.2f deg, gated=%u)\n",
              roll_deg, pitch_deg, imu.accel_gated_count());
}

void test_cpp_uart_parser() {
  UartParser parser;
  uint8_t frame_buf[64]{};
  const uint8_t mode_val = kModeTrot;
  const size_t n = pack_frame(kCmdMode, 17, &mode_val, 1, frame_buf, sizeof(frame_buf));
  CHECK(n == 10, "CMD_MODE wire length is 10 bytes");

  ParsedFrame out{};
  // Feed oversized bogus frame (len=255) + repeated 0xAA + valid frame
  const uint8_t noise_prefix[] = {0xAA, 0x55, 0xFF, 0xAA, 0xAA, 0x55};
  for (uint8_t b : noise_prefix) {
    CHECK(!parser.feed_byte(b, out), "Noise must not yield a frame");
  }
  // Feed corrupted valid frame (flip byte 4)
  uint8_t bad_buf[64]{};
  std::memcpy(bad_buf, frame_buf, n);
  bad_buf[4] ^= 0x55;
  for (size_t i = 0; i < n; ++i) {
    CHECK(!parser.feed_byte(bad_buf[i], out), "Corrupted CRC frame must be rejected");
  }
  CHECK(parser.dropped_frames() >= 2, "Dropped frame counter increments");

  // Now feed valid frame after repeated 0xAA
  CHECK(!parser.feed_byte(0xAA, out), "Extra 0xAA resync");
  bool got = false;
  for (size_t i = 0; i < n; ++i) {
    if (parser.feed_byte(frame_buf[i], out)) {
      got = true;
    }
  }
  CHECK(got, "Valid frame parsed after resync");
  CHECK(out.cmd == kCmdMode && out.seq == 17 && out.payload[0] == kModeTrot,
        "Decoded cmd/seq/payload match");
  std::printf("[PASS] test_cpp_uart_parser (valid=%u, dropped=%u)\n",
              parser.valid_frames(), parser.dropped_frames());
}

void test_cpp_fsr_debouncer() {
  FsrDebouncer db(0.25f, 0.15f, 20);
  // Short 10 ms glitch >= 0.25 must be ignored
  CHECK(!db.update(0.35f, 0), "Glitch start");
  CHECK(!db.update(0.35f, 10), "Glitch at 10ms still unconfirmed");
  CHECK(!db.update(0.05f, 15), "Glitch drops before 20ms");

  // Sustained press >= 20 ms latches contact
  CHECK(!db.update(0.40f, 20), "Press start at 20ms");
  CHECK(!db.update(0.40f, 35), "Still <20ms");
  CHECK(db.update(0.40f, 40), "Contact latches at 20ms");

  // Hysteresis holds contact at 0.20 (between 0.15 release and 0.25 press)
  CHECK(db.update(0.20f, 60), "Hysteresis keeps contact above 0.15");
  CHECK(!db.update(0.05f, 80) == false, "Release starts timer at 80ms");
  CHECK(!db.update(0.05f, 100), "Contact releases after 20ms below 0.15");
  std::printf("[PASS] test_cpp_fsr_debouncer\n");
}

void test_cpp_joint_controller_loop() {
  mock_hal().reset();
  set_mock_mpu6050_tilt_deg(0.0f, 0.0f);

  JointController ctrl;
  ctrl.init(false);

  // Safe stance must NOT be zeroed straight leg (knee must be bent)
  CHECK(ctrl.target_angle(2) < -1.0f, "Safe stance knee is bent (~-1.40 rad)");
  CHECK(mock_hal().ledc_freq[0] == 50 && mock_hal().ledc_res[0] == 16,
        "LEDC configured for 50 Hz 16-bit");
  const uint32_t neutral_duty = JointController::us_to_ledc_duty(1500.0f);
  CHECK(neutral_duty == 4915U, "1500 us maps to 16-bit duty 4915");

  // Send CMD_MODE(TROT)
  uint8_t mode_wire[16]{};
  const uint8_t trot_mode = kModeTrot;
  const size_t mn = pack_frame(kCmdMode, 5, &trot_mode, 1, mode_wire, sizeof(mode_wire));
  mock_hal().push_uart_rx(mode_wire, mn);
  ctrl.step_100hz(10);

  // Verify ACK was emitted
  UartParser tx_parser;
  ParsedFrame ack_f{};
  bool saw_ack = false;
  for (uint8_t b : mock_hal().uart_tx) {
    if (tx_parser.feed_byte(b, ack_f) && ack_f.cmd == kCmdAck) {
      saw_ack = (ack_f.seq == 5 && ack_f.payload[0] == kCmdMode && ack_f.payload[1] == 0);
    }
  }
  CHECK(saw_ack, "JointController sends matching CMD_ACK for CMD_MODE");

  // Tilt robot roll=+10 deg and send nominal hip-frame foot targets via CMD_FOOT_TARGETS
  set_mock_mpu6050_tilt_deg(10.0f, 0.0f);
  uint8_t foot_payload[24]{};
  for (uint8_t leg = 0; leg < 4; ++leg) {
    const int16_t x_tmm = 0;
    const int16_t y_tmm = static_cast<int16_t>(std::lround(kLegSide[leg] * kL0 * kMeterToTenthMm));
    const int16_t z_tmm = static_cast<int16_t>(std::lround(kNominalStandZ * kMeterToTenthMm));
    auto w_i16 = [](uint8_t *p, int16_t v) {
      p[0] = static_cast<uint8_t>(static_cast<uint16_t>(v) & 0xFF);
      p[1] = static_cast<uint8_t>((static_cast<uint16_t>(v) >> 8) & 0xFF);
    };
    w_i16(foot_payload + leg * 6 + 0, x_tmm);
    w_i16(foot_payload + leg * 6 + 2, y_tmm);
    w_i16(foot_payload + leg * 6 + 4, z_tmm);
  }
  uint8_t foot_wire[48]{};
  const size_t fn = pack_frame(kCmdFootTargets, 6, foot_payload, 24, foot_wire, sizeof(foot_wire));
  mock_hal().uart_tx.clear();
  for (uint32_t t = 20; t <= 1000; t += 10) {
    if (t % 20 == 0) mock_hal().push_uart_rx(foot_wire, fn);
    ctrl.step_100hz(t);
  }
  // With positive roll (+10 deg), right legs (RF=1, RR=3) extend further (less bent knee)
  // than left legs (LF=0, LR=2) to restore body upright.
  CHECK(ctrl.target_angle(1 * 3 + 2) > ctrl.target_angle(0 * 3 + 2),
        "Onboard IMU stabilization extends right leg vs left leg on +roll");

  // Verify 50 Hz telemetry emitted
  int telem_count = 0;
  ParsedFrame tf{};
  for (uint8_t b : mock_hal().uart_tx) {
    if (tx_parser.feed_byte(b, tf) && tf.cmd == kCmdTelemetry) ++telem_count;
  }
  CHECK(telem_count >= 45, "Telemetry emitted at 50 Hz");

  // Silence UART for 150 ms -> watchdog must trigger STATE_TIMEOUT_SAFE
  for (uint32_t t = 1010; t <= 1160; t += 10) {
    ctrl.step_100hz(t);
  }
  CHECK(ctrl.state() == kStateTimeoutSafe, "100 ms watchdog enters STATE_TIMEOUT_SAFE");
  std::printf("[PASS] test_cpp_joint_controller_loop (telem=%d, state=%u)\n",
              telem_count, ctrl.state());
}

bool read_exact(uint8_t *dst, size_t n) {
  size_t got = std::fread(dst, 1, n, stdin);
  return got == n;
}

int run_bridge() {
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  mock_hal().reset();
  set_mock_mpu6050_tilt_deg(0.0f, 0.0f);

  JointController ctrl;
  ctrl.init(false);
  uint32_t now_ms = 0;

  while (true) {
    uint8_t hdr[11]{};
    if (std::fread(hdr, 1, 1, stdin) != 1) break;
    if (std::fread(hdr + 1, 1, 10, stdin) != 10) return 2;

    const uint16_t n_bytes = static_cast<uint16_t>(hdr[0]) | (static_cast<uint16_t>(hdr[1]) << 8);
    float roll_deg = 0.0f, pitch_deg = 0.0f;
    std::memcpy(&roll_deg, hdr + 2, 4);
    std::memcpy(&pitch_deg, hdr + 6, 4);
    const uint8_t n_ticks = hdr[10];

    if (n_bytes > 0) {
      std::vector<uint8_t> buf(n_bytes);
      if (!read_exact(buf.data(), n_bytes)) return 3;
      mock_hal().push_uart_rx(buf.data(), buf.size());
    }

    set_mock_mpu6050_tilt_deg(roll_deg, pitch_deg);
    for (uint8_t i = 0; i < n_ticks; ++i) {
      now_ms += 10;
      ctrl.step_100hz(now_ms);
    }

    if (!mock_hal().uart_tx.empty()) {
      std::fwrite(mock_hal().uart_tx.data(), 1, mock_hal().uart_tx.size(), stdout);
      mock_hal().uart_tx.clear();
    }
  }
  std::fflush(stdout);
  return 0;
}

int run_ik_batch() {
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  float in[4]{};
  while (std::fread(in, sizeof(float), 4, stdin) == 4) {
    const LegAngles ik = leg_ik(in[0], in[1], in[2], in[3]);
    const Vec3 fk = leg_fk(ik.hip_roll, ik.hip_pitch, ik.knee, in[3]);
    const float out[6] = {ik.hip_roll, ik.hip_pitch, ik.knee, fk.x, fk.y, fk.z};
    std::fwrite(out, sizeof(float), 6, stdout);
  }
  std::fflush(stdout);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--bridge") {
    return run_bridge();
  }
  if (argc > 1 && std::string(argv[1]) == "--ik-batch") {
    return run_ik_batch();
  }

  test_cpp_kinematics();
  test_cpp_mpu6050_and_mahony();
  test_cpp_uart_parser();
  test_cpp_fsr_debouncer();
  test_cpp_joint_controller_loop();

  if (g_failures > 0) {
    std::fprintf(stderr, "\nFAILED: %d assertion(s)\n", g_failures);
    return 1;
  }
  std::printf("\nALL C++ ESP32 FIRMWARE TESTS PASSED\n");
  return 0;
}
