// MPU6050 I2C driver, startup gyro bias calibrator, and Mahony AHRS filter.
// Mirrors `pi/imu_filter.py` scale factors, shock gating, and Euler output.

#pragma once

#include <cstdint>

namespace quad {

constexpr uint8_t kMpu6050Addr = 0x68;
constexpr uint8_t kRegSmplrtDiv = 0x19;
constexpr uint8_t kRegConfig = 0x1A;
constexpr uint8_t kRegGyroConfig = 0x1B;
constexpr uint8_t kRegAccelConfig = 0x1C;
constexpr uint8_t kRegAccelXoutH = 0x3B;
constexpr uint8_t kRegPwrMgmt1 = 0x6B;
constexpr uint8_t kRegWhoAmI = 0x75;

constexpr float kAccelScale4g = 8192.0f;
constexpr float kGyroScale500dps = 65.5f;

struct ImuSample {
  float ax_g = 0.0f;
  float ay_g = 0.0f;
  float az_g = 1.0f;
  float temp_c = 25.0f;
  float gx_rads = 0.0f;
  float gy_rads = 0.0f;
  float gz_rads = 0.0f;
};

struct EulerAngles {
  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;
};

class MahonyAhrs {
 public:
  explicit MahonyAhrs(float kp = 2.0f, float ki = 0.05f, float accel_gate_g = 0.25f);

  void reset();
  EulerAngles update(float ax_g, float ay_g, float az_g,
                     float gx_rads, float gy_rads, float gz_rads, float dt);
  EulerAngles euler() const;
  uint32_t accel_gated_count() const { return accel_gated_count_; }

 private:
  float kp_;
  float ki_;
  float accel_gate_g_;
  float qw_ = 1.0f, qx_ = 0.0f, qy_ = 0.0f, qz_ = 0.0f;
  float ex_int_ = 0.0f, ey_int_ = 0.0f, ez_int_ = 0.0f;
  uint32_t accel_gated_count_ = 0;
};

class ImuMpu6050 {
 public:
  bool init();
  bool read_raw(ImuSample &out) const;
  bool calibrate_gyro(uint16_t num_samples = 64);
  bool update(float dt);

  bool healthy() const { return healthy_; }
  bool gyro_calibrated() const { return gyro_calibrated_; }
  EulerAngles attitude() const { return attitude_; }
  const ImuSample &last_sample() const { return last_sample_; }
  void gyro_bias(float &bx, float &by, float &bz) const {
    bx = bias_gx_;
    by = bias_gy_;
    bz = bias_gz_;
  }
  uint32_t accel_gated_count() const { return ahrs_.accel_gated_count(); }

 private:
  MahonyAhrs ahrs_;
  bool healthy_ = false;
  bool gyro_calibrated_ = false;
  float bias_gx_ = 0.0f;
  float bias_gy_ = 0.0f;
  float bias_gz_ = 0.0f;
  ImuSample last_sample_{};
  EulerAngles attitude_{};
};

}  // namespace quad
