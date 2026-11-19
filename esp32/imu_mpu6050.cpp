#include "imu_mpu6050.h"

#include <cmath>
#include "hal.h"
#include "kinematics.h"

namespace quad {

namespace {

constexpr float kDegToRad = kPi / 180.0f;

inline int16_t read_i16_be(const uint8_t *p) {
  uint16_t u = (static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]);
  return static_cast<int16_t>(u);
}

}  // namespace

MahonyAhrs::MahonyAhrs(float kp, float ki, float accel_gate_g)
    : kp_(kp), ki_(ki), accel_gate_g_(accel_gate_g) {}

void MahonyAhrs::reset() {
  qw_ = 1.0f;
  qx_ = qy_ = qz_ = 0.0f;
  ex_int_ = ey_int_ = ez_int_ = 0.0f;
  accel_gated_count_ = 0;
}

EulerAngles MahonyAhrs::update(float ax_g, float ay_g, float az_g,
                              float gx_rads, float gy_rads, float gz_rads,
                              float dt) {
  if (dt <= 0.0f) return euler();

  float gx = gx_rads;
  float gy = gy_rads;
  float gz = gz_rads;

  const float a_norm = std::sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
  if (a_norm > 1e-6f && std::fabs(a_norm - 1.0f) <= accel_gate_g_) {
    const float ax = ax_g / a_norm;
    const float ay = ay_g / a_norm;
    const float az = az_g / a_norm;

    // Gravity direction estimated from current body-to-world quaternion.
    const float vx = 2.0f * (qx_ * qz_ - qw_ * qy_);
    const float vy = 2.0f * (qw_ * qx_ + qy_ * qz_);
    const float vz = 1.0f - 2.0f * (qx_ * qx_ + qy_ * qy_);

    const float ex = ay * vz - az * vy;
    const float ey = az * vx - ax * vz;
    const float ez = ax * vy - ay * vx;

    if (ki_ > 0.0f) {
      ex_int_ += ex * dt;
      ey_int_ += ey * dt;
      ez_int_ += ez * dt;
      gx += ki_ * ex_int_;
      gy += ki_ * ey_int_;
      gz += ki_ * ez_int_;
    }

    gx += kp_ * ex;
    gy += kp_ * ey;
    gz += kp_ * ez;
  } else {
    ++accel_gated_count_;
  }

  const float half_dt = 0.5f * dt;
  const float dqw = (-qx_ * gx - qy_ * gy - qz_ * gz) * half_dt;
  const float dqx = (qw_ * gx + qy_ * gz - qz_ * gy) * half_dt;
  const float dqy = (qw_ * gy - qx_ * gz + qz_ * gx) * half_dt;
  const float dqz = (qw_ * gz + qx_ * gy - qy_ * gx) * half_dt;

  qw_ += dqw;
  qx_ += dqx;
  qy_ += dqy;
  qz_ += dqz;

  const float q_norm = std::sqrt(qw_ * qw_ + qx_ * qx_ + qy_ * qy_ + qz_ * qz_);
  if (q_norm > 1e-9f) {
    const float inv = 1.0f / q_norm;
    qw_ *= inv;
    qx_ *= inv;
    qy_ *= inv;
    qz_ *= inv;
  }

  return euler();
}

EulerAngles MahonyAhrs::euler() const {
  const float sinr_cosp = 2.0f * (qw_ * qx_ + qy_ * qz_);
  const float cosr_cosp = 1.0f - 2.0f * (qx_ * qx_ + qy_ * qy_);
  const float roll = std::atan2(sinr_cosp, cosr_cosp);

  const float sinp = clampf(2.0f * (qw_ * qy_ - qz_ * qx_), -1.0f, 1.0f);
  const float pitch = std::asin(sinp);

  const float siny_cosp = 2.0f * (qw_ * qz_ + qx_ * qy_);
  const float cosy_cosp = 1.0f - 2.0f * (qy_ * qy_ + qz_ * qz_);
  const float yaw = std::atan2(siny_cosp, cosy_cosp);

  return EulerAngles{roll, pitch, yaw};
}

bool ImuMpu6050::init() {
  hal_i2c_init();
  ahrs_.reset();
  gyro_calibrated_ = false;
  bias_gx_ = bias_gy_ = bias_gz_ = 0.0f;

  uint8_t who = 0;
  if (!hal_i2c_read_regs(kMpu6050Addr, kRegWhoAmI, &who, 1) || who != kMpu6050Addr) {
    healthy_ = false;
    return false;
  }

  bool ok = true;
  ok &= hal_i2c_write_reg(kMpu6050Addr, kRegPwrMgmt1, 0x01);    // Wake, PLL X-gyro
  ok &= hal_i2c_write_reg(kMpu6050Addr, kRegSmplrtDiv, 0x09);   // 100 Hz sample rate
  ok &= hal_i2c_write_reg(kMpu6050Addr, kRegConfig, 0x03);      // 44 Hz DLPF
  ok &= hal_i2c_write_reg(kMpu6050Addr, kRegGyroConfig, 0x08);  // +/- 500 deg/s
  ok &= hal_i2c_write_reg(kMpu6050Addr, kRegAccelConfig, 0x08); // +/- 4 g
  healthy_ = ok;
  return ok;
}

bool ImuMpu6050::read_raw(ImuSample &out) const {
  uint8_t buf[14]{};
  if (!hal_i2c_read_regs(kMpu6050Addr, kRegAccelXoutH, buf, sizeof(buf))) {
    return false;
  }
  const int16_t ax_raw = read_i16_be(buf + 0);
  const int16_t ay_raw = read_i16_be(buf + 2);
  const int16_t az_raw = read_i16_be(buf + 4);
  const int16_t temp_raw = read_i16_be(buf + 6);
  const int16_t gx_raw = read_i16_be(buf + 8);
  const int16_t gy_raw = read_i16_be(buf + 10);
  const int16_t gz_raw = read_i16_be(buf + 12);

  out.ax_g = static_cast<float>(ax_raw) / kAccelScale4g;
  out.ay_g = static_cast<float>(ay_raw) / kAccelScale4g;
  out.az_g = static_cast<float>(az_raw) / kAccelScale4g;
  out.temp_c = (static_cast<float>(temp_raw) / 340.0f) + 36.53f;
  out.gx_rads = (static_cast<float>(gx_raw) / kGyroScale500dps) * kDegToRad - bias_gx_;
  out.gy_rads = (static_cast<float>(gy_raw) / kGyroScale500dps) * kDegToRad - bias_gy_;
  out.gz_rads = (static_cast<float>(gz_raw) / kGyroScale500dps) * kDegToRad - bias_gz_;
  return true;
}

bool ImuMpu6050::calibrate_gyro(uint16_t num_samples) {
  if (!healthy_ || num_samples == 0) return false;

  bias_gx_ = bias_gy_ = bias_gz_ = 0.0f;
  float sum_gx = 0.0f, sum_gy = 0.0f, sum_gz = 0.0f;
  float sum_a = 0.0f, sum_a2 = 0.0f;

  for (uint16_t i = 0; i < num_samples; ++i) {
    ImuSample s{};
    if (!read_raw(s)) return false;
    sum_gx += s.gx_rads;
    sum_gy += s.gy_rads;
    sum_gz += s.gz_rads;
    const float a_norm = std::sqrt(s.ax_g * s.ax_g + s.ay_g * s.ay_g + s.az_g * s.az_g);
    sum_a += a_norm;
    sum_a2 += a_norm * a_norm;
  }

  const float n = static_cast<float>(num_samples);
  const float mean_a = sum_a / n;
  const float var_a = (sum_a2 / n) - (mean_a * mean_a);
  if (var_a > 0.08f * 0.08f) {
    return false;
  }

  bias_gx_ = sum_gx / n;
  bias_gy_ = sum_gy / n;
  bias_gz_ = sum_gz / n;
  gyro_calibrated_ = true;
  return true;
}

bool ImuMpu6050::update(float dt) {
  if (!healthy_) return false;
  ImuSample s{};
  if (!read_raw(s)) return false;
  last_sample_ = s;
  attitude_ = ahrs_.update(s.ax_g, s.ay_g, s.az_g, s.gx_rads, s.gy_rads, s.gz_rads, dt);
  return true;
}

}  // namespace quad
