// ESP32 low-level 100 Hz joint, IMU, FSR, and LEDC PWM controller.

#pragma once

#include <cstdint>

#include "imu_mpu6050.h"
#include "kinematics.h"
#include "uart_parser.h"

namespace quad {

constexpr uint8_t kNumJoints = 12;
constexpr uint16_t kLoopHz = 100;
constexpr uint32_t kLoopPeriodMs = 1000 / kLoopHz;
constexpr uint32_t kLinkTimeoutMs = 100;
constexpr uint32_t kPwmFreqHz = 50;
constexpr uint8_t kPwmResolutionBits = 16;

constexpr uint8_t kServoPins[kNumJoints] = {
    13, 12, 14,  // LF: hip_roll, hip_pitch, knee
    27, 26, 25,  // RF: hip_roll, hip_pitch, knee
    33, 32, 15,  // LR: hip_roll, hip_pitch, knee
    4,  16, 17,  // RR: hip_roll, hip_pitch, knee
};
constexpr uint8_t kFsrPins[4] = {34, 35, 36, 37};

class FsrDebouncer {
 public:
  explicit FsrDebouncer(float press_thresh = 0.25f, float release_thresh = 0.15f,
                        uint32_t debounce_ms = 20);

  void reset();
  bool update(float normalized_val, uint32_t now_ms);
  bool contact() const { return stable_contact_; }

 private:
  float press_thresh_;
  float release_thresh_;
  uint32_t debounce_ms_;
  bool stable_contact_ = false;
  bool candidate_contact_ = false;
  uint32_t candidate_since_ms_ = 0;
};

class JointController {
 public:
  void init(bool calibrate_imu = false);
  void tick();
  void step_100hz(uint32_t now_ms);

  void set_servo_calibration(uint8_t joint_idx, float zero_offset_rad, float sign);
  void set_attitude_gains(float kp_att, float kd_att);
  void set_smoothing_alpha(float alpha);

  static float angle_to_us(float angle_rad, float zero_offset_rad = 0.0f, float sign = 1.0f);
  static uint32_t us_to_ledc_duty(float us);

  uint8_t state() const { return state_; }
  float target_angle(uint8_t idx) const { return (idx < kNumJoints) ? target_angle_[idx] : 0.0f; }
  float current_angle(uint8_t idx) const { return (idx < kNumJoints) ? current_angle_[idx] : 0.0f; }
  float fsr_value(uint8_t leg) const { return (leg < 4) ? fsr_val_[leg] : 0.0f; }
  bool fsr_contact(uint8_t leg) const { return (leg < 4) ? fsr_debouncers_[leg].contact() : false; }
  const ImuMpu6050 &imu() const { return imu_; }
  const UartParser &uart_parser() const { return parser_; }

 private:
  void compute_safe_stance(float out_angles[kNumJoints]) const;
  void set_safe_stance();
  void handle_frame(const ParsedFrame &frame, uint32_t now_ms);
  void apply_foot_targets(uint32_t now_ms);
  void write_servos();
  void send_telemetry();

  ImuMpu6050 imu_;
  UartParser parser_;
  FsrDebouncer fsr_debouncers_[4];

  float fsr_val_[4]{0.0f, 0.0f, 0.0f, 0.0f};
  float target_angle_[kNumJoints]{};
  float current_angle_[kNumJoints]{};
  float servo_zero_rad_[kNumJoints]{};
  float servo_sign_[kNumJoints]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float last_foot_targets_[12]{};
  bool has_foot_targets_ = false;
  bool foot_target_mode_ = false;

  float kp_att_ = 0.85f;
  float kd_att_ = 0.12f;
  float smooth_alpha_ = 0.5f;

  uint8_t state_ = kStateStand;
  uint8_t tx_seq_ = 0;
  uint32_t tick_count_ = 0;
  uint32_t last_tick_ms_ = 0;
  uint32_t last_frame_ms_ = 0;
};

}  // namespace quad
