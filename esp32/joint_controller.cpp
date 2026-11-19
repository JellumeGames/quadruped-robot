#include "joint_controller.h"

#include <cmath>
#include "hal.h"

namespace quad {

FsrDebouncer::FsrDebouncer(float press_thresh, float release_thresh, uint32_t debounce_ms)
    : press_thresh_(press_thresh), release_thresh_(release_thresh), debounce_ms_(debounce_ms) {}

void FsrDebouncer::reset() {
  stable_contact_ = false;
  candidate_contact_ = false;
  candidate_since_ms_ = 0;
}

bool FsrDebouncer::update(float normalized_val, uint32_t now_ms) {
  const bool raw = stable_contact_ ? (normalized_val >= release_thresh_)
                                   : (normalized_val >= press_thresh_);
  if (raw != candidate_contact_) {
    candidate_contact_ = raw;
    candidate_since_ms_ = now_ms;
  } else if (raw != stable_contact_ && (now_ms - candidate_since_ms_) >= debounce_ms_) {
    stable_contact_ = raw;
  }
  return stable_contact_;
}

float JointController::angle_to_us(float angle_rad, float zero_offset_rad, float sign) {
  const float effective = sign * angle_rad + zero_offset_rad;
  const float deg = effective * (180.0f / kPi);
  const float us = 1500.0f + deg * (1000.0f / 90.0f);
  return clampf(us, 500.0f, 2500.0f);
}

uint32_t JointController::us_to_ledc_duty(float us) {
  const float clamped = clampf(us, 500.0f, 2500.0f);
  return static_cast<uint32_t>(std::lround((clamped * 65535.0f) / 20000.0f));
}

void JointController::set_servo_calibration(uint8_t joint_idx, float zero_offset_rad, float sign) {
  if (joint_idx < kNumJoints) {
    servo_zero_rad_[joint_idx] = zero_offset_rad;
    servo_sign_[joint_idx] = (sign < 0.0f) ? -1.0f : 1.0f;
  }
}

void JointController::set_attitude_gains(float kp_att, float kd_att) {
  kp_att_ = kp_att;
  kd_att_ = kd_att;
}

void JointController::set_smoothing_alpha(float alpha) {
  smooth_alpha_ = clampf(alpha, 0.05f, 1.0f);
}

void JointController::compute_safe_stance(float out_angles[kNumJoints]) const {
  for (uint8_t leg = 0; leg < 4; ++leg) {
    const LegAngles ik = leg_ik(0.0f, kLegSide[leg] * kL0, kNominalStandZ,
                                kLegSide[leg], -1.0f, true);
    out_angles[leg * 3 + 0] = ik.hip_roll;
    out_angles[leg * 3 + 1] = ik.hip_pitch;
    out_angles[leg * 3 + 2] = ik.knee;
  }
}

void JointController::set_safe_stance() {
  compute_safe_stance(target_angle_);
  foot_target_mode_ = false;
}

void JointController::init(bool calibrate_imu) {
  hal_uart_begin(115200);
  for (uint8_t i = 0; i < kNumJoints; ++i) {
    hal_ledc_setup(i, kServoPins[i], kPwmFreqHz, kPwmResolutionBits);
    servo_zero_rad_[i] = 0.0f;
    servo_sign_[i] = 1.0f;
  }

  parser_.reset();
  for (uint8_t leg = 0; leg < 4; ++leg) {
    fsr_debouncers_[leg].reset();
    fsr_val_[leg] = 0.0f;
  }

  imu_.init();
  if (calibrate_imu && imu_.healthy()) {
    imu_.calibrate_gyro(64);
  }

  compute_safe_stance(target_angle_);
  for (uint8_t i = 0; i < kNumJoints; ++i) {
    current_angle_[i] = target_angle_[i];
  }
  state_ = kStateStand;
  has_foot_targets_ = false;
  foot_target_mode_ = false;
  tick_count_ = 0;
  tx_seq_ = 0;
  last_tick_ms_ = hal_millis();
  last_frame_ms_ = last_tick_ms_;

  for (uint8_t i = 0; i < kNumJoints; ++i) {
    const float us = angle_to_us(current_angle_[i], servo_zero_rad_[i], servo_sign_[i]);
    hal_ledc_write(i, kServoPins[i], us_to_ledc_duty(us));
  }
}

void JointController::apply_foot_targets(uint32_t now_ms) {
  (void)now_ms;
  if (!has_foot_targets_) return;

  const EulerAngles rpy = imu_.attitude();
  const ImuSample &s = imu_.last_sample();
  const float comp_roll = clampf(-(kp_att_ * rpy.roll + kd_att_ * s.gx_rads), -0.30f, 0.30f);
  const float comp_pitch = clampf(-(kp_att_ * rpy.pitch + kd_att_ * s.gy_rads), -0.30f, 0.30f);

  for (uint8_t leg = 0; leg < 4; ++leg) {
    const Vec3 hip_raw{last_foot_targets_[leg * 3 + 0],
                       last_foot_targets_[leg * 3 + 1],
                       last_foot_targets_[leg * 3 + 2]};
    const Vec3 hip_comp = compensate_body_tilt(leg, hip_raw, comp_roll, comp_pitch);
    const LegAngles ik = leg_ik(hip_comp.x, hip_comp.y, hip_comp.z,
                                kLegSide[leg], -1.0f, true);
    target_angle_[leg * 3 + 0] = ik.hip_roll;
    target_angle_[leg * 3 + 1] = ik.hip_pitch;
    target_angle_[leg * 3 + 2] = ik.knee;
  }
}

void JointController::handle_frame(const ParsedFrame &frame, uint32_t now_ms) {
  switch (frame.cmd) {
    case kCmdMode: {
      if (frame.payload_len < 1) break;
      const uint8_t mode = frame.payload[0];
      uint8_t ack_status = 0;
      if (mode == kModeEstop) {
        state_ = kStateEstop;
        foot_target_mode_ = false;
        for (uint8_t i = 0; i < kNumJoints; ++i) {
          target_angle_[i] = current_angle_[i];
        }
      } else if (mode == kModeIdle) {
        state_ = kStateIdle;
        set_safe_stance();
      } else if (mode == kModeStand) {
        state_ = kStateStand;
        set_safe_stance();
      } else if (mode == kModeTrot) {
        state_ = kStateActive;
      } else {
        ack_status = 1;  // Unknown mode
      }
      if (ack_status == 0) {
        last_frame_ms_ = now_ms;
      }
      uint8_t ack_buf[16]{};
      const size_t n = pack_ack(frame.seq, kCmdMode, ack_status, ack_buf, sizeof(ack_buf));
      if (n > 0) hal_uart_write(ack_buf, n);
      break;
    }

    case kCmdJoints: {
      if (state_ == kStateEstop) break;
      float decoded[12]{};
      if (!decode_joints_payload(frame.payload, frame.payload_len, decoded)) break;
      for (uint8_t leg = 0; leg < 4; ++leg) {
        float r = decoded[leg * 3 + 0];
        float p = decoded[leg * 3 + 1];
        float k = decoded[leg * 3 + 2];
        clamp_leg_joints(r, p, k);
        target_angle_[leg * 3 + 0] = r;
        target_angle_[leg * 3 + 1] = p;
        target_angle_[leg * 3 + 2] = k;
      }
      foot_target_mode_ = false;
      state_ = kStateActive;
      last_frame_ms_ = now_ms;
      break;
    }

    case kCmdFootTargets: {
      if (state_ == kStateEstop) break;
      if (!decode_foot_targets_payload(frame.payload, frame.payload_len, last_foot_targets_)) {
        break;
      }
      has_foot_targets_ = true;
      foot_target_mode_ = true;
      state_ = kStateActive;
      last_frame_ms_ = now_ms;
      apply_foot_targets(now_ms);
      break;
    }

    default:
      break;
  }
}

void JointController::write_servos() {
  for (uint8_t i = 0; i < kNumJoints; ++i) {
    current_angle_[i] += (target_angle_[i] - current_angle_[i]) * smooth_alpha_;
    const float us = angle_to_us(current_angle_[i], servo_zero_rad_[i], servo_sign_[i]);
    hal_ledc_write(i, kServoPins[i], us_to_ledc_duty(us));
  }
}

void JointController::send_telemetry() {
  TelemetryPacket pkt{};
  const EulerAngles rpy = imu_.attitude();
  const ImuSample &s = imu_.last_sample();
  pkt.roll_rad = rpy.roll;
  pkt.pitch_rad = rpy.pitch;
  pkt.yaw_rad = rpy.yaw;
  pkt.gx_rads = s.gx_rads;
  pkt.gy_rads = s.gy_rads;
  pkt.gz_rads = s.gz_rads;
  for (uint8_t leg = 0; leg < 4; ++leg) {
    pkt.fsr[leg] = fsr_val_[leg];
  }
  pkt.controller_state = state_;
  const uint32_t dropped = parser_.dropped_frames();
  pkt.rx_dropped_frames = static_cast<uint8_t>(dropped > 255U ? 255U : dropped);

  uint8_t buf[64]{};
  const size_t n = pack_telemetry(tx_seq_++, pkt, buf, sizeof(buf));
  if (n > 0) hal_uart_write(buf, n);
}

void JointController::step_100hz(uint32_t now_ms) {
  constexpr float kDt = 1.0f / static_cast<float>(kLoopHz);

  imu_.update(kDt);

  for (uint8_t leg = 0; leg < 4; ++leg) {
    fsr_val_[leg] = static_cast<float>(hal_analog_read(kFsrPins[leg])) / 4095.0f;
    fsr_debouncers_[leg].update(fsr_val_[leg], now_ms);
  }

  ParsedFrame frame{};
  while (hal_uart_available() > 0) {
    const int c = hal_uart_read();
    if (c < 0) break;
    if (parser_.feed_byte(static_cast<uint8_t>(c), frame)) {
      handle_frame(frame, now_ms);
    }
  }

  if (state_ == kStateActive && (now_ms - last_frame_ms_) > kLinkTimeoutMs) {
    state_ = kStateTimeoutSafe;
    set_safe_stance();
  } else if (state_ == kStateActive && foot_target_mode_) {
    apply_foot_targets(now_ms);
  }

  write_servos();

  ++tick_count_;
  if ((tick_count_ % 2U) == 0U) {
    send_telemetry();
  }
}

void JointController::tick() {
  const uint32_t now_ms = hal_millis();
  if ((now_ms - last_tick_ms_) < kLoopPeriodMs) return;
  last_tick_ms_ = now_ms;
  step_100hz(now_ms);
}

}  // namespace quad

#ifdef ARDUINO
static quad::JointController g_controller;

void setup() { g_controller.init(true); }

void loop() { g_controller.tick(); }
#endif
