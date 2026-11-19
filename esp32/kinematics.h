// 3-DOF analytical leg kinematics and body attitude projection for ESP32.
// Mirrors `pi/kinematics.py` constants, frame conventions, and clamping rules.

#pragma once

#include <cmath>
#include <cstdint>

namespace quad {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kL0 = 0.045f;           // Coxa / hip-roll offset (m)
constexpr float kL1 = 0.080f;           // Thigh length (m)
constexpr float kL2 = 0.120f;           // Shin length (m)
constexpr float kBodyLength = 0.240f;
constexpr float kBodyWidth = 0.140f;
constexpr float kNominalStandZ = -0.155f;

constexpr float kHipRollMin = -0.60f;
constexpr float kHipRollMax = 0.60f;
constexpr float kHipPitchMin = -1.20f;
constexpr float kHipPitchMax = 1.40f;
constexpr float kKneeMin = -2.45f;
constexpr float kKneeMax = -0.35f;

// Leg order: 0=LF, 1=RF, 2=LR, 3=RR
constexpr float kLegSide[4] = {1.0f, -1.0f, 1.0f, -1.0f};
constexpr float kHipOffsets[4][3] = {
    {0.5f * kBodyLength, 0.5f * kBodyWidth, 0.0f},
    {0.5f * kBodyLength, -0.5f * kBodyWidth, 0.0f},
    {-0.5f * kBodyLength, 0.5f * kBodyWidth, 0.0f},
    {-0.5f * kBodyLength, -0.5f * kBodyWidth, 0.0f},
};

struct Vec3 {
  float x;
  float y;
  float z;
};

struct LegAngles {
  float hip_roll;
  float hip_pitch;
  float knee;
  bool clamped;
};

inline float clampf(float v, float lo, float hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

inline bool clamp_leg_joints(float &hip_roll, float &hip_pitch, float &knee) {
  float r = clampf(hip_roll, kHipRollMin, kHipRollMax);
  float p = clampf(hip_pitch, kHipPitchMin, kHipPitchMax);
  float k = clampf(knee, kKneeMin, kKneeMax);
  bool changed = (r != hip_roll) || (p != hip_pitch) || (k != knee);
  hip_roll = r;
  hip_pitch = p;
  knee = k;
  return changed;
}

inline Vec3 leg_fk(float hip_roll, float hip_pitch, float knee, float side) {
  const float y_s = side * kL0;
  const float x_s = kL1 * std::sin(hip_pitch) + kL2 * std::sin(hip_pitch + knee);
  const float z_s = -kL1 * std::cos(hip_pitch) - kL2 * std::cos(hip_pitch + knee);

  const float c_r = std::cos(hip_roll);
  const float s_r = std::sin(hip_roll);
  return Vec3{x_s, y_s * c_r - z_s * s_r, y_s * s_r + z_s * c_r};
}

inline LegAngles leg_ik(float x, float y, float z, float side,
                        float knee_sign = -1.0f, bool clamp_joints = false) {
  const float y_s = side * kL0;
  bool clamped = false;

  float r_yz2 = y * y + z * z;
  const float min_yz2 = kL0 * kL0 + 1e-6f;
  if (r_yz2 < min_yz2) {
    clamped = true;
    const float r_yz = std::sqrt(r_yz2);
    const float target_r = std::sqrt(min_yz2);
    if (r_yz < 1e-9f) {
      y = y_s;
      z = -1e-3f;
    } else {
      const float scale = target_r / r_yz;
      y *= scale;
      z *= scale;
    }
    r_yz2 = y * y + z * z;
  }

  const float z_s_sq = r_yz2 - kL0 * kL0;
  const float z_s = -std::sqrt(z_s_sq > 0.0f ? z_s_sq : 0.0f);
  float hip_roll = std::atan2(y_s * z - z_s * y, y_s * y + z_s * z);

  float x_s = x;
  float z_s_adj = z_s;
  const float d_min = std::fabs(kL1 - kL2) + 1e-4f;
  const float d_max = (kL1 + kL2) - 1e-4f;
  float d = std::hypot(x_s, z_s_adj);
  if (d < d_min || d > d_max) {
    clamped = true;
    const float d_clamped = clampf(d, d_min, d_max);
    if (d < 1e-9f) {
      x_s = 0.0f;
      z_s_adj = -d_clamped;
    } else {
      const float scale = d_clamped / d;
      x_s *= scale;
      z_s_adj *= scale;
    }
    d = d_clamped;
  }

  const float cos_k =
      clampf((d * d - kL1 * kL1 - kL2 * kL2) / (2.0f * kL1 * kL2), -1.0f, 1.0f);
  float knee = (knee_sign < 0.0f ? -1.0f : 1.0f) * std::acos(cos_k);

  const float a = kL1 + kL2 * std::cos(knee);
  const float b = kL2 * std::sin(knee);
  float hip_pitch = std::atan2(a * x_s + b * z_s_adj, b * x_s - a * z_s_adj);

  if (clamp_joints) {
    if (clamp_leg_joints(hip_roll, hip_pitch, knee)) {
      clamped = true;
    }
  }
  return LegAngles{hip_roll, hip_pitch, knee, clamped};
}

// Applies a body tilt (roll, pitch) compensation around the body center to
// a hip-frame target, returning the adjusted hip-frame foot target.
inline Vec3 compensate_body_tilt(uint8_t leg, Vec3 hip_target,
                                 float comp_roll, float comp_pitch) {
  const float hx = kHipOffsets[leg][0];
  const float hy = kHipOffsets[leg][1];
  const float hz = kHipOffsets[leg][2];
  const float wx = hx + hip_target.x;
  const float wy = hy + hip_target.y;
  const float wz = hz + hip_target.z;

  const float cr = std::cos(comp_roll), sr = std::sin(comp_roll);
  const float cp = std::cos(comp_pitch), sp = std::sin(comp_pitch);

  // R = Ry(pitch) * Rx(roll); p_body = R^T * p_world
  const float bx = cp * wx - sp * wz;
  const float by = sr * sp * wx + cr * wy + sr * cp * wz;
  const float bz = cr * sp * wx - sr * wy + cr * cp * wz;

  return Vec3{bx - hx, by - hy, bz - hz};
}

}  // namespace quad
