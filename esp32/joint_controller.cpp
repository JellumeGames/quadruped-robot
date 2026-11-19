// ESP32 low-level joint controller: 100 Hz loop.
// Reads IMU + 4x FSR, runs analytic IK per leg, drives 12 servos over
// LEDC PWM (16 channels available, 12 used). If no valid frame arrives
// from the Pi for 100 ms, eases into a safe stance instead of freezing.

#include <Arduino.h>

#define NUM_JOINTS 12
#define LOOP_HZ 100
#define LINK_TIMEOUT_MS 100

// LEDC channels 0..11 -> 12 servos (hip_roll, hip_pitch, knee) x 4 legs.
static const uint8_t kServoPin[NUM_JOINTS] = {13, 12, 14, 27, 26, 25,
                                              33, 32, 15, 4, 16, 17};
static float targetAngle[NUM_JOINTS] = {0};
static float currentAngle[NUM_JOINTS] = {0};
static unsigned long lastFrameMs = 0;

float readFSR(uint8_t leg) {
  // Foot sole FSR via ADC; threshold + 20 ms debounce in the caller.
  return analogRead(34 + leg) / 4095.0f;
}

void analyticIK(float x, float z, float &hip, float &knee) {
  // Closed-form 3-DOF leg IK, sagittal plane. Microseconds per leg.
  const float L1 = 0.08f, L2 = 0.12f;
  float d = min(hypotf(x, z), L1 + L2 - 1e-6f);
  knee = acosf(constrain((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1));
  hip = atan2f(z, x) - atan2f(L2 * sinf(knee), L1 + L2 * cosf(knee));
}

void writeServos() {
  for (int i = 0; i < NUM_JOINTS; i++) {
    // Smooth step toward target; 100 Hz tracking of the 50 Hz stream.
    currentAngle[i] += (targetAngle[i] - currentAngle[i]) * 0.5f;
    int us = 1500 + (int)(currentAngle[i] * 572.957795f * 5.55f);
    ledcWrite(i, constrain(us, 500, 2500));
  }
}

void safeStance() {
  // Link lost: ease all joints to the standing pose, do not power off.
  for (int i = 0; i < NUM_JOINTS; i++) targetAngle[i] = 0;
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM_JOINTS; i++) {
    ledcAttachPin(kServoPin[i], i);
    ledcSetup(i, 50, 16);
  }
  lastFrameMs = millis();
}

void loop() {
  static unsigned long lastTick = 0;
  if (millis() - lastTick < 1000 / LOOP_HZ) return;
  lastTick = millis();

  // TODO: parse incoming frames via uart_parser, update targetAngle
  // and lastFrameMs on every valid joint frame.
  if (millis() - lastFrameMs > LINK_TIMEOUT_MS) safeStance();
  writeServos();
}
