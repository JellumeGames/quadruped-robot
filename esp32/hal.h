// Hardware Abstraction Layer for ESP32 firmware.
// Compiles against real ESP32 Arduino APIs when ARDUINO is defined, or against
// an in-memory MockHal when built natively on a host machine with g++.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef ARDUINO

#include <Arduino.h>
#include <Wire.h>

inline uint32_t hal_millis() { return static_cast<uint32_t>(millis()); }

inline uint16_t hal_analog_read(uint8_t pin) {
  return static_cast<uint16_t>(analogRead(pin));
}

inline void hal_ledc_setup(uint8_t channel, uint8_t pin, uint32_t freq_hz, uint8_t resolution_bits) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcAttachChannel(pin, freq_hz, resolution_bits, channel);
#else
  ledcSetup(channel, freq_hz, resolution_bits);
  ledcAttachPin(pin, channel);
#endif
}

inline void hal_ledc_write(uint8_t channel, uint8_t pin, uint32_t duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  (void)channel;
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(channel, duty);
#endif
}

inline void hal_uart_begin(uint32_t baud) { Serial.begin(baud); }
inline int hal_uart_available() { return Serial.available(); }
inline int hal_uart_read() { return Serial.read(); }
inline size_t hal_uart_write(const uint8_t *buf, size_t len) { return Serial.write(buf, len); }

inline void hal_i2c_init() {
  Wire.begin();
  Wire.setClock(400000);
}

inline bool hal_i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

inline bool hal_i2c_read_regs(uint8_t addr, uint8_t reg, uint8_t *out, size_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  size_t got = Wire.requestFrom(static_cast<int>(addr), static_cast<int>(len));
  if (got != len) return false;
  for (size_t i = 0; i < len; ++i) {
    out[i] = static_cast<uint8_t>(Wire.read());
  }
  return true;
}

#else

#include <array>
#include <deque>
#include <vector>

struct MockHal {
  uint32_t millis_now = 0;
  std::array<uint16_t, 40> adc_raw{};
  std::array<uint32_t, 16> ledc_duty{};
  std::array<uint8_t, 16> ledc_pin{};
  std::array<uint32_t, 16> ledc_freq{};
  std::array<uint8_t, 16> ledc_res{};
  std::deque<uint8_t> uart_rx{};
  std::vector<uint8_t> uart_tx{};
  uint32_t uart_baud = 0;
  bool i2c_initialized = false;
  bool i2c_present = true;
  uint8_t i2c_addr = 0x68;
  std::array<uint8_t, 256> i2c_regs{};

  void reset() {
    millis_now = 0;
    adc_raw.fill(0);
    ledc_duty.fill(0);
    ledc_pin.fill(0);
    ledc_freq.fill(0);
    ledc_res.fill(0);
    uart_rx.clear();
    uart_tx.clear();
    uart_baud = 0;
    i2c_initialized = false;
    i2c_present = true;
    i2c_addr = 0x68;
    i2c_regs.fill(0);
    i2c_regs[0x75] = 0x68;  // MPU6050 WHO_AM_I default
  }

  void push_uart_rx(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; ++i) uart_rx.push_back(data[i]);
  }
};

inline MockHal &mock_hal() {
  static MockHal inst;
  return inst;
}

inline uint32_t hal_millis() { return mock_hal().millis_now; }

inline uint16_t hal_analog_read(uint8_t pin) {
  if (pin >= mock_hal().adc_raw.size()) return 0;
  return mock_hal().adc_raw[pin];
}

inline void hal_ledc_setup(uint8_t channel, uint8_t pin, uint32_t freq_hz, uint8_t resolution_bits) {
  if (channel < 16) {
    mock_hal().ledc_pin[channel] = pin;
    mock_hal().ledc_freq[channel] = freq_hz;
    mock_hal().ledc_res[channel] = resolution_bits;
  }
}

inline void hal_ledc_write(uint8_t channel, uint8_t pin, uint32_t duty) {
  (void)pin;
  if (channel < 16) {
    mock_hal().ledc_duty[channel] = duty;
  }
}

inline void hal_uart_begin(uint32_t baud) { mock_hal().uart_baud = baud; }
inline int hal_uart_available() { return static_cast<int>(mock_hal().uart_rx.size()); }

inline int hal_uart_read() {
  if (mock_hal().uart_rx.empty()) return -1;
  uint8_t b = mock_hal().uart_rx.front();
  mock_hal().uart_rx.pop_front();
  return static_cast<int>(b);
}

inline size_t hal_uart_write(const uint8_t *buf, size_t len) {
  mock_hal().uart_tx.insert(mock_hal().uart_tx.end(), buf, buf + len);
  return len;
}

inline void hal_i2c_init() { mock_hal().i2c_initialized = true; }

inline bool hal_i2c_write_reg(uint8_t addr, uint8_t reg, uint8_t val) {
  if (!mock_hal().i2c_present || addr != mock_hal().i2c_addr) return false;
  mock_hal().i2c_regs[reg] = val;
  return true;
}

inline bool hal_i2c_read_regs(uint8_t addr, uint8_t reg, uint8_t *out, size_t len) {
  if (!mock_hal().i2c_present || addr != mock_hal().i2c_addr) return false;
  for (size_t i = 0; i < len; ++i) {
    out[i] = mock_hal().i2c_regs[static_cast<uint8_t>(reg + i)];
  }
  return true;
}

#endif
