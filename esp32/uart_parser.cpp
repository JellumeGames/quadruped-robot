// UART frame parser for the ESP32 side.
// Frame: 0xAA 0x55 | len | cmd | seq | payload | CRC16-CCITT | 0x0D 0x0A
// Byte-wise state machine: resyncs on the header, verifies length and
// CRC, silently drops bad frames. Mirrors pi/uart_link.py.

#include <Arduino.h>

#define HDR0 0xAA
#define HDR1 0x55

enum ParseState { WAIT_HDR0, WAIT_HDR1, WAIT_LEN, WAIT_BODY };

static ParseState state = WAIT_HDR0;
static uint8_t buf[64];
static uint8_t len = 0, idx = 0;

static uint16_t crc16(const uint8_t *data, uint8_t n) {
  uint16_t crc = 0xFFFF;
  for (uint8_t i = 0; i < n; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (uint8_t b = 0; b < 8; b++)
      crc = (crc & 0x8000) ? ((crc << 1) ^ 0x1021) : (crc << 1);
  }
  return crc;
}

// Returns true when a complete valid frame is ready in buf.
// buf layout on success: [cmd, seq, payload...], *outLen = payload length.
bool parseByte(uint8_t b, uint8_t *outLen) {
  switch (state) {
    case WAIT_HDR0:
      if (b == HDR0) state = WAIT_HDR1;
      return false;
    case WAIT_HDR1:
      state = (b == HDR1) ? WAIT_LEN : WAIT_HDR0;
      return false;
    case WAIT_LEN:
      len = b;
      idx = 0;
      state = WAIT_BODY;
      return false;
    case WAIT_BODY:
      buf[idx++] = b;
      if (idx < len + 4) return false;  // cmd+seq+payload+CRC16+tail
      state = WAIT_HDR0;
      if (buf[len + 2] != 0x0D || buf[len + 3] != 0x0A) return false;
      // CRC covers the length byte + body (cmd, seq, payload),
      // exactly as packed by pi/uart_link.py.
      uint8_t tmp[65];
      tmp[0] = len;
      memcpy(tmp + 1, buf, len);
      uint16_t crc = crc16(tmp, len + 1);
      uint16_t want = (uint16_t)(buf[len] << 8) | buf[len + 1];
      if (crc != want) return false;
      *outLen = len - 2;
      return true;
  }
  return false;
}
