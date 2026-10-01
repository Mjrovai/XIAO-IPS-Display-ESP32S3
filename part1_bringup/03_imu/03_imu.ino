/*
  Test 03 - IMU (6-axis accelerometer + gyroscope)
  I2C: SDA=D4, SCL=D5, address 0x6A/0x6B. The board may carry a QMI8658
  or an LSM6-family chip, so the sketch reads WHO_AM_I and adapts.
  Screen: chip name, a bubble level, and live ax/ay/az (g) and gx/gy/gz (dps).
  Serial: one CSV line per frame (ax,ay,az,gx,gy,gz).
  Drawing goes into an off-screen sprite and is pushed in one transfer;
  drawing directly on the panel took 740 ms per frame in our measurement.
  Pass: with the board flat and still, az is about +/-1.00 g, ax and ay
  are near 0, and |a| is between 0.95 and 1.05 g.
*/

#include <Arduino.h>
#include <Wire.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

Seeed_GFX display;
Seeed_Sprite canvas;  // full-screen off-screen buffer, pushed once per frame

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

enum ImuType { IMU_NONE, IMU_QMI8658, IMU_LSM6 };
static ImuType imuType = IMU_NONE;
static uint8_t imuAddr = 0;
static const char *imuName = "none";
static float accelScale = 1.0f;  // LSB -> g
static float gyroScale = 1.0f;   // LSB -> dps

static bool i2cWrite8(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cRead(uint8_t addr, uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)addr, (int)len) != (int)len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

static int16_t le16(const uint8_t *p) {
  return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

// QMI8658: WHO_AM_I is register 0x00 and reads 0x05.
static bool initQmi(uint8_t addr) {
  uint8_t who = 0;
  if (!i2cRead(addr, 0x00, &who, 1) || who != 0x05) return false;
  i2cWrite8(addr, 0x02, 0x60);  // CTRL1: address auto-increment
  i2cWrite8(addr, 0x03, 0x03);  // CTRL2: accel +/-2 g
  i2cWrite8(addr, 0x04, 0x53);  // CTRL3: gyro +/-512 dps
  i2cWrite8(addr, 0x08, 0x03);  // CTRL7: enable accel + gyro
  delay(20);
  imuType = IMU_QMI8658;
  imuAddr = addr;
  imuName = "QMI8658";
  accelScale = 1.0f / 16384.0f;
  gyroScale = 1.0f / 64.0f;
  return true;
}

// LSM6 family: WHO_AM_I is register 0x0F (0x6A = LSM6DS3TR-C, 0x69 = LSM6DS3).
static bool initLsm(uint8_t addr) {
  uint8_t who = 0;
  if (!i2cRead(addr, 0x0F, &who, 1)) return false;
  if (who != 0x69 && who != 0x6A && who != 0x6C) return false;
  i2cWrite8(addr, 0x10, 0x60);  // CTRL1_XL: 416 Hz, +/-2 g
  i2cWrite8(addr, 0x11, 0x60);  // CTRL2_G: 416 Hz, 250 dps
  delay(20);
  imuType = IMU_LSM6;
  imuAddr = addr;
  imuName = (who == 0x6A) ? "LSM6DS3TR-C" : (who == 0x69) ? "LSM6DS3" : "LSM6DSx";
  accelScale = 0.000061f;
  gyroScale = 0.00875f;
  return true;
}

// Reads one sample: acceleration in g and rotation in degrees per second.
static bool readImu(float a[3], float g[3]) {
  uint8_t d[12];
  if (imuType == IMU_QMI8658) {
    if (!i2cRead(imuAddr, 0x35, d, 12)) return false;  // accel then gyro
  } else if (imuType == IMU_LSM6) {
    uint8_t gd[6], ad[6];
    if (!i2cRead(imuAddr, 0x22, gd, 6)) return false;  // gyro first
    if (!i2cRead(imuAddr, 0x28, ad, 6)) return false;
    memcpy(d, ad, 6);
    memcpy(d + 6, gd, 6);
  } else {
    return false;
  }
  for (int i = 0; i < 3; i++) {
    a[i] = le16(&d[i * 2]) * accelScale;
    g[i] = le16(&d[6 + i * 2]) * gyroScale;
  }
  return true;
}

static uint32_t fpsFrames = 0, fpsStamp = 0;
static float fpsValue = 0;
static uint32_t lastFrameMs = 0, worstFrameMs = 0;

static constexpr int CX = 86, CY = 120, RING = 70;

static void drawStatic() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString(imuName, 86, 6, 1);
  canvas.drawCircle(CX, CY, RING, TFT_DARKGREY);
  canvas.drawCircle(CX, CY, RING / 2, TFT_DARKGREY);
  canvas.drawFastHLine(CX - RING, CY, 2 * RING, TFT_DARKGREY);
  canvas.drawFastVLine(CX, CY - RING, 2 * RING, TFT_DARKGREY);
}

static void drawValues(const float a[3], const float g[3]) {
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  char buf[32];
  const char *an[] = {"ax", "ay", "az"};
  const char *gn[] = {"gx", "gy", "gz"};
  for (int i = 0; i < 3; i++) {
    snprintf(buf, sizeof(buf), "%s %+6.2f g", an[i], a[i]);
    canvas.setCursor(8, 210 + i * 12);
    canvas.print(buf);
    snprintf(buf, sizeof(buf), "%s %+7.1f dps", gn[i], g[i]);
    canvas.setCursor(8, 252 + i * 12);
    canvas.print(buf);
  }
  float mag = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  snprintf(buf, sizeof(buf), "|a| %.3f g", mag);
  canvas.setTextColor(fabsf(mag - 1.0f) < 0.05f ? TFT_GREEN : TFT_YELLOW, TFT_BLACK);
  canvas.setCursor(8, 294);
  canvas.print(buf);
  snprintf(buf, sizeof(buf), "%.1f fps  %lums", fpsValue, (unsigned long)worstFrameMs);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.setCursor(8, 308);
  canvas.print(buf);
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);  // never block when no host is reading the USB port
  delay(500);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) {
    Serial.println(display.lastResult().message);
    return;
  }
  Wire.begin(D4, D5);
  Wire.setClock(400000);
  bool ok = initQmi(0x6B) || initQmi(0x6A) || initLsm(0x6A) || initLsm(0x6B);
  Serial.printf("IMU: %s at 0x%02X\n", ok ? imuName : "NOT FOUND", imuAddr);
  if (!canvas.createSprite(display, 172, 320)) {
    Serial.println("sprite allocation failed");
  }
  Serial.println("ax,ay,az,gx,gy,gz");
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last < 20) return;
  last = millis();

  float a[3], g[3];
  if (!readImu(a, g)) {
    Serial.println("read error");
    return;
  }
  Serial.printf("%.3f,%.3f,%.3f,%.1f,%.1f,%.1f\n", a[0], a[1], a[2], g[0], g[1], g[2]);

  uint32_t t0 = millis();
  drawStatic();
  // Bubble: tilt of 0.5 g (about 30 degrees) reaches the outer ring.
  float dx = constrain(a[0] / 0.5f, -1.0f, 1.0f) * (RING - 10);
  float dy = constrain(a[1] / 0.5f, -1.0f, 1.0f) * (RING - 10);
  bool level = fabsf(a[0]) < 0.05f && fabsf(a[1]) < 0.05f;
  canvas.fillCircle(CX + (int)dx, CY - (int)dy, 8, level ? TFT_GREEN : TFT_ORANGE);
  drawValues(a, g);
  canvas.pushSprite(0, 0);
  fpsFrames++;
  uint32_t now = millis();
  if (now - lastFrameMs > worstFrameMs || now - fpsStamp > 1000) worstFrameMs = now - lastFrameMs;
  lastFrameMs = now;
  if (now - fpsStamp >= 1000) { fpsValue = fpsFrames * 1000.0f / (now - fpsStamp); fpsFrames = 0; fpsStamp = now; worstFrameMs = 0; }
  Serial.printf("# draw_ms=%lu\n", (unsigned long)(millis() - t0));
}
