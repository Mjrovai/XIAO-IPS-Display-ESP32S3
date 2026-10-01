/*
  Test 06 - Battery voltage
  The battery is sensed on D16 through a 316k / 160k divider, so
  VBAT = ADC_mV * (316 + 160) / 160 = ADC_mV * 2.975.
  Do not trust the number without a LiPo connected: on our unit it read
  about 3.97 V with no battery and the power switch off, so a reading alone
  does not prove a battery is present.
  Shows volts, millivolts, a rough percentage (3.30 V empty .. 4.20 V full
  for a LiPo, a coarse estimate only), and a history graph.
  Note: the ESP32-S3 version cannot detect charge status, only voltage.
*/

#include <Arduino.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

Seeed_GFX display;
Seeed_Sprite canvas;

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

static constexpr uint8_t BAT_ADC_PIN = D16;
// VBAT -- 316k -- D16 -- 160k -- GND: the pin sees only 160/476 of VBAT.
static constexpr float DIVIDER = (316.0f + 160.0f) / 160.0f;

static constexpr int HIST_W = 172;
static float hist[HIST_W];

static uint32_t readBatteryMv() {
  uint32_t sum = 0;
  // Average 16 readings to smooth the ADC noise.
  for (int i = 0; i < 16; i++) { sum += analogReadMilliVolts(BAT_ADC_PIN); delay(2); }
  return (uint32_t)((sum / 16.0f) * DIVIDER);
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(500);
  analogReadResolution(12);
  // 11 dB attenuation gives the widest input range.
  analogSetPinAttenuation(BAT_ADC_PIN, ADC_11db);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) {
    Serial.println(display.lastResult().message);
    return;
  }
  canvas.createSprite(display, 172, 320);
  for (int i = 0; i < HIST_W; i++) hist[i] = 0;
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last < 500) return;
  last = millis();

  uint32_t mv = readBatteryMv();
  float v = mv / 1000.0f;
  // A straight line from 3.30 V (empty) to 4.20 V (full). A real LiPo curve is
  // not straight, so this is only a rough guide.
  int pct = constrain((int)((v - 3.30f) / (4.20f - 3.30f) * 100.0f), 0, 100);
  Serial.printf("battery_mV=%lu  V=%.3f  est=%d%%\n", (unsigned long)mv, v, pct);

  memmove(hist, hist + 1, sizeof(float) * (HIST_W - 1));
  hist[HIST_W - 1] = v;

  // Red below 3.4 V, yellow below 3.7 V, green above.
  uint16_t col = v < 3.4f ? TFT_RED : (v < 3.7f ? TFT_YELLOW : TFT_GREEN);
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.setTextSize(2);
  canvas.drawCentreString("BATTERY", 86, 8, 1);
  char t[24];
  snprintf(t, sizeof(t), "%.3f V", v);
  canvas.setTextColor(col, TFT_BLACK);
  canvas.setTextSize(3);
  canvas.drawCentreString(t, 86, 50, 1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.setTextSize(2);
  snprintf(t, sizeof(t), "%lu mV", (unsigned long)mv);
  canvas.drawCentreString(t, 86, 92, 1);
  snprintf(t, sizeof(t), "~%d %%", pct);
  canvas.drawCentreString(t, 86, 118, 1);

  canvas.drawRect(10, 150, 152, 26, TFT_WHITE);
  canvas.fillRect(12, 152, (148 * pct) / 100, 22, col);

  canvas.setTextSize(1);
  canvas.setCursor(4, 190);
  canvas.print("history 0 .. 4.5 V");
  canvas.drawRect(0, 202, 172, 101, TFT_DARKGREY);
  for (int i = 0; i < HIST_W; i++) {
    int h = (int)(100 * constrain(hist[i] / 4.5f, 0.0f, 1.0f));
    canvas.drawFastVLine(i, 302 - h, h, col);
  }
  canvas.pushSprite(0, 0);
}
