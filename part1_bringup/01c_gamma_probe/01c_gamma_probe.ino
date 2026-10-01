/*
  Test 01c - Gamma curve probe
  The JD9853A driver in Seeed_GFX2 reuses the ST7789 init table, so the
  grays may carry a blue cast. The standard GAMSET command (0x26) selects
  one of four built-in gamma curves: 0x01, 0x02, 0x04, 0x08.
  This sketch shows an 8-step gray ramp. Press USR1 (D19) to step to the
  next curve; it holds still so you can photograph each one. USR2 (D15)
  goes back one curve. The active curve is printed on screen and on Serial.
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

static uint16_t gray565(uint8_t v) { return ((v >> 3) << 11) | ((v >> 2) << 5) | (v >> 3); }

static void drawRamp(const char *label) {
  canvas.fillScreen(TFT_BLACK);
  for (int i = 0; i < 8; i++) {
    uint8_t v = i * 255 / 7;
    canvas.fillRect(0, 40 + i * 30, 172, 30, gray565(v));
  }
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.setCursor(4, 10);
  canvas.print(label);
  canvas.fillRect(0, 290, 57, 30, TFT_RED);
  canvas.fillRect(57, 290, 58, 30, TFT_GREEN);
  canvas.fillRect(115, 290, 57, 30, TFT_BLUE);
  canvas.pushSprite(0, 0);
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(800);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) {
    Serial.println(display.lastResult().message);
    return;
  }
  canvas.createSprite(display, 172, 320);
}

static void applyCurve(int idx) {
  // GAMSET parameter values that select gamma curves 1 to 4.
  static const uint8_t curves[] = {0x01, 0x02, 0x04, 0x08};
  // Reach the raw display bus to send a command the library does not wrap.
  IBus &bus = display.panel().driver().bus();
  bus.writeCommand(0x26);  // GAMSET
  bus.writeData(curves[idx]);
  char label[24];
  snprintf(label, sizeof(label), "gamma %02X (%d/4)", curves[idx], idx + 1);
  drawRamp(label);
  Serial.printf("GAMSET 0x%02X\n", curves[idx]);
}

void loop() {
  static int idx = -1;
  static bool p1 = false, p2 = false;
  if (idx < 0) {
    pinMode(D19, INPUT_PULLUP);
    pinMode(D15, INPUT_PULLUP);
    idx = 0;
    applyCurve(idx);
  }
  bool d1 = digitalRead(D19) == LOW, d2 = digitalRead(D15) == LOW;
  // Edge detection: act once per press, not while the button is held.
  if (d1 && !p1) { idx = (idx + 1) % 4; applyCurve(idx); delay(200); }
  if (d2 && !p2) { idx = (idx + 3) % 4; applyCurve(idx); delay(200); }
  p1 = d1; p2 = d2;
  delay(10);
}
