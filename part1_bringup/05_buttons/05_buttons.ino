/*
  Test 05 - User buttons
  USR1 = D19, USR2 = D15, both active-low with the internal pull-up.
  Shows each button as a lit box while pressed, plus a press counter,
  and prints every change on Serial.
  Pass: each press lights only its own box and adds exactly 1 to its counter.
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

static constexpr uint8_t BTN_USR1 = D19;
static constexpr uint8_t BTN_USR2 = D15;

// Per-button state: debounced level, last raw level and when it changed, press count.
struct Button {
  uint8_t pin;
  const char *name;
  bool down = false;
  bool lastRaw = false;
  uint32_t changeMs = 0;
  uint32_t presses = 0;
};
Button btn[2] = {{BTN_USR1, "USR1"}, {BTN_USR2, "USR2"}};

// Draw everything into the sprite, then send it to the panel once.
static void draw() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("BUTTONS", 86, 8, 1);
  for (int i = 0; i < 2; i++) {
    int y = 50 + i * 120;
    uint16_t fill = btn[i].down ? TFT_GREEN : TFT_BLACK;
    canvas.fillRoundRect(16, y, 140, 100, 10, fill);
    canvas.drawRoundRect(16, y, 140, 100, 10, TFT_WHITE);
    canvas.setTextColor(btn[i].down ? TFT_BLACK : TFT_WHITE, fill);
    canvas.setTextSize(3);
    canvas.drawCentreString(btn[i].name, 86, y + 14, 1);
    char t[16];
    snprintf(t, sizeof(t), "x%lu", (unsigned long)btn[i].presses);
    canvas.setTextSize(4);
    canvas.drawCentreString(t, 86, y + 52, 1);
  }
  canvas.pushSprite(0, 0);
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(500);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) {
    Serial.println(display.lastResult().message);
    return;
  }
  canvas.createSprite(display, 172, 320);
  // A pressed button pulls its pin to ground, so enable the internal pull-up.
  for (auto &b : btn) pinMode(b.pin, INPUT_PULLUP);
  draw();
}

void loop() {
  bool dirty = false;
  for (auto &b : btn) {
    // LOW means pressed (active low).
    bool raw = digitalRead(b.pin) == LOW;
    // Restart the stability timer whenever the raw level changes.
    if (raw != b.lastRaw) { b.lastRaw = raw; b.changeMs = millis(); }
    // 20 ms debounce: accept the new state once it has been stable
    if (raw != b.down && millis() - b.changeMs > 20) {
      b.down = raw;
      if (raw) b.presses++;
      Serial.printf("%8lu ms  %s %s (count %lu)\n", (unsigned long)millis(), b.name, raw ? "DOWN" : "UP", (unsigned long)b.presses);
      dirty = true;
    }
  }
  if (dirty) draw();
  delay(2);
}
