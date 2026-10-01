/*
  Test 02 - Touch
  AXS5106L capacitive touch (I2C 0x63, INT=D7). Touch draws red dots and
  prints coordinates on Serial. Touch the four corner targets to verify
  the mapping. A tap on the CLEAR box (top center) wipes the canvas.
  Note: touch RST is shared with the LCD, so the touch object gets RST=-1.
*/

#include <Arduino.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"
#include "touch/Touch_AXS5106L.h"

Seeed_GFX display;

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

// Arguments: reset pin (-1, because the display's begin() already resets the
// shared line), interrupt pin D7, I2C bus, panel width and height.
Touch_AXS5106L touch(-1, D7, Wire, 172, 320);

// The CLEAR box is small and central so the four corners stay free to test.
static constexpr int CLEAR_X = 51, CLEAR_Y = 0, CLEAR_W = 70, CLEAR_H = 30;

void drawTarget(int cx, int cy) {
  display.drawCircle(cx, cy, 10, TFT_YELLOW);
  display.drawFastHLine(cx - 14, cy, 29, TFT_YELLOW);
  display.drawFastVLine(cx, cy - 14, 29, TFT_YELLOW);
}

void drawUi() {
  display.fillScreen(TFT_BLACK);
  display.fillRect(CLEAR_X, CLEAR_Y, CLEAR_W, CLEAR_H, TFT_DARKGREY);
  display.setTextColor(TFT_WHITE, TFT_DARKGREY);
  display.setTextSize(2);
  display.drawCentreString("CLEAR", 86, 7, 1);
  drawTarget(16, 16);
  drawTarget(156, 16);
  drawTarget(16, 304);
  drawTarget(156, 304);
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);  // never block when no host is reading the USB port
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) {
    Serial.println(display.lastResult().message);
    return;
  }
  // Attach the touch controller to the display so getTouch() reports screen pixels.
  if (!display.attachTouch(touch, display.panel().driver().bus())) {
    Serial.println(display.lastResult().message);
    return;
  }
  drawUi();
}

void loop() {
  int32_t x = 0, y = 0;
  // getTouch() is true while a finger is down; x and y are in screen pixels.
  if (display.getTouch(&x, &y)) {
    Serial.printf("touch x=%ld y=%ld\n", (long)x, (long)y);
    if (x >= CLEAR_X && x < CLEAR_X + CLEAR_W && y < CLEAR_H) {
      drawUi();
    } else {
      display.fillCircle(x, y, 4, TFT_RED);
    }
  }
}
