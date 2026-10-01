/*
  Test 01 - Display
  XIAO 1.47" IPS Touch Display (ESP32-S3 Plus), JD9853A, 172x320.
  Draws color bars, a white border, and corner markers, then rotates
  through the four orientations every 3 seconds.
  Pass: all bars are the right colors (red, green, blue, white, black)
  and the border is fully visible on every edge.
*/

#include <Arduino.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

Seeed_GFX display;

// GPIO numbers for the display reset and backlight on the XIAO ESP32-S3 Plus.
static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

void drawPattern(uint8_t rotation) {
  // 0 and 2 are portrait (172x320); 1 and 3 are landscape (320x172).
  display.setRotation(rotation);
  const int w = display.width();
  const int h = display.height();

  display.fillScreen(TFT_BLACK);
  const uint16_t colors[] = {TFT_RED, TFT_GREEN, TFT_BLUE, TFT_WHITE};
  // Sizes are fractions of the screen, so the layout works in every rotation.
  const int barH = h / 8;
  for (int i = 0; i < 4; i++) {
    display.fillRect(0, barH * (1 + i), w, barH, colors[i]);
  }
  // A frame on the very edge shows at once if any edge is clipped.
  display.drawRect(0, 0, w, h, TFT_WHITE);
  display.fillRect(0, 0, 10, 10, TFT_YELLOW);          // top-left
  display.fillRect(w - 10, h - 10, 10, 10, TFT_CYAN);  // bottom-right

  display.setTextColor(TFT_WHITE, TFT_BLACK);
  display.setTextSize(2);
  // Without this, a string that is too long wraps onto the next line.
  display.setTextWrap(false);
  char line[24];
  snprintf(line, sizeof(line), "%dx%d", w, h);
  display.setCursor(14, barH * 6);
  display.print(line);
  snprintf(line, sizeof(line), "rot %d", rotation);
  display.setCursor(14, barH * 6 + 22);
  display.print(line);
  Serial.printf("%dx%d rot %d\n", w, h, rotation);
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);  // never block when no host is reading the USB port
  // begin<Board, Config>() takes two types. The Board names the pins the LCD uses;
  // the Config names the panel (JD9853A, 172x320) and its color order.
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) {
    Serial.println(display.lastResult().message);
    return;
  }
}

void loop() {
  for (uint8_t r = 0; r < 4; r++) {
    drawPattern(r);
    delay(3000);
  }
}
