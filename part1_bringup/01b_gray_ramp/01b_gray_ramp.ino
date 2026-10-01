/*
  Test 01b - Gray ramp
  Draws 8 gray steps from black to white, built from raw RGB565 values
  where R, G and B are equal, plus the library's named grays.
  On a neutral panel every step looks gray. If the steps look bluish,
  the panel (or its gamma / inversion setting) has a color cast.
  Names are printed next to the library constants so you can compare.
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

// level 0..255 -> RGB565 with R = G = B
static uint16_t gray565(uint8_t v) {
  // RGB565 packs 5 bits of red, 6 of green, and 5 of blue into 16 bits.
  return ((v >> 3) << 11) | ((v >> 2) << 5) | (v >> 3);
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
  // A full-screen off-screen buffer (172 x 320 x 2 bytes = 110 KB). Everything is
  // drawn there and sent to the panel with one pushSprite().
  canvas.createSprite(display, 172, 320);
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.setCursor(4, 4);
  canvas.print("raw R=G=B ramp");

  // 8 steps of raw gray
  for (int i = 0; i < 8; i++) {
    // Eight steps: 0, 36, 72, 109, 145, 182, 218, 255.
    uint8_t v = i * 255 / 7;
    canvas.fillRect(0, 18 + i * 20, 172, 20, gray565(v));
    canvas.setTextColor(v > 110 ? TFT_BLACK : TFT_WHITE, gray565(v));
    char t[24];
    snprintf(t, sizeof(t), "%3d  0x%04X", v, gray565(v));
    canvas.setCursor(6, 24 + i * 20);
    canvas.print(t);
  }

  // Library named grays next to a true gray of the same 565 value
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.setCursor(4, 186);
  canvas.print("named constants");
  // Compare the library's named grays with a computed gray of the same average level.
  struct N { const char *name; uint16_t c; } names[] = {
    {"LIGHTGREY", TFT_LIGHTGREY}, {"DARKGREY", TFT_DARKGREY}, {"GREY", TFT_DARKGREY}};
  for (int i = 0; i < 2; i++) {
    uint16_t c = names[i].c;
    uint8_t r = (c >> 11) << 3, g = ((c >> 5) & 0x3F) << 2, b = (c & 0x1F) << 3;
    canvas.fillRect(0, 200 + i * 40, 86, 36, c);
    canvas.fillRect(86, 200 + i * 40, 86, 36, gray565((r + g + b) / 3));
    canvas.setTextColor(TFT_BLACK, c);
    canvas.setCursor(3, 208 + i * 40);
    canvas.print(names[i].name);
    canvas.setTextColor(TFT_BLACK, gray565((r + g + b) / 3));
    canvas.setCursor(90, 208 + i * 40);
    canvas.print("true gray");
    Serial.printf("%s = 0x%04X  (R=%d G=%d B=%d)\n", names[i].name, c, r, g, b);
  }
  // Pure primaries for reference
  canvas.fillRect(0, 286, 57, 34, TFT_RED);
  canvas.fillRect(57, 286, 58, 34, TFT_GREEN);
  canvas.fillRect(115, 286, 57, 34, TFT_BLUE);
  // One transfer to the panel.
  canvas.pushSprite(0, 0);
  Serial.println("ramp drawn");
}

void loop() { delay(3000); Serial.println("ramp drawn"); }
