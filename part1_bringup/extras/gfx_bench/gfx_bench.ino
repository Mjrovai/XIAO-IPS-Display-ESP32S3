/*
  Bench - times individual Seeed_GFX operations on the 1.47" display.
  Results go to Serial (microseconds per operation, averaged).
*/

#include <Arduino.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"
#include "touch/Touch_AXS5106L.h"

Seeed_GFX display;
Seeed_Sprite canvas;
static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;
Touch_AXS5106L touch(-1, D7, Wire, 172, 320);

template <typename F>
void bench(const char *name, int n, F fn) {
  uint32_t t0 = micros();
  for (int i = 0; i < n; i++) fn(i);
  uint32_t dt = micros() - t0;
  Serial.printf("%-28s %8lu us/op\n", name, (unsigned long)(dt / n));
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                Config_Seeed_1inch47_Touch_JD9853A>();
  display.attachTouch(touch, display.panel().driver().bus());
  display.setTextSize(2);
  display.setTextColor(TFT_WHITE, TFT_BLACK);
  display.setTextWrap(false);
  Serial.printf("CPU %u MHz, free heap %u, PSRAM %u\n", getCpuFrequencyMhz(),
                ESP.getFreeHeap(), ESP.getPsramSize());
}

void loop() {
  Serial.println("---- direct to panel ----");
  bench("fillScreen (172x320)", 5, [](int i) { display.fillScreen(i & 1 ? TFT_RED : TFT_BLUE); });
  bench("fillRect 50x50", 20, [](int i) { display.fillRect(10, 10, 50, 50, TFT_GREEN); });
  bench("fillCircle r=8", 20, [](int i) { display.fillCircle(80, 80, 8, TFT_RED); });
  bench("drawPixel x100", 5, [](int i) { for (int k = 0; k < 100; k++) display.drawPixel(k, 100, TFT_WHITE); });
  bench("print 12 chars size2", 10, [](int i) { display.setCursor(0, 150); display.print("Hello World!"); });
  bench("drawFastHLine 172", 50, [](int i) { display.drawFastHLine(0, 200, 172, TFT_YELLOW); });
  int32_t x, y;
  bench("getTouch (no touch)", 20, [&](int i) { display.getTouch(&x, &y); });

  Serial.println("---- sprite ----");
  static bool made = false;
  if (!made) { made = canvas.createSprite(display, 172, 320); Serial.printf("sprite ok=%d\n", made); }
  if (made) {
    bench("sprite fillScreen", 5, [](int i) { canvas.fillScreen(TFT_BLACK); });
    bench("sprite fillCircle r=8", 20, [](int i) { canvas.fillCircle(80, 80, 8, TFT_RED); });
    bench("sprite print 12 chars", 10, [](int i) { canvas.setCursor(0, 150); canvas.print("Hello World!"); });
    bench("sprite drawCircle r=70", 10, [](int i) { canvas.drawCircle(86, 120, 70, TFT_WHITE); });
    bench("pushSprite full", 5, [](int i) { canvas.pushSprite(0, 0); });
  }
  Serial.println();
  delay(3000);
}
