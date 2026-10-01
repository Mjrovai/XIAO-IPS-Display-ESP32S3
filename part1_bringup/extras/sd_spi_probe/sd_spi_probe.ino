// Probe: is a display SPI transaction left open, blocking SD.begin?
#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"
#include "bus/Bus_SPI.h"

Seeed_GFX display;
SPIClass sdSpi(HSPI);
static constexpr int8_t LCD_RST_PIN = 13, LCD_BL_PIN = 12;

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(4000);
  pinMode(D6, OUTPUT); digitalWrite(D6, HIGH);
  display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>, Config_Seeed_1inch47_Touch_JD9853A>();
  display.fillScreen(TFT_BLUE);
  SPIClass *lcdSpi = static_cast<Bus_SPI &>(display.panel().driver().bus()).spiInstance();
  Serial.printf("reusing the display SPIClass at %p\n", (void *)lcdSpi);
  // The display only writes, so its SPI was created without MISO. Attach D9 for the card.
  Serial.printf("attach MISO D9 -> %d\n", spiAttachMISO(lcdSpi->bus(), D9));
  bool ok = SD.begin(D6, *lcdSpi, 4000000);
  Serial.printf("SD.begin -> %d\n", ok);
  if (ok) Serial.printf("type %d size %u MB\n", (int)SD.cardType(), (unsigned)(SD.cardSize() / (1024ULL * 1024ULL)));
  // The display must still work after the card traffic.
  display.fillScreen(TFT_RED);
  Serial.println("done: screen should be RED");
}
void loop() { delay(3000); Serial.println("alive"); }
