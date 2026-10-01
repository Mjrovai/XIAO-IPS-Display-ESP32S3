/*
  Test 08 - microSD card
  The card shares SCK=D8 / MOSI=D10 / MISO=D9 with the display; its chip
  select is D6 and the display's is D2. The display is started first, and
  the card is mounted through the display's own SPIClass, with MISO (D9)
  attached to it. Do NOT create a second SPIClass(HSPI): with arduino-esp32
  3.3.x it stops the bus the display is using and SD.begin never returns.
  Steps: mount, list the root folder, write 64 KB of a known pattern,
  read it back and compare byte by byte, report speeds, then redraw the
  screen to prove the display still works after card traffic.
  Pass: all four lines are PASS and the color bars at the bottom are clean.
*/

#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"
#include "bus/Bus_SPI.h"

Seeed_GFX display;
Seeed_Sprite canvas;

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;
static constexpr uint8_t SD_CS = D6;
static constexpr uint8_t SPI_MISO = D9;

static constexpr size_t TEST_BYTES = 64 * 1024;
static const char *TEST_PATH = "/xiao_test.bin";

struct Line { String text; uint16_t color; };
static Line lines[10];
static int nLines = 0;

static void say(const String &s, uint16_t color = TFT_WHITE) {
  Serial.println(s);
  if (nLines < 10) lines[nLines++] = {s, color};
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("microSD", 86, 6, 1);
  canvas.setTextSize(1);
  for (int i = 0; i < nLines; i++) {
    canvas.setTextColor(lines[i].color, TFT_BLACK);
    canvas.setCursor(4, 34 + i * 14);
    canvas.print(lines[i].text);
  }
  canvas.pushSprite(0, 0);
}

// A known pattern, so any mismatch on read-back means the data was corrupted.
static uint8_t patternByte(size_t i) { return (uint8_t)((i * 31 + (i >> 8)) & 0xFF); }

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(1500);

  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);  // keep the card deselected while the display starts
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) {
    Serial.println(display.lastResult().message);
    return;
  }
  canvas.createSprite(display, 172, 320);
  say("display ok", TFT_GREEN);

  // 1. mount
  SPIClass *sdSpi = static_cast<Bus_SPI &>(display.panel().driver().bus()).spiInstance();
  if (!sdSpi || !spiAttachMISO(sdSpi->bus(), SPI_MISO)) {
    say("1 mount: FAIL (SPI setup)", TFT_RED);
    return;
  }
  // Try the fastest SPI clock first, and fall back if the card does not mount.
  const uint32_t freqs[] = {20000000, 10000000, 4000000, 1000000, 400000};
  uint32_t usedFreq = 0;
  for (uint32_t f : freqs) {
    digitalWrite(SD_CS, HIGH);
    if (SD.begin(SD_CS, *sdSpi, f)) { usedFreq = f; break; }
  }
  if (!usedFreq) {
    say("1 mount: FAIL", TFT_RED);
    say("card inserted? FAT32?", TFT_YELLOW);
    return;
  }
  const char *type = SD.cardType() == CARD_SDHC ? "SDHC" : SD.cardType() == CARD_SD ? "SD" : SD.cardType() == CARD_MMC ? "MMC" : "?";
  say(String("1 mount: PASS ") + (usedFreq / 1000000) + " MHz", TFT_GREEN);
  say(String("  ") + type + "  " + (uint32_t)(SD.cardSize() / (1024ULL * 1024ULL)) + " MB", TFT_WHITE);

  // 2. list root
  File root = SD.open("/");
  int entries = 0;
  Serial.println("root folder:");
  for (File f = root.openNextFile(); f && entries < 50; f = root.openNextFile()) {
    Serial.printf("  %s%s  %u\n", f.name(), f.isDirectory() ? "/" : "", (unsigned)f.size());
    entries++;
  }
  root.close();
  say(String("2 list: PASS ") + entries + " items", TFT_GREEN);

  // 3. write
  static uint8_t buf[4096];
  SD.remove(TEST_PATH);
  File w = SD.open(TEST_PATH, FILE_WRITE);
  if (!w) { say("3 write: FAIL (open)", TFT_RED); return; }
  uint32_t t0 = millis();
  size_t done = 0;
  while (done < TEST_BYTES) {
    for (size_t i = 0; i < sizeof(buf); i++) buf[i] = patternByte(done + i);
    if (w.write(buf, sizeof(buf)) != sizeof(buf)) break;
    done += sizeof(buf);
  }
  w.close();
  uint32_t wms = millis() - t0;
  if (done != TEST_BYTES) { say("3 write: FAIL (short)", TFT_RED); return; }
  say(String("3 write: PASS ") + (int)((TEST_BYTES / 1024.0f) / (wms / 1000.0f)) + " KB/s", TFT_GREEN);

  // 4. read back and compare
  File r = SD.open(TEST_PATH);
  if (!r) { say("4 verify: FAIL (open)", TFT_RED); return; }
  size_t pos = 0, bad = 0;
  t0 = millis();
  while (pos < TEST_BYTES) {
    int got = r.read(buf, sizeof(buf));
    if (got <= 0) break;
    for (int i = 0; i < got; i++) if (buf[i] != patternByte(pos + i)) bad++;
    pos += got;
  }
  r.close();
  uint32_t rms = millis() - t0;
  SD.remove(TEST_PATH);
  if (pos != TEST_BYTES || bad) {
    say(String("4 verify: FAIL read=") + pos + " bad=" + bad, TFT_RED);
    return;
  }
  say(String("4 verify: PASS ") + (int)((TEST_BYTES / 1024.0f) / (rms / 1000.0f)) + " KB/s", TFT_GREEN);
  say("ALL PASS", TFT_GREEN);

  // Display still healthy after card traffic? Draw bars below the text.
  const uint16_t bars[] = {TFT_RED, TFT_GREEN, TFT_BLUE, TFT_WHITE};
  for (int i = 0; i < 4; i++) canvas.fillRect(i * 43, 250, 43, 70, bars[i]);
  canvas.pushSprite(0, 0);
}

void loop() {
  // Repeat the summary so a serial monitor opened late still sees the result.
  delay(3000);
  Serial.println("--- summary ---");
  for (int i = 0; i < nLines; i++) Serial.println(lines[i].text);
}
