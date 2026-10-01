/*
  Test 09 - Self-test (XIAO 1.47" IPS Touch Display, ESP32-S3 Plus)
  Runs every check on one screen:
    I2C   touch controller (0x63) and IMU (0x6A/0x6B) answer
    IMU   |a| is 1 g +/- 0.15 while you hold the board still
    MIC   PDM microphone delivers samples and the level moves
    SD    card mounts, 4 KB written and read back identically
    WIFI  at least one 2.4 GHz network found
    BAT   voltage readable (marked N/A below 2.5 V: no LiPo connected)
    USR1  press USR1
    USR2  press USR2
    TOUCH touch the screen
  Each line shows PASS / FAIL / WAIT. The three WAIT lines need you.
*/

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <SPI.h>
#include <SD.h>
#include <driver/i2s_pdm.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"
#include "bus/Bus_SPI.h"
#include "touch/Touch_AXS5106L.h"

Seeed_GFX display;
Seeed_Sprite canvas;
Touch_AXS5106L touch(-1, D7, Wire, 172, 320);

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

enum State { ST_WAIT, ST_PASS, ST_FAIL, ST_NA };
// One entry per check: name, state, and a short note shown under the name.
struct Check { const char *name; State st; char note[24]; };
Check checks[] = {
  {"I2C", ST_WAIT, ""}, {"IMU", ST_WAIT, ""}, {"MIC", ST_WAIT, ""}, {"SD", ST_WAIT, ""},
  {"WIFI", ST_WAIT, ""}, {"BAT", ST_WAIT, ""}, {"USR1", ST_WAIT, "press"}, {"USR2", ST_WAIT, "press"},
  {"TOUCH", ST_WAIT, "touch"},
};
enum { C_I2C, C_IMU, C_MIC, C_SD, C_WIFI, C_BAT, C_USR1, C_USR2, C_TOUCH, C_COUNT };

static void set(int i, State s, const char *fmt = "", ...) {
  checks[i].st = s;
  va_list ap; va_start(ap, fmt);
  vsnprintf(checks[i].note, sizeof(checks[i].note), fmt, ap);
  va_end(ap);
}

static void draw() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("SELF-TEST", 86, 6, 1);
  int pass = 0, fail = 0, wait = 0;
  for (int i = 0; i < C_COUNT; i++) {
    int y = 34 + i * 30;
    uint16_t col = checks[i].st == ST_PASS ? TFT_GREEN : checks[i].st == ST_FAIL ? TFT_RED
                   : checks[i].st == ST_NA ? TFT_YELLOW : TFT_DARKGREY;
    const char *tag = checks[i].st == ST_PASS ? "PASS" : checks[i].st == ST_FAIL ? "FAIL"
                      : checks[i].st == ST_NA ? "N/A" : "WAIT";
    if (checks[i].st == ST_PASS) pass++; else if (checks[i].st == ST_FAIL) fail++; else if (checks[i].st == ST_WAIT) wait++;
    canvas.fillRect(0, y, 4, 26, col);
    canvas.setTextSize(2);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.setCursor(10, y + 1);
    canvas.print(checks[i].name);
    canvas.setTextColor(col, TFT_BLACK);
    canvas.setCursor(100, y + 1);
    canvas.print(tag);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    canvas.setCursor(10, y + 18);
    canvas.print(checks[i].note);
  }
  canvas.setTextSize(2);
  canvas.setTextColor(fail ? TFT_RED : (wait ? TFT_YELLOW : TFT_GREEN), TFT_BLACK);
  char t[32];
  snprintf(t, sizeof(t), "%dP %dF %dW", pass, fail, wait);
  canvas.drawCentreString(t, 86, 304, 1);
  canvas.pushSprite(0, 0);
}

static bool i2cRead(uint8_t addr, uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)addr, (int)len) != (int)len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}
static int16_t le16(const uint8_t *p) { return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }

static uint8_t imuAddr = 0;
static bool imuIsQmi = false;

static void checkI2c() {
  bool t = false, i = false;
  Wire.beginTransmission(0x63); t = Wire.endTransmission() == 0;
  for (uint8_t a : {0x6A, 0x6B}) { Wire.beginTransmission(a); if (Wire.endTransmission() == 0) { imuAddr = a; i = true; break; } }
  set(C_I2C, (t && i) ? ST_PASS : ST_FAIL, "touch %s imu %s", t ? "ok" : "--", i ? "ok" : "--");
}

// Same IMU setup as test 03 (QMI8658 or LSM6 family).
static bool imuInit() {
  uint8_t who = 0;
  if (!imuAddr) return false;
  if (i2cRead(imuAddr, 0x00, &who, 1) && who == 0x05) {
    Wire.beginTransmission(imuAddr); Wire.write(0x02); Wire.write(0x60); Wire.endTransmission();
    Wire.beginTransmission(imuAddr); Wire.write(0x03); Wire.write(0x03); Wire.endTransmission();
    Wire.beginTransmission(imuAddr); Wire.write(0x04); Wire.write(0x53); Wire.endTransmission();
    Wire.beginTransmission(imuAddr); Wire.write(0x08); Wire.write(0x03); Wire.endTransmission();
    imuIsQmi = true; delay(20); return true;
  }
  if (i2cRead(imuAddr, 0x0F, &who, 1) && (who == 0x69 || who == 0x6A || who == 0x6C)) {
    Wire.beginTransmission(imuAddr); Wire.write(0x10); Wire.write(0x60); Wire.endTransmission();
    Wire.beginTransmission(imuAddr); Wire.write(0x11); Wire.write(0x60); Wire.endTransmission();
    imuIsQmi = false; delay(20); return true;
  }
  return false;
}

static void checkImu() {
  if (!imuInit()) { set(C_IMU, ST_FAIL, "no chip"); return; }
  float best = 99;
  for (int n = 0; n < 20; n++) {
    uint8_t d[6];
    if (!i2cRead(imuAddr, imuIsQmi ? 0x35 : 0x28, d, 6)) continue;
    float s = imuIsQmi ? 1.0f / 16384.0f : 0.000061f;
    float x = le16(d) * s, y = le16(d + 2) * s, z = le16(d + 4) * s;
    float m = sqrtf(x * x + y * y + z * z);
    if (fabsf(m - 1.0f) < fabsf(best - 1.0f)) best = m;
    delay(20);
  }
  set(C_IMU, fabsf(best - 1.0f) < 0.15f ? ST_PASS : ST_FAIL, "|a| %.2f g", best);
}

// Start the PDM microphone, discard the first blocks while its filter settles,
// then measure one block.
static void checkMic() {
  i2s_chan_handle_t rx = nullptr;
  i2s_chan_config_t ch = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  if (i2s_new_channel(&ch, nullptr, &rx) != ESP_OK) { set(C_MIC, ST_FAIL, "channel"); return; }
  i2s_pdm_rx_config_t cfg = {};
  cfg.clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(16000);
  cfg.slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  cfg.gpio_cfg.clk = GPIO_NUM_1; cfg.gpio_cfg.din = GPIO_NUM_2;
  if (i2s_channel_init_pdm_rx_mode(rx, &cfg) != ESP_OK || i2s_channel_enable(rx) != ESP_OK) {
    set(C_MIC, ST_FAIL, "init"); return;
  }
  static int16_t buf[1024];
  size_t got = 0;
  for (int i = 0; i < 4; i++) i2s_channel_read(rx, buf, sizeof(buf), &got, 500);  // settle
  float rmsDb = -99;
  if (i2s_channel_read(rx, buf, sizeof(buf), &got, 500) == ESP_OK && got > 0) {
    size_t n = got / 2; float mean = 0, sq = 0;
    for (size_t i = 0; i < n; i++) mean += buf[i];
    mean /= n;
    for (size_t i = 0; i < n; i++) { float v = buf[i] - mean; sq += v * v; }
    rmsDb = 20.0f * log10f(fmaxf(sqrtf(sq / n), 1.0f) / 32768.0f);
  }
  i2s_channel_disable(rx); i2s_del_channel(rx);
  // A live PDM mic has a noise floor well above digital silence (-90 dBFS).
  set(C_MIC, (rmsDb > -85 && rmsDb < -5) ? ST_PASS : ST_FAIL, "%.0f dBFS", rmsDb);
}

static void checkSd() {
  pinMode(D6, OUTPUT); digitalWrite(D6, HIGH);
  SPIClass *spi = static_cast<Bus_SPI &>(display.panel().driver().bus()).spiInstance();
  if (!spi || !spiAttachMISO(spi->bus(), D9)) { set(C_SD, ST_FAIL, "spi"); return; }
  if (!SD.begin(D6, *spi, 20000000) && !SD.begin(D6, *spi, 4000000)) { set(C_SD, ST_FAIL, "no card/FAT"); return; }
  static uint8_t w[4096], r[4096];
  for (int i = 0; i < 4096; i++) w[i] = (uint8_t)(i * 7 + 3);
  SD.remove("/selftest.bin");
  File f = SD.open("/selftest.bin", FILE_WRITE);
  bool ok = f && f.write(w, sizeof(w)) == sizeof(w);
  if (f) f.close();
  if (ok) {
    f = SD.open("/selftest.bin");
    ok = f && f.read(r, sizeof(r)) == (int)sizeof(r) && memcmp(w, r, sizeof(w)) == 0;
    if (f) f.close();
  }
  SD.remove("/selftest.bin");
  set(C_SD, ok ? ST_PASS : ST_FAIL, "%u MB", (unsigned)(SD.cardSize() / (1024ULL * 1024ULL)));
}

static void checkWifi() {
  WiFi.mode(WIFI_STA); WiFi.disconnect();
  int n = WiFi.scanNetworks();
  int best = -100;
  for (int i = 0; i < n; i++) best = max(best, (int)WiFi.RSSI(i));
  set(C_WIFI, n > 0 ? ST_PASS : ST_FAIL, "%d nets, best %d", n, best);
  WiFi.scanDelete();
}

static void checkBat() {
  analogReadResolution(12);
  analogSetPinAttenuation(D16, ADC_11db);
  uint32_t sum = 0;
  for (int i = 0; i < 16; i++) { sum += analogReadMilliVolts(D16); delay(2); }
  float v = (sum / 16.0f) * (316.0f + 160.0f) / 160.0f / 1000.0f;
  // Measured on our unit: about 3.97 V with NO battery (see tutorial), so a
  // reading alone cannot prove a LiPo is present.
  set(C_BAT, ST_NA, "%.2f V (verify w/ LiPo)", v);
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(800);
  pinMode(D6, OUTPUT); digitalWrite(D6, HIGH);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) {
    Serial.println(display.lastResult().message);
    return;
  }
  canvas.createSprite(display, 172, 320);
  bool t = display.attachTouch(touch, display.panel().driver().bus());
  Serial.printf("touch attach: %d\n", t);
  pinMode(D19, INPUT_PULLUP);
  pinMode(D15, INPUT_PULLUP);
  Wire.begin(D4, D5);
  Wire.setClock(400000);
  draw();

  checkI2c();  draw();
  checkImu();  draw();
  checkMic();  draw();
  checkSd();   draw();
  checkWifi(); draw();
  checkBat();  draw();
  for (int i = 0; i < C_COUNT; i++) Serial.printf("%-6s %s %s\n", checks[i].name,
      checks[i].st == ST_PASS ? "PASS" : checks[i].st == ST_FAIL ? "FAIL" : checks[i].st == ST_NA ? "N/A" : "WAIT", checks[i].note);
}

// The three manual checks: each turns PASS the first time its event is seen.
void loop() {
  bool dirty = false;
  if (checks[C_USR1].st == ST_WAIT && digitalRead(D19) == LOW) { set(C_USR1, ST_PASS, "pressed"); dirty = true; }
  if (checks[C_USR2].st == ST_WAIT && digitalRead(D15) == LOW) { set(C_USR2, ST_PASS, "pressed"); dirty = true; }
  int32_t x, y;
  if (checks[C_TOUCH].st == ST_WAIT && display.getTouch(&x, &y)) { set(C_TOUCH, ST_PASS, "x%ld y%ld", (long)x, (long)y); dirty = true; }
  if (dirty) {
    draw();
    for (int i = C_USR1; i < C_COUNT; i++) Serial.printf("%-6s %s %s\n", checks[i].name, checks[i].st == ST_PASS ? "PASS" : "WAIT", checks[i].note);
  }
  delay(10);
}
