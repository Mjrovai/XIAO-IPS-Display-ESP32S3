/*
  Test 07 - Wi-Fi scan
  Scans for 2.4 GHz networks every 6 seconds and lists the strongest ones
  with their RSSI (dBm), channel, and security type. No credentials needed.
  Pass: your own router shows up, and the RSSI gets stronger (closer to 0)
  when you bring the board nearer to it.
  If every RSSI is weaker than about -80 dBm next to a router you know is
  close, check that the antenna is attached and seated.
*/

#include <Arduino.h>
#include <WiFi.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

Seeed_GFX display;
Seeed_Sprite canvas;

// Set to 1 to show only the first 2 characters of each network name, so a
// photo or a pasted log does not expose your neighbors' networks.
#define MASK_SSIDS 1

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

static String shownName(int i) {
  String n = WiFi.SSID(i);
  if (n.length() == 0) return "(hidden)";
#if MASK_SSIDS
  return n.substring(0, 2) + "***";
#else
  return n;
#endif
}

static const char *authName(wifi_auth_mode_t a) {
  switch (a) {
    case WIFI_AUTH_OPEN: return "open";
    case WIFI_AUTH_WEP: return "WEP";
    case WIFI_AUTH_WPA_PSK: return "WPA";
    case WIFI_AUTH_WPA2_PSK: return "WPA2";
    case WIFI_AUTH_WPA_WPA2_PSK: return "WPA/2";
    case WIFI_AUTH_WPA3_PSK: return "WPA3";
    case WIFI_AUTH_WPA2_WPA3_PSK: return "WPA2/3";
    default: return "other";
  }
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
  // Station mode and not connected to anything: scanning needs no password.
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
}

void loop() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("WiFi scan", 86, 8, 1);
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.drawCentreString("scanning...", 86, 150, 1);
  canvas.pushSprite(0, 0);

  uint32_t t0 = millis();
  // A blocking scan of all 2.4 GHz channels; it takes about 2.5 to 3 seconds.
  int n = WiFi.scanNetworks();
  uint32_t dt = millis() - t0;
  Serial.printf("scan: %d networks in %lu ms\n", n, (unsigned long)dt);

  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("WiFi scan", 86, 6, 1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  char t[40];
  snprintf(t, sizeof(t), "%d nets  %.1f s", n, dt / 1000.0f);
  canvas.drawCentreString(t, 86, 30, 1);

  // WiFi.scanNetworks() returns the list sorted by RSSI, strongest first.
  // One row per network: signal (colored) and name at size 2, channel and
  // security at size 1 on the right. Row pitch 26 px, 10 rows fit.
  int shown = n < 10 ? n : 10;
  for (int i = 0; i < shown; i++) {
    int rssi = WiFi.RSSI(i);
    Serial.printf("%2d  %4d dBm  ch%-2d  %-6s  %s\n", i + 1, rssi, WiFi.channel(i),
                  authName(WiFi.encryptionType(i)), shownName(i).c_str());
    uint16_t col = rssi > -60 ? TFT_GREEN : (rssi > -75 ? TFT_YELLOW : TFT_RED);
    int y = 56 + i * 26;
    canvas.setTextSize(2);
    canvas.setTextColor(col, TFT_BLACK);
    snprintf(t, sizeof(t), "%d", rssi);
    canvas.setCursor(2, y);
    canvas.print(t);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    String name = shownName(i);
    if (name.length() > 6) name = name.substring(0, 6);
    canvas.setCursor(50, y);
    canvas.print(name);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    canvas.setCursor(126, y);
    snprintf(t, sizeof(t), "ch%d", WiFi.channel(i));
    canvas.print(t);
    canvas.setCursor(126, y + 9);
    canvas.print(authName(WiFi.encryptionType(i)));
  }
  canvas.pushSprite(0, 0);
  WiFi.scanDelete();
  delay(6000);
}
