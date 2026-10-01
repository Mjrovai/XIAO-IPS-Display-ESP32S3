/*
  Test 04 - PDM microphone
  The on-board digital MEMS microphone is PDM: CLK=D0, DATA=D1.
  Captures 16 kHz mono 16-bit audio with the ESP-IDF 5 I2S PDM driver,
  and shows the level in dBFS: a big number, a bar with peak hold, and a
  scrolling history.
  Serial: "rms_dBFS,peak_dBFS,dc_offset" once per frame.
  Pass: in a quiet room the level sits steady (roughly -70 to -50 dBFS);
  speech, a clap, or a tap on the board pushes it up by 20 dB or more.
*/

#include <Arduino.h>
#include <driver/i2s_pdm.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

Seeed_GFX display;
Seeed_Sprite canvas;

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

static constexpr gpio_num_t MIC_CLK = GPIO_NUM_1;   // D0
static constexpr gpio_num_t MIC_DATA = GPIO_NUM_2;  // D1
static constexpr uint32_t SAMPLE_RATE = 16000;
static constexpr size_t BLOCK = 1024;                // 64 ms per read

static i2s_chan_handle_t rx = nullptr;
static int16_t buf[BLOCK];

static constexpr int HIST_W = 172;
static float hist[HIST_W];
static float peakHold = -90.0f;
static uint32_t peakStamp = 0;

static bool micBegin() {
  i2s_chan_config_t ch = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  ch.dma_desc_num = 8;      // 8 x 256 frames = 128 ms of buffering, so a slow
  ch.dma_frame_num = 256;   // screen frame does not drop samples
  if (i2s_new_channel(&ch, nullptr, &rx) != ESP_OK) return false;

  i2s_pdm_rx_config_t cfg = {};
  cfg.clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
  cfg.slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  // PDM needs only two wires: a clock the ESP32 drives and a one-bit data stream back.
  cfg.gpio_cfg.clk = MIC_CLK;
  cfg.gpio_cfg.din = MIC_DATA;
  if (i2s_channel_init_pdm_rx_mode(rx, &cfg) != ESP_OK) return false;
  return i2s_channel_enable(rx) == ESP_OK;
}

// 16-bit full scale is 32768, so 0 dBFS is a full-scale signal.
static float toDb(float x) {
  if (x < 1.0f) x = 1.0f;
  return 20.0f * log10f(x / 32768.0f);
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
  for (int i = 0; i < HIST_W; i++) hist[i] = -90.0f;
  Serial.println(micBegin() ? "mic: PDM RX ok" : "mic: init FAILED");
}

static uint16_t levelColor(float db) {
  if (db > -20) return TFT_RED;
  if (db > -40) return TFT_YELLOW;
  return TFT_GREEN;
}

void loop() {
  size_t got = 0;
  // Blocks until 1024 samples (64 ms at 16 kHz) have arrived, which paces the loop.
  if (i2s_channel_read(rx, buf, sizeof(buf), &got, 500) != ESP_OK || got == 0) {
    Serial.println("mic: read timeout");
    return;
  }
  size_t n = got / sizeof(int16_t);

  // Remove the DC offset before measuring level.
  float mean = 0;
  for (size_t i = 0; i < n; i++) mean += buf[i];
  mean /= n;
  float sq = 0;
  int peak = 0;
  for (size_t i = 0; i < n; i++) {
    float v = buf[i] - mean;
    sq += v * v;
    if (fabsf(v) > peak) peak = (int)fabsf(v);
  }
  float rmsDb = toDb(sqrtf(sq / n));
  float peakDb = toDb((float)peak);
  Serial.printf("%.1f,%.1f,%.0f\n", rmsDb, peakDb, mean);

  if (rmsDb > peakHold || millis() - peakStamp > 1500) { peakHold = rmsDb; peakStamp = millis(); }
  memmove(hist, hist + 1, sizeof(float) * (HIST_W - 1));
  hist[HIST_W - 1] = rmsDb;

  // Map -90..0 dBFS to 0..1
  auto frac = [](float db) { return constrain((db + 90.0f) / 90.0f, 0.0f, 1.0f); };

  canvas.fillScreen(TFT_BLACK);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.setTextSize(2);
  canvas.drawCentreString("MIC  PDM", 86, 6, 1);

  char t[24];
  snprintf(t, sizeof(t), "%.0f", rmsDb);
  canvas.setTextColor(levelColor(rmsDb), TFT_BLACK);
  canvas.setTextSize(5);
  canvas.drawCentreString(t, 86, 40, 1);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.drawCentreString("dBFS", 86, 88, 1);

  // Level bar with peak-hold marker
  canvas.drawRect(10, 120, 152, 22, TFT_DARKGREY);
  canvas.fillRect(12, 122, (int)(148 * frac(rmsDb)), 18, levelColor(rmsDb));
  int px = 12 + (int)(148 * frac(peakHold));
  canvas.drawFastVLine(px, 118, 26, TFT_WHITE);

  // Scrolling history, 100 px tall
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.setCursor(4, 158);
  canvas.print("history (-90 .. 0 dBFS)");
  for (int i = 0; i < HIST_W; i++) {
    int h = (int)(100 * frac(hist[i]));
    canvas.drawFastVLine(i, 270 - h, h, levelColor(hist[i]));
  }
  canvas.drawRect(0, 170, 172, 101, TFT_DARKGREY);

  snprintf(t, sizeof(t), "peak %.0f  dc %.0f", peakDb, mean);
  canvas.setCursor(4, 282);
  canvas.print(t);
  canvas.pushSprite(0, 0);
}
