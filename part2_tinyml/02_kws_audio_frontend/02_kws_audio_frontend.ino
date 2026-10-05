/*
  Part 2, Test 02 - Audio front end for keyword spotting (no model yet)

  Captures the microphone continuously at 16 kHz and cuts it into 250 ms slices,
  the way Edge Impulse's continuous inference expects: a 1-second window made of
  4 slices that slides forward one slice at a time. There is no classifier here.
  The goal is to prove the part that differs from the book (the ESP32 core 3.x PDM
  driver instead of I2S.h) before a trained model exists, and to measure it:
  are the slices regular, and does the main loop keep up while it draws?

  Screen: the level of the last four slices (the sliding 1-second window, newest on
  the right), the current level in dBFS, and the capture statistics.
  Serial: one line per slice with the level, the slice interval, and the delay
  before the main loop picked the slice up.

  Where a model goes: process() receives each finished slice as 4000 int16
  samples. Edge Impulse's run_classifier_continuous() takes its place there.
*/

#include <Arduino.h>
#include <driver/i2s_pdm.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

static constexpr int SAMPLE_RATE = 16000;
static constexpr int WINDOW_SAMPLES = 16000;  // the model input: 1 second
static constexpr int SLICES_PER_WINDOW = 4;
static constexpr int SLICE_SAMPLES = WINDOW_SAMPLES / SLICES_PER_WINDOW;  // 4000 = 250 ms
static constexpr int CHUNK_SAMPLES = 500;                                 // one read = 31 ms

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;
static constexpr gpio_num_t MIC_CLK = GPIO_NUM_1;   // D0
static constexpr gpio_num_t MIC_DATA = GPIO_NUM_2;  // D1

Seeed_GFX display;
Seeed_Sprite canvas;
static i2s_chan_handle_t rx = nullptr;

// Two slice buffers: the capture task fills one while the main loop reads the other.
static int16_t sliceBuf[2][SLICE_SAMPLES];
static volatile int readyBuf = -1;             // index of a finished slice, or -1
static volatile uint32_t sliceDoneUs = 0;      // when that slice finished
static volatile uint32_t slicesCaptured = 0;
static volatile uint32_t overruns = 0;         // a slice finished before the last was read
static volatile uint32_t readErrors = 0;

static void captureTask(void *) {
  static int16_t chunk[CHUNK_SAMPLES];
  int cur = 0, fill = 0;
  // Discard the first half second: the PDM filter is still settling.
  size_t got = 0;
  for (int i = 0; i < 16; i++) i2s_channel_read(rx, chunk, sizeof(chunk), &got, 1000);
  for (;;) {
    if (i2s_channel_read(rx, chunk, sizeof(chunk), &got, 1000) != ESP_OK) { readErrors++; continue; }
    size_t n = got / sizeof(int16_t);
    for (size_t i = 0; i < n; i++) {
      sliceBuf[cur][fill++] = chunk[i];
      if (fill == SLICE_SAMPLES) {
        if (readyBuf != -1) overruns++;  // the main loop was too slow
        sliceDoneUs = micros();
        readyBuf = cur;
        slicesCaptured++;
        cur ^= 1;
        fill = 0;
      }
    }
  }
}

static bool micBegin() {
  i2s_chan_config_t ch = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  ch.dma_desc_num = 8;
  ch.dma_frame_num = 256;
  if (i2s_new_channel(&ch, nullptr, &rx) != ESP_OK) return false;
  i2s_pdm_rx_config_t cfg = {};
  cfg.clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
  cfg.slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  cfg.gpio_cfg.clk = MIC_CLK;
  cfg.gpio_cfg.din = MIC_DATA;
  if (i2s_channel_init_pdm_rx_mode(rx, &cfg) != ESP_OK) return false;
  return i2s_channel_enable(rx) == ESP_OK;
}

// ---- Slice processing ------------------------------------------------------------
static float levelDb[SLICES_PER_WINDOW] = {-90, -90, -90, -90};  // the sliding window
static uint32_t prevDoneUs = 0;
static uint32_t intervalMinUs = 0xFFFFFFFF, intervalMaxUs = 0, delayMaxUs = 0;
static uint32_t processed = 0;
static uint32_t firstDoneUs = 0;

static float sliceDb(const int16_t *s, int n) {
  float mean = 0, sq = 0;
  for (int i = 0; i < n; i++) mean += s[i];
  mean /= n;
  for (int i = 0; i < n; i++) { float v = s[i] - mean; sq += v * v; }
  float rms = sqrtf(sq / n);
  if (rms < 1.0f) rms = 1.0f;
  return 20.0f * log10f(rms / 32768.0f);
}

static uint16_t colorFor(float db) { return db > -20 ? TFT_RED : (db > -40 ? TFT_YELLOW : TFT_GREEN); }

static void draw() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("KWS FRONT END", 86, 6, 1);
  char t[40];
  float now = levelDb[SLICES_PER_WINDOW - 1];
  canvas.setTextColor(colorFor(now), TFT_BLACK);
  canvas.setTextSize(5);
  snprintf(t, sizeof(t), "%.0f", now);
  canvas.drawCentreString(t, 86, 34, 1);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.drawCentreString("dBFS", 86, 82, 1);

  // The sliding 1-second window: four slices, oldest on the left.
  canvas.setTextSize(1);
  canvas.drawCentreString("last 4 slices (1 s window)", 86, 112, 1);
  for (int i = 0; i < SLICES_PER_WINDOW; i++) {
    int h = (int)(100.0f * constrain((levelDb[i] + 90.0f) / 90.0f, 0.0f, 1.0f));
    int x = 8 + i * 40;
    canvas.drawRect(x, 128, 34, 104, TFT_DARKGREY);
    canvas.fillRect(x + 1, 231 - h, 32, h, colorFor(levelDb[i]));
  }

  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  snprintf(t, sizeof(t), "slices %lu  overruns %lu", (unsigned long)slicesCaptured, (unsigned long)overruns);
  canvas.setCursor(6, 248);
  canvas.print(t);
  if (processed > 1) {
    snprintf(t, sizeof(t), "interval %.1f..%.1f ms", intervalMinUs / 1000.0f, intervalMaxUs / 1000.0f);
    canvas.setCursor(6, 262);
    canvas.print(t);
    snprintf(t, sizeof(t), "max pickup delay %.1f ms", delayMaxUs / 1000.0f);
    canvas.setCursor(6, 276);
    canvas.print(t);
  }
  snprintf(t, sizeof(t), "read errors %lu  heap %luK", (unsigned long)readErrors, (unsigned long)(ESP.getFreeHeap() / 1024));
  canvas.setCursor(6, 290);
  canvas.print(t);
  canvas.pushSprite(0, 0);
}

// A model would run here, on a finished slice of SLICE_SAMPLES int16 samples.
static void process(const int16_t *slice, uint32_t doneUs) {
  uint32_t pickupUs = micros();
  float db = sliceDb(slice, SLICE_SAMPLES);
  for (int i = 0; i < SLICES_PER_WINDOW - 1; i++) levelDb[i] = levelDb[i + 1];
  levelDb[SLICES_PER_WINDOW - 1] = db;
  processed++;
  if (!firstDoneUs) firstDoneUs = doneUs;
  uint32_t interval = prevDoneUs ? doneUs - prevDoneUs : 0;
  prevDoneUs = doneUs;
  uint32_t delay_us = pickupUs - doneUs;
  if (interval) {
    if (interval < intervalMinUs) intervalMinUs = interval;
    if (interval > intervalMaxUs) intervalMaxUs = interval;
  }
  if (delay_us > delayMaxUs) delayMaxUs = delay_us;
  if (processed % 40 == 0) {
    // The average over many slices tells the true audio rate; single intervals are
    // rounded to the 16 ms DMA blocks.
    float avgMs = (doneUs - firstDoneUs) / 1000.0f / (processed - 1);
    Serial.printf("SUMMARY slices=%lu overruns=%lu read_errors=%lu avg_interval=%.3f ms min=%.1f max=%.1f max_pickup=%.2f ms\n",
                  (unsigned long)processed, (unsigned long)overruns, (unsigned long)readErrors, avgMs,
                  intervalMinUs / 1000.0f, intervalMaxUs / 1000.0f, delayMaxUs / 1000.0f);
  }
  Serial.printf("slice %lu  %.1f dBFS  interval %lu us  pickup delay %lu us\n", (unsigned long)processed, db,
                (unsigned long)interval, (unsigned long)delay_us);
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
  canvas.createSprite(display, 172, 320);
  if (!micBegin()) { Serial.println("mic init FAILED"); return; }
  // The capture task lives on core 0, so drawing on core 1 cannot delay the audio.
  xTaskCreatePinnedToCore(captureTask, "audio", 6144, nullptr, configMAX_PRIORITIES - 2, nullptr, 0);
  Serial.printf("front end running: %d Hz, %d-sample slices (%d ms), window %d samples\n", SAMPLE_RATE, SLICE_SAMPLES,
                SLICE_SAMPLES * 1000 / SAMPLE_RATE, WINDOW_SAMPLES);
}

void loop() {
  if (readyBuf == -1) { delay(2); return; }
  int idx = readyBuf;
  uint32_t doneUs = sliceDoneUs;
  process(sliceBuf[idx], doneUs);
  readyBuf = -1;  // hand the buffer back to the capture task
  draw();
}
