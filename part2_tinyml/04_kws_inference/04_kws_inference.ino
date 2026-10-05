/*
  Part 2, Test 04 - Keyword spotting on the board (Edge Impulse model)

  Runs the "XIAO IPS Display - KWS" model (classes no, noise, unknown, yes) on the
  microphone, continuously. It uses the audio front end from test 02: the microphone
  is cut into 250 ms slices, and every slice runs the classifier on the last
  second of audio (Edge Impulse's continuous inference).

  Screen: the detected word, a bar for each class, the processing times, and the
  input level. Serial: one line per slice with the four scores.

  Important: this sketch must be built with ESP-NN turned off. The file build_opt.h next
  to it does that (-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=0). With ESP-NN on, the quantized
  network gives wrong answers on the ESP32-S3 with Arduino core 3.3.12 (it called almost
  everything "noise"), while the same model on a computer was right.

  The library name below is the one Edge Impulse generated for the project. If you
  train your own, change the include to your library's header.
*/

#include <Arduino.h>
#include <driver/i2s_pdm.h>
#include <XIAO_IPS_Display_-_KWS_inferencing.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

static constexpr int SLICE = EI_CLASSIFIER_SLICE_SIZE;  // 4000 samples = 250 ms
static constexpr int CHUNK_SAMPLES = 500;
static constexpr float DETECT_THRESHOLD = 0.80f;  // a first guess, not a tuned value
static constexpr uint32_t HOLD_MS = 1000;         // keep a detected word on screen

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;
static constexpr gpio_num_t MIC_CLK = GPIO_NUM_1;   // D0
static constexpr gpio_num_t MIC_DATA = GPIO_NUM_2;  // D1

Seeed_GFX display;
Seeed_Sprite canvas;
static i2s_chan_handle_t rx = nullptr;

// ---- Capture: two slice buffers, filled by a task on core 0 ---------------------
static int16_t sliceBuf[2][SLICE];
static volatile int readyBuf = -1;
static volatile uint32_t slicesCaptured = 0;
static volatile uint32_t overruns = 0;

static void captureTask(void *) {
  static int16_t chunk[CHUNK_SAMPLES];
  int cur = 0, fill = 0;
  size_t got = 0;
  for (int i = 0; i < 16; i++) i2s_channel_read(rx, chunk, sizeof(chunk), &got, 1000);  // filter settles
  for (;;) {
    if (i2s_channel_read(rx, chunk, sizeof(chunk), &got, 1000) != ESP_OK) continue;
    size_t n = got / sizeof(int16_t);
    for (size_t i = 0; i < n; i++) {
      sliceBuf[cur][fill++] = chunk[i];
      if (fill == SLICE) {
        if (readyBuf != -1) overruns++;
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
  cfg.clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(EI_CLASSIFIER_FREQUENCY);
  cfg.slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  cfg.gpio_cfg.clk = MIC_CLK;
  cfg.gpio_cfg.din = MIC_DATA;
  if (i2s_channel_init_pdm_rx_mode(rx, &cfg) != ESP_OK) return false;
  return i2s_channel_enable(rx) == ESP_OK;
}

// ---- Classifier ----------------------------------------------------------------------
static const int16_t *currentSlice = nullptr;  // the slice being classified
// Edge Impulse asks for the slice's samples as floats in the int16 range.
static int sliceGetData(size_t offset, size_t length, float *out) {
  numpy::int16_to_float(&currentSlice[offset], out, length);
  return 0;
}

static ei_impulse_result_t result;
static int warmup = EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW;  // a full window is needed first
static float score[EI_CLASSIFIER_LABEL_COUNT];
static int topIndex = -1;
static uint32_t dspMs = 0, nnMs = 0, processed = 0;
static float levelDb = -90.0f;
static char heldWord[16] = "";
static uint32_t heldSince = 0;

static float sliceDb(const int16_t *s, int n) {
  float mean = 0, sq = 0;
  for (int i = 0; i < n; i++) mean += s[i];
  mean /= n;
  for (int i = 0; i < n; i++) { float v = s[i] - mean; sq += v * v; }
  float rms = sqrtf(sq / n);
  return 20.0f * log10f(fmaxf(rms, 1.0f) / 32768.0f);
}

static void classify(const int16_t *slice) {
  currentSlice = slice;
  levelDb = sliceDb(slice, SLICE);
  signal_t signal;
  signal.total_length = SLICE;
  signal.get_data = &sliceGetData;
  EI_IMPULSE_ERROR err = run_classifier_continuous(&signal, &result, false);
  if (err != EI_IMPULSE_OK) { Serial.printf("classifier error %d\n", (int)err); return; }
  processed++;
  dspMs = result.timing.dsp;
  nnMs = result.timing.classification;
  if (warmup > 0) { warmup--; return; }
  topIndex = 0;
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    score[i] = result.classification[i].value;
    if (score[i] > score[topIndex]) topIndex = i;
  }
  const char *top = result.classification[topIndex].label;
  // A word is shown when yes or no wins with enough confidence.
  if (score[topIndex] >= DETECT_THRESHOLD && (strcmp(top, "yes") == 0 || strcmp(top, "no") == 0)) {
    snprintf(heldWord, sizeof(heldWord), "%s", top);
    heldSince = millis();
  }
  Serial.printf("slice %lu |", (unsigned long)processed);
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) Serial.printf(" %s %.2f", result.classification[i].label, score[i]);
  Serial.printf(" | dsp %lu ms nn %lu ms | level %.0f dBFS\n", (unsigned long)dspMs, (unsigned long)nnMs, levelDb);
}

// ---- Screen --------------------------------------------------------------------------------
static uint16_t colorOf(const char *label) {
  if (!strcmp(label, "yes")) return TFT_GREEN;
  if (!strcmp(label, "no")) return TFT_RED;
  if (!strcmp(label, "unknown")) return TFT_YELLOW;
  return TFT_CYAN;
}

static void draw() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("KEYWORD SPOTTING", 86, 4, 1);
  char t[40];
  bool held = heldWord[0] && millis() - heldSince < HOLD_MS;
  canvas.setTextSize(6);
  if (warmup > 0) {
    canvas.setTextSize(2);
    canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    canvas.drawCentreString("warming up", 86, 56, 1);
  } else if (held) {
    canvas.setTextColor(colorOf(heldWord), TFT_BLACK);
    for (char *c = heldWord; *c; c++) *c = toupper(*c);
    canvas.drawCentreString(heldWord, 86, 40, 1);
    for (char *c = heldWord; *c; c++) *c = tolower(*c);
  } else {
    canvas.setTextColor(TFT_DARKGREY, TFT_BLACK);
    canvas.drawCentreString("...", 86, 40, 1);
  }
  // One bar per class, in a fixed order.
  const char *order[] = {"yes", "no", "unknown", "noise"};
  canvas.setTextSize(2);
  for (int r = 0; r < 4; r++) {
    int y = 112 + r * 36;
    float v = 0;
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
      // The labels are only filled in once the classifier has run, so check warmup first.
      if (warmup == 0 && result.classification[i].label && !strcmp(result.classification[i].label, order[r])) v = score[i];
    canvas.setTextColor(colorOf(order[r]), TFT_BLACK);
    canvas.setCursor(4, y);
    canvas.print(order[r]);
    canvas.drawRect(88, y, 80, 16, TFT_DARKGREY);
    canvas.fillRect(89, y + 1, (int)(78 * v), 14, colorOf(order[r]));
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    snprintf(t, sizeof(t), "%.0f%%", v * 100);
    canvas.setCursor(92, y + 20);
    canvas.print(t);
    canvas.setTextSize(2);
  }
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  snprintf(t, sizeof(t), "dsp %lu ms  nn %lu ms", (unsigned long)dspMs, (unsigned long)nnMs);
  canvas.setCursor(4, 262);
  canvas.print(t);
  snprintf(t, sizeof(t), "level %.0f dBFS", levelDb);
  canvas.setCursor(4, 276);
  canvas.print(t);
  canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  snprintf(t, sizeof(t), "slices %lu  overruns %lu", (unsigned long)processed, (unsigned long)overruns);
  canvas.setCursor(4, 292);
  canvas.print(t);
  canvas.pushSprite(0, 0);
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(800);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) { Serial.println(display.lastResult().message); return; }
  canvas.createSprite(display, 172, 320);
  Serial.printf("model: %s, %d classes, window %d samples, slice %d samples\n", EI_CLASSIFIER_PROJECT_NAME,
                EI_CLASSIFIER_LABEL_COUNT, EI_CLASSIFIER_RAW_SAMPLE_COUNT, SLICE);
  run_classifier_init();
  if (!micBegin()) { Serial.println("mic init FAILED"); return; }
  xTaskCreatePinnedToCore(captureTask, "audio", 6144, nullptr, configMAX_PRIORITIES - 2, nullptr, 0);
  draw();
}

void loop() {
  if (readyBuf == -1) { delay(2); return; }
  int idx = readyBuf;
  classify(sliceBuf[idx]);
  readyBuf = -1;
  draw();
}
