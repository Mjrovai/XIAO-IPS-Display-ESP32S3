/*
  Part 2, Test 06 - Motion classification on the board (Edge Impulse model)

  Runs the "XIAO IPS Display - Motion" model (classes idle, lift, maritime,
  terrestrial) on the IMU. The sampler from test 01 fills a ring buffer at exactly
  50 Hz; every 500 ms the last 2 seconds (100 samples of 6 axes) go to the
  classifier. The six axes are in the order the model was trained with: accX, accY,
  accZ in m/s2, then gyrX, gyrY, gyrZ in degrees per second, the same units the
  logger wrote to the CSV files.

  Screen: the winning class, a bar for each class, the anomaly score, and the
  processing times. Serial: one line per classification.

  Replay for checking: send the four bytes "WIND" followed by 600 float32 values
  (little endian, one window, six values per sample) and the board answers with one
  "RESULT" line instead of using the IMU. tools/motion_replay.py does this.

  Unlike test 04, this sketch needs no build_opt.h: the network is two small dense layers,
  and with ESP-NN on and off the board gave identical scores on 432 replayed windows.

  The library name below is the one Edge Impulse generated for the project. If you
  train your own, change the include to your library's header.
*/

#include <Arduino.h>
#include <Wire.h>
#include <LSM6DS3.h>
#include <XIAO_IPS_Display_-_Motion_inferencing.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

static constexpr int SAMPLE_HZ = EI_CLASSIFIER_FREQUENCY;               // 50
static constexpr int WINDOW = EI_CLASSIFIER_RAW_SAMPLE_COUNT;           // 100 samples
static constexpr int AXES = EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;        // 6
static constexpr uint32_t RUN_EVERY_MS = 500;

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

Seeed_GFX display;
Seeed_Sprite canvas;
LSM6DS3 imu(I2C_MODE, 0x6A);

// ---- Sampler: a separate task keeps the 50 Hz clock steady -----------------------
static float ring[WINDOW][AXES];
static volatile uint32_t head = 0;  // number of samples written so far
static portMUX_TYPE ringMux = portMUX_INITIALIZER_UNLOCKED;

static void samplerTask(void *) {
  TickType_t lastWake = xTaskGetTickCount();
  const float G = 9.80665f;
  for (;;) {
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(1000 / SAMPLE_HZ));
    float s[AXES] = {imu.readFloatAccelX() * G, imu.readFloatAccelY() * G, imu.readFloatAccelZ() * G,
                     imu.readFloatGyroX(),      imu.readFloatGyroY(),      imu.readFloatGyroZ()};
    portENTER_CRITICAL(&ringMux);
    memcpy(ring[head % WINDOW], s, sizeof(s));
    head = head + 1;
    portEXIT_CRITICAL(&ringMux);
  }
}

// Copy the last WINDOW samples, oldest first, into a flat buffer (interleaved axes).
static void snapshot(float *out) {
  portENTER_CRITICAL(&ringMux);
  uint32_t h = head;
  for (int i = 0; i < WINDOW; i++) memcpy(out + i * AXES, ring[(h + i) % WINDOW], AXES * sizeof(float));
  portEXIT_CRITICAL(&ringMux);
}

// ---- Classification ----------------------------------------------------------------
static float features[WINDOW * AXES];
static ei_impulse_result_t result;
static bool haveResult = false;
static uint32_t dspMs = 0, nnMs = 0, runs = 0;

static int getData(size_t offset, size_t length, float *out) {
  memcpy(out, features + offset, length * sizeof(float));
  return 0;
}

static bool classify() {
  signal_t signal;
  signal.total_length = WINDOW * AXES;
  signal.get_data = &getData;
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) { Serial.printf("classifier error %d\n", (int)err); return false; }
  dspMs = result.timing.dsp;
  nnMs = result.timing.classification;
  haveResult = true;
  return true;
}

static uint16_t colorOf(const char *c) {
  if (!strcmp(c, "idle")) return TFT_LIGHTGREY;
  if (!strcmp(c, "lift")) return TFT_GREEN;
  if (!strcmp(c, "maritime")) return TFT_CYAN;
  return TFT_ORANGE;  // terrestrial
}

// ---- Screen ------------------------------------------------------------------------
static void draw() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("MOTION", 86, 4, 1);
  char t[40];
  int best = 0;
  if (haveResult)
    for (int i = 1; i < EI_CLASSIFIER_LABEL_COUNT; i++)
      if (result.classification[i].value > result.classification[best].value) best = i;
  if (!haveResult) {
    canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    canvas.drawCentreString("filling", 86, 56, 1);
  } else {
    // "terrestrial" is 11 characters: size 2 fits, size 3 would not.
    canvas.setTextSize(2);
    canvas.setTextColor(colorOf(result.classification[best].label), TFT_BLACK);
    char name[16];
    strncpy(name, result.classification[best].label, sizeof(name) - 1);
    name[sizeof(name) - 1] = 0;
    for (char *c = name; *c; c++) *c = toupper(*c);
    canvas.drawCentreString(name, 86, 50, 1);
  }
  const char *order[] = {"idle", "lift", "maritime", "terrestrial"};
  for (int r = 0; r < 4; r++) {
    int y = 100 + r * 40;
    float v = 0;
    for (int i = 0; haveResult && i < EI_CLASSIFIER_LABEL_COUNT; i++)
      if (!strcmp(result.classification[i].label, order[r])) v = result.classification[i].value;
    canvas.setTextSize(1);
    canvas.setTextColor(colorOf(order[r]), TFT_BLACK);
    canvas.setCursor(4, y);
    canvas.print(order[r]);
    canvas.drawRect(4, y + 12, 164, 14, TFT_DARKGREY);
    canvas.fillRect(5, y + 13, (int)(162 * v), 12, colorOf(order[r]));
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    snprintf(t, sizeof(t), "%.0f%%", v * 100);
    canvas.setCursor(140, y);
    canvas.print(t);
  }
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  snprintf(t, sizeof(t), "anomaly %.2f", haveResult ? result.anomaly : 0.0f);
  canvas.setCursor(4, 268);
  canvas.print(t);
  snprintf(t, sizeof(t), "dsp %lu ms  nn %lu ms", (unsigned long)dspMs, (unsigned long)nnMs);
  canvas.setCursor(4, 282);
  canvas.print(t);
  canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  snprintf(t, sizeof(t), "runs %lu", (unsigned long)runs);
  canvas.setCursor(4, 296);
  canvas.print(t);
  canvas.pushSprite(0, 0);
}

// ---- Replay over USB serial ---------------------------------------------------------
static void handleReplay() {
  static uint8_t state = 0;  // how many bytes of "WIND" have matched
  static const char tag[] = "WIND";
  while (Serial.available()) {
    char c = Serial.read();
    if (c == tag[state]) { state++; } else { state = (c == tag[0]) ? 1 : 0; }
    if (state == 4) {
      state = 0;
      size_t need = sizeof(features), got = 0;
      uint32_t t0 = millis();
      uint8_t *p = (uint8_t *)features;
      while (got < need && millis() - t0 < 3000) {
        int n = Serial.readBytes(p + got, need - got);
        if (n > 0) got += n;
      }
      if (got < need) { Serial.println("REPLAY timeout"); return; }
      if (!classify()) return;
      Serial.print("RESULT");
      for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
        Serial.printf(" %s %.4f", result.classification[i].label, result.classification[i].value);
      Serial.printf(" anomaly %.3f dsp %lu nn %lu\n", result.anomaly, (unsigned long)dspMs, (unsigned long)nnMs);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  Serial.setRxBufferSize(4096);
  delay(800);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) { Serial.println(display.lastResult().message); return; }
  canvas.createSprite(display, 172, 320);
  Serial.printf("model: %s, %d classes, window %d samples x %d axes at %d Hz\n", EI_CLASSIFIER_PROJECT_NAME,
                EI_CLASSIFIER_LABEL_COUNT, WINDOW, AXES, SAMPLE_HZ);
  // Same ranges as the logger, so the model sees the units it was trained on.
  imu.settings.accelRange = 2;
  imu.settings.gyroRange = 245;
  if (imu.begin() != 0) { Serial.println("IMU init FAILED"); return; }
  run_classifier_init();
  xTaskCreatePinnedToCore(samplerTask, "imu", 4096, nullptr, configMAX_PRIORITIES - 2, nullptr, 0);
  draw();
}

void loop() {
  static uint32_t lastRun = 0;
  handleReplay();
  if (millis() - lastRun < RUN_EVERY_MS) { delay(2); return; }
  lastRun = millis();
  if (head < (uint32_t)WINDOW) { draw(); return; }  // wait for the first full window
  snapshot(features);
  if (classify()) {
    runs++;
    Serial.print("SCORES");
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
      Serial.printf(" %s %.2f", result.classification[i].label, result.classification[i].value);
    Serial.printf(" anomaly %.2f\n", result.anomaly);
  }
  draw();
}
