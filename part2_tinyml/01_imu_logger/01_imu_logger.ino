/*
  Part 2, Test 01 - IMU data logger (CSV files on the microSD card)

  Records 10-second motion samples at exactly 50 Hz and saves each one as a CSV
  file on the card, named after its class: maritime.001.csv, idle.003.csv, and so
  on. No computer is needed while recording, so the board can ride in a bag, a
  car, or on a cart. Upload the files to Edge Impulse Studio afterwards (Data
  acquisition, Upload data, label inferred from the file name).

  Use it:
    1. Touch a class on the screen (or press USR2 to step through them).
    2. Press USR1 (or touch START). A 3-second countdown lets you set the board.
    3. Move the board as that class would move it for 10 seconds. It saves itself.
    Press USR1 during a recording to abort it (the partial file is deleted).

  CSV columns: timestamp (ms), accX, accY, accZ (m/s2), gyrX, gyrY, gyrZ (deg/s).
  The first three sensor columns match the accelerometer-only model in the book.

  Debug commands on the serial port: c = next class, r = record, l = list files,
  p = print the last file, x = delete the last file.
*/

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <LSM6DS3.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"
#include "bus/Bus_SPI.h"
#include "touch/Touch_AXS5106L.h"

// ---- Settings ---------------------------------------------------------------
static constexpr int SAMPLE_HZ = 50;
static constexpr int RECORD_SECONDS = 10;
static constexpr int TARGET_SAMPLES = SAMPLE_HZ * RECORD_SECONDS;
static constexpr int COUNTDOWN_SECONDS = 3;
static const char *LABELS[] = {"maritime", "terrestrial", "lift", "idle"};
static const char *TITLES[] = {"MARITIME", "TERRESTRIAL", "LIFT", "IDLE"};
static constexpr int NUM_LABELS = 4;

// ---- Hardware ---------------------------------------------------------------
static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;
static constexpr uint8_t SD_CS = D6;
static constexpr uint8_t SD_MISO = D9;
static constexpr uint8_t BTN_USR1 = D19;
static constexpr uint8_t BTN_USR2 = D15;

Seeed_GFX display;
Seeed_Sprite canvas;
Touch_AXS5106L touch(-1, D7, Wire, 172, 320);
LSM6DS3 imu(I2C_MODE, 0x6A);

// ---- Sampler: a separate task keeps the 50 Hz clock steady ---------------------
// It is the only code that talks to the IMU while recording, and the main loop
// does not touch the I2C bus (the touch controller shares it) until it stops.
struct Sample { float ax, ay, az, gx, gy, gz; };
static QueueHandle_t sampleQueue;
static volatile bool sampling = false;
static volatile bool samplerBusy = false;
static volatile uint32_t produced = 0;
static volatile uint32_t dropped = 0;
static volatile uint32_t periodMinUs = 0xFFFFFFFF, periodMaxUs = 0;

static void samplerTask(void *) {
  TickType_t lastWake = 0;
  bool started = false;
  uint32_t prevUs = 0;
  for (;;) {
    if (!sampling) { started = false; vTaskDelay(pdMS_TO_TICKS(2)); continue; }
    if (!started) { lastWake = xTaskGetTickCount(); started = true; prevUs = 0; }
    // vTaskDelayUntil keeps a fixed period with no drift, unlike delay().
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(1000 / SAMPLE_HZ));
    if (!sampling) continue;
    samplerBusy = true;
    Sample s;
    const float G = 9.80665f;  // the book converts g to m/s2 for Edge Impulse
    s.ax = imu.readFloatAccelX() * G;
    s.ay = imu.readFloatAccelY() * G;
    s.az = imu.readFloatAccelZ() * G;
    s.gx = imu.readFloatGyroX();
    s.gy = imu.readFloatGyroY();
    s.gz = imu.readFloatGyroZ();
    samplerBusy = false;
    uint32_t now = micros();
    if (prevUs) {
      uint32_t d = now - prevUs;
      if (d < periodMinUs) periodMinUs = d;
      if (d > periodMaxUs) periodMaxUs = d;
    }
    prevUs = now;
    if (xQueueSend(sampleQueue, &s, 0) != pdTRUE) dropped++;
    if (++produced >= (uint32_t)TARGET_SAMPLES) sampling = false;
  }
}

// ---- State ------------------------------------------------------------------
enum State { S_IDLE, S_COUNTDOWN, S_RECORD, S_DONE };
static State state = S_IDLE;
static int selected = 3;  // start on "idle"
static int nextIndex[NUM_LABELS];
static bool sdOk = false;
static uint32_t sdMegabytes = 0;
static uint32_t stateSince = 0;
static File outFile;
static char outName[40];
static char lastSaved[40] = "";
static uint32_t rowsWritten = 0;
static uint32_t lastFlushMs = 0;
static uint32_t lastDrawMs = 0;
static char lastSummary[80] = "";
static char lineBuf[4096];
static size_t lineLen = 0;

// ---- SD card ------------------------------------------------------------------
static bool initSd() {
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  // Reuse the display's own SPI object and attach MISO to it (see Part 1, test 08).
  SPIClass *spi = static_cast<Bus_SPI &>(display.panel().driver().bus()).spiInstance();
  if (!spi || !spiAttachMISO(spi->bus(), SD_MISO)) return false;
  const uint32_t clocks[] = {20000000, 10000000, 4000000};
  for (uint32_t f : clocks) {
    if (SD.begin(SD_CS, *spi, f)) {
      sdMegabytes = (uint32_t)(SD.cardSize() / (1024ULL * 1024ULL));
      return true;
    }
  }
  return false;
}

static void pathFor(int label, int index, char *out, size_t n) {
  snprintf(out, n, "/%s.%03d.csv", LABELS[label], index);
}

static void scanExisting() {
  for (int l = 0; l < NUM_LABELS; l++) {
    int i = 1;
    char path[40];
    for (;; i++) {
      pathFor(l, i, path, sizeof(path));
      if (!SD.exists(path)) break;
    }
    nextIndex[l] = i;  // first free number for this class
  }
}

// ---- Screen ------------------------------------------------------------------
static void drawButtonRow(int i) {
  int y = 30 + i * 54;
  bool sel = (i == selected);
  canvas.fillRoundRect(6, y, 160, 48, 8, sel ? TFT_GREEN : TFT_BLACK);
  canvas.drawRoundRect(6, y, 160, 48, 8, sel ? TFT_GREEN : TFT_WHITE);
  canvas.setTextSize(2);
  canvas.setTextColor(sel ? TFT_BLACK : TFT_WHITE, sel ? TFT_GREEN : TFT_BLACK);
  canvas.drawCentreString(TITLES[i], 86, y + 8, 1);
  char t[16];
  snprintf(t, sizeof(t), "%d saved", nextIndex[i] - 1);
  canvas.setTextSize(1);
  canvas.drawCentreString(t, 86, y + 32, 1);
}

static void draw() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.setTextSize(2);
  char t[48];
  if (!sdOk) {
    canvas.drawCentreString("IMU LOGGER", 86, 4, 1);
    canvas.setTextColor(TFT_RED, TFT_BLACK);
    canvas.drawCentreString("NO SD CARD", 86, 120, 1);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.drawCentreString("Insert a FAT card", 86, 150, 1);
    canvas.drawCentreString("and press RESET", 86, 164, 1);
    canvas.pushSprite(0, 0);
    return;
  }
  if (state == S_IDLE) {
    canvas.drawCentreString("IMU LOGGER", 86, 4, 1);
    for (int i = 0; i < NUM_LABELS; i++) drawButtonRow(i);
    canvas.fillRoundRect(6, 252, 160, 40, 8, TFT_BLUE);
    canvas.setTextColor(TFT_WHITE, TFT_BLUE);
    canvas.drawCentreString("START", 86, 262, 1);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    snprintf(t, sizeof(t), "USR1 start  USR2 next");
    canvas.drawCentreString(t, 86, 298, 1);
    snprintf(t, sizeof(t), "%d Hz, %d s, SD %lu MB", SAMPLE_HZ, RECORD_SECONDS, (unsigned long)sdMegabytes);
    canvas.drawCentreString(t, 86, 310, 1);
  } else if (state == S_COUNTDOWN) {
    int left = COUNTDOWN_SECONDS - (int)((millis() - stateSince) / 1000);
    canvas.drawCentreString(TITLES[selected], 86, 20, 1);
    canvas.setTextColor(TFT_YELLOW, TFT_BLACK);
    canvas.setTextSize(10);
    snprintf(t, sizeof(t), "%d", left < 0 ? 0 : left);
    canvas.drawCentreString(t, 86, 100, 1);
    canvas.setTextSize(2);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.drawCentreString("get ready", 86, 230, 1);
  } else if (state == S_RECORD) {
    float secs = (millis() - stateSince) / 1000.0f;
    int left = RECORD_SECONDS - (int)secs;
    canvas.setTextColor(TFT_RED, TFT_BLACK);
    canvas.drawCentreString("RECORDING", 86, 8, 1);
    canvas.setTextColor(TFT_CYAN, TFT_BLACK);
    canvas.drawCentreString(TITLES[selected], 86, 34, 1);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.setTextSize(8);
    snprintf(t, sizeof(t), "%d", left < 0 ? 0 : left);
    canvas.drawCentreString(t, 86, 90, 1);
    canvas.drawRect(10, 200, 152, 20, TFT_WHITE);
    int w = (int)(148.0f * produced / TARGET_SAMPLES);
    canvas.fillRect(12, 202, w, 16, TFT_RED);
    canvas.setTextSize(1);
    snprintf(t, sizeof(t), "%lu / %d samples", (unsigned long)produced, TARGET_SAMPLES);
    canvas.drawCentreString(t, 86, 230, 1);
    canvas.drawCentreString(outName + 1, 86, 250, 1);
    canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    canvas.drawCentreString("USR1 = abort", 86, 300, 1);
  } else {  // S_DONE
    canvas.setTextColor(TFT_GREEN, TFT_BLACK);
    canvas.drawCentreString("SAVED", 86, 30, 1);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.drawCentreString(lastSaved + 1, 86, 80, 1);
    canvas.drawCentreString(lastSummary, 86, 100, 1);
  }
  canvas.pushSprite(0, 0);
}

// ---- Recording ----------------------------------------------------------------
static void flushLines(bool force) {
  if (lineLen == 0) return;
  if (!force && lineLen < sizeof(lineBuf) - 256) return;
  outFile.write((const uint8_t *)lineBuf, lineLen);
  lineLen = 0;
}

static bool startRecording() {
  pathFor(selected, nextIndex[selected], outName, sizeof(outName));
  outFile = SD.open(outName, FILE_WRITE);
  if (!outFile) { Serial.printf("cannot create %s\n", outName); return false; }
  lineLen = snprintf(lineBuf, sizeof(lineBuf), "timestamp,accX,accY,accZ,gyrX,gyrY,gyrZ\n");
  rowsWritten = 0;
  xQueueReset(sampleQueue);
  produced = 0; dropped = 0; periodMinUs = 0xFFFFFFFF; periodMaxUs = 0;
  lastFlushMs = millis();
  stateSince = millis();
  sampling = true;  // the sampler task starts producing from here
  state = S_RECORD;
  Serial.printf("recording %s\n", outName);
  return true;
}

static void drainQueue() {
  Sample s;
  while (xQueueReceive(sampleQueue, &s, 0) == pdTRUE) {
    uint32_t ts = rowsWritten * (1000 / SAMPLE_HZ);  // exact 20 ms steps
    lineLen += snprintf(lineBuf + lineLen, sizeof(lineBuf) - lineLen, "%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
                        (unsigned long)ts, s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
    rowsWritten++;
    flushLines(false);
  }
  if (millis() - lastFlushMs > 2000) {  // push data to the card every 2 s
    flushLines(true);
    outFile.flush();
    lastFlushMs = millis();
  }
}

static void stopSampler() {
  sampling = false;
  while (samplerBusy) delay(1);  // let the sampler finish its last I2C read
}

static void finishRecording() {
  drainQueue();
  flushLines(true);
  outFile.close();
  snprintf(lastSaved, sizeof(lastSaved), "%s", outName);
  snprintf(lastSummary, sizeof(lastSummary), "%lu rows, %lu dropped", (unsigned long)rowsWritten, (unsigned long)dropped);
  Serial.printf("SAVED %s rows=%lu dropped=%lu period_us=%lu..%lu\n", outName, (unsigned long)rowsWritten,
                (unsigned long)dropped, (unsigned long)periodMinUs, (unsigned long)periodMaxUs);
  nextIndex[selected]++;
  state = S_DONE;
  stateSince = millis();
  draw();
}

static void abortRecording() {
  stopSampler();
  outFile.close();
  SD.remove(outName);
  Serial.printf("aborted, removed %s\n", outName);
  state = S_IDLE;
  draw();
}

static void startCountdown() {
  state = S_COUNTDOWN;
  stateSince = millis();
  draw();
}

static void printFile(const char *path) {
  File f = SD.open(path);
  if (!f) { Serial.printf("cannot open %s\n", path); return; }
  Serial.printf("--- %s (%u bytes)\n", path, (unsigned)f.size());
  while (f.available()) Serial.write(f.read());
  f.close();
  Serial.println("--- end");
}

// ---- Arduino ------------------------------------------------------------------
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
  display.attachTouch(touch, display.panel().driver().bus());

  // The book assumes +/-2 g and +/-250 dps, but the library defaults to +/-16 g
  // and +/-2000 dps. Set the ranges explicitly so training and inference match.
  imu.settings.accelRange = 2;
  imu.settings.gyroRange = 245;
  if (imu.begin() != 0) {
    Serial.println("IMU init FAILED");
  }
  pinMode(BTN_USR1, INPUT_PULLUP);
  pinMode(BTN_USR2, INPUT_PULLUP);

  sdOk = initSd();
  if (sdOk) scanExisting();
  Serial.printf("SD %s, %lu MB\n", sdOk ? "ok" : "MISSING", (unsigned long)sdMegabytes);

  sampleQueue = xQueueCreate(256, sizeof(Sample));
  xTaskCreatePinnedToCore(samplerTask, "imu", 4096, nullptr, configMAX_PRIORITIES - 2, nullptr, 0);
  draw();
}

void loop() {
  static bool d1 = false, d2 = false, touchWasDown = false;
  bool b1 = digitalRead(BTN_USR1) == LOW, b2 = digitalRead(BTN_USR2) == LOW;
  bool press1 = b1 && !d1, press2 = b2 && !d2;
  d1 = b1; d2 = b2;

  // Debug commands from the serial port.
  bool cmdRecord = false, cmdNext = false;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'r') cmdRecord = true;
    if (c == 'c') cmdNext = true;
    if (c == 'l') { File root = SD.open("/"); for (File f = root.openNextFile(); f; f = root.openNextFile()) Serial.printf("%s %u\n", f.name(), (unsigned)f.size()); }
    if (c == 'p' && lastSaved[0]) printFile(lastSaved);
    if (c == 'x' && lastSaved[0]) {
      SD.remove(lastSaved);
      Serial.printf("deleted %s\n", lastSaved);
      for (int l = 0; l < NUM_LABELS; l++) {  // re-scan so numbering stays contiguous
        char p2[40]; pathFor(l, nextIndex[l] - 1, p2, sizeof(p2));
        if (nextIndex[l] > 1 && !SD.exists(p2)) nextIndex[l]--;
      }
      lastSaved[0] = 0;
    }
  }
  if (!sdOk) { delay(100); return; }

  switch (state) {
    case S_IDLE: {
      if (press2 || cmdNext) { selected = (selected + 1) % NUM_LABELS; draw(); }
      // The touch controller is only read while idle, so it never shares the I2C
      // bus with the sampler.
      int32_t x, y;
      bool down = display.getTouch(&x, &y);
      if (down && !touchWasDown) {
        if (y >= 30 && y < 30 + NUM_LABELS * 54) { selected = (y - 30) / 54; draw(); }
        else if (y >= 252 && y < 292) startCountdown();
      }
      touchWasDown = down;
      if (press1 || cmdRecord) startCountdown();
      break;
    }
    case S_COUNTDOWN:
      if (press1) { state = S_IDLE; draw(); break; }
      if (millis() - lastDrawMs > 250) { lastDrawMs = millis(); draw(); }
      if (millis() - stateSince >= (uint32_t)COUNTDOWN_SECONDS * 1000) {
        if (!startRecording()) { state = S_IDLE; draw(); }
      }
      break;
    case S_RECORD:
      drainQueue();
      if (press1) { abortRecording(); break; }
      if (!sampling && uxQueueMessagesWaiting(sampleQueue) == 0 && produced >= (uint32_t)TARGET_SAMPLES) {
        stopSampler();
        finishRecording();
        break;
      }
      if (millis() - lastDrawMs > 250) { lastDrawMs = millis(); draw(); }
      break;
    case S_DONE:
      if (millis() - stateSince > 3000 || press1) { state = S_IDLE; touchWasDown = true; draw(); }
      break;
  }
  delay(2);
}
