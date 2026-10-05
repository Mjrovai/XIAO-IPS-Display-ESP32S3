# Part 2: TinyML with the IMU and the microphone (Edge Impulse)

This part follows two chapters of the book
[TinyML Made Easy: Hands-On with Seeed Studio Devices](https://mjrovai.github.io/TinyML_Made_Easy_XIAO_ESP32S3_ebook/)
by Marcelo Rovai: *Motion Classification and Anomaly Detection* and *Keyword
Spotting (KWS)*. The book uses the XIAOML Kit; here the same labs run on the
XIAO IPS Display (ESP32-S3) boards. The vision chapters do not apply, because
these boards have no camera.

**Status: in progress.** The data logger for motion is written and tested. The
Edge Impulse Studio steps (training, testing, and building the library) have
not been run yet, so this page has no model results. They will be added when
they exist.

## What is different from the book

| | The book (XIAOML Kit) | These boards |
|---|---|---|
| IMU | LSM6DS3TR-C, I2C 0x6A | Same chip, same address |
| IMU library | Seeed Arduino LSM6DS3 | Same library |
| Data collection | Edge Impulse CLI data forwarder over USB | **CSV files on the microSD card**, recorded with no computer attached (the book lists this as an alternative) |
| Microphone | PDM, GPIO 41 (data) and 42 (clock) on the Sense board | PDM, **D0 (clock) and D1 (data)**, see Part 1, test 04 |
| Result display | An OLED | The 1.47" touch display |
| ESP32 Arduino core | 2.0.17 (its `I2S.h` is used for the microphone) | 3.3.12 (the IDF 5 PDM driver is used instead) |

One point where the book and the library disagree. The book says the IMU runs
at +/-2 g and +/-250 dps by default. In the installed Seeed Arduino LSM6DS3
library (2.0.7) the defaults are **+/-16 g and +/-2000 dps**. The book's code
still works, because the library scales the readings to g correctly, but the
resolution is eight times coarser, and the book's inference code clips at 2 g
anyway. The sketches here set +/-2 g and +/-245 dps (the library offers 245,
not 250) explicitly, in both the data logger and, later, the inference sketch,
so training and inference see the same scale.

## Motion classification

The book's four classes simulate how a container moves during transport:

- **maritime**: movement on all three axes, wave-like.
- **terrestrial**: mostly horizontal movement, with small vibrations.
- **lift**: mostly vertical movement, like a forklift: up, pause, down.
- **idle**: the board resting on a stable surface.

### Test 01: IMU data logger

Records 10-second samples at **exactly 50 Hz** and saves each one on the card
as a CSV file named after its class, such as `idle.003.csv`. No computer is
needed while recording, so the board can ride in a bag, a car, or a cart.

**How to use it**

1. Touch a class on the screen, or press USR2 to step through them.
2. Press USR1 or touch START. A 3-second countdown lets you set the board.
3. Move the board the way that class would move, for 10 seconds. The file saves
   itself. Press USR1 during a recording to abort it; the partial file is
   deleted.

**The CSV format:** `timestamp,accX,accY,accZ,gyrX,gyrY,gyrZ`, with the time in
milliseconds (0, 20, 40, and so on), acceleration in m/s2, and rotation in
degrees per second. The book's model uses the three acceleration columns; the
gyroscope columns are there for you to try.

**Measured.** One test recording with the board at rest gave 500 rows, no lost
samples, timestamps from 0 to 9980 ms in steps of exactly 20 ms, and a time
between samples that varied from 19.75 to 20.22 ms (about 1% around 20 ms).
The mean magnitude of the acceleration was 9.72 m/s2 (0.99 g), and the
gyroscope showed a bias of 1 to 2 degrees per second. We did not test it in
motion or outside the lab.

#### How it works

- **A separate task owns the clock.** The sampler runs as its own task and uses
  `vTaskDelayUntil`, which keeps a fixed period without drift. The main loop
  can then draw on the screen and write to the card without delaying a sample.
- **A queue carries the samples** from the sampler to the main loop, which
  formats them and writes to the card in blocks, with a flush every two seconds
  so little is lost if the card is pulled.
- **The I2C bus is shared.** The touch controller and the IMU use the same
  bus, so the touch controller is read only while idle, and the sampler is the
  only code that touches the bus while recording.
- **The card shares SPI with the display**, so it uses the display's own SPI
  object, as in Part 1, test 08.
- **Debug commands** on the serial port: `c` next class, `r` record, `l` list
  files, `p` print the last file, `x` delete the last file.

<!-- sketch: part2_tinyml/01_imu_logger/01_imu_logger.ino -->
**Sketch:** [`part2_tinyml/01_imu_logger/01_imu_logger.ino`](https://github.com/Mjrovai/XIAO-IPS-Display-ESP32S3/blob/main/part2_tinyml/01_imu_logger/01_imu_logger.ino)

```cpp
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
```
<!-- /sketch -->

### Next steps for motion (in the Edge Impulse Studio)

These follow the book's chapter. They need an Edge Impulse account, so they are
done by hand in the Studio, not by the sketches.

1. **Record data.** About 12 samples of 10 seconds for each of the four classes
   (around two minutes per class).
2. **Create a project** in the Studio, and upload the CSV files with
   *Data acquisition, Upload data*, inferring the label from the file name. The
   first upload needs the
   [CSV Wizard](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/csv-wizard)
   to say which column is the timestamp and which three columns are the sensor
   axes (accX, accY, accZ).
3. **Split** the data into training and test sets (the book keeps about 20%
   for testing).
4. **Impulse** (the book's settings): window 2000 ms, stride 200 ms, one
   *Spectral Analysis* block (the book reports 63 features for a 32-point FFT),
   a *Classifier* (a dense network with hidden layers of 20 and 10 neurons,
   learning rate 0.005, 30 epochs, 20% validation), and an *Anomaly detection*
   block (K-means with 32 clusters).
5. **Test and deploy** as an Arduino library, quantized (int8).
6. **Inference sketch** on the board, showing the class and the anomaly score on
   the display. It needs the exact library name that the Studio generates, so it
   is written after step 5.

## Keyword spotting

**Status: not started on the board side beyond planning.**

The book uses four classes, **yes**, **no**, **noise**, and **unknown**, from the
Edge Impulse keyword-spotting pre-built dataset (derived from Pete Warden's
Speech Commands), 1-second clips at 16 kHz and 16 bits. Its model uses MFCC
features and a small 1D convolutional network.

**A finding about the local copies.** The two copies of this dataset in the
author's Dropbox (`XIAOML-Kit/keywords` and `2021/.../keywords2`) have the right
folders and file names, but **every one of the 6,036 files is empty (0 bytes)**.
They cannot be used. The original archive,
`https://cdn.edgeimpulse.com/datasets/keywords2.zip`, is about 145.5 MB and must
be downloaded again.

**What changes on this board.** The microphone is on D0 and D1, not on GPIO 41
and 42, and the board runs core 3.3.12, so the book's `I2S.h` sketch does not
apply. The capture code reuses the IDF 5 PDM driver from Part 1, test 04.
