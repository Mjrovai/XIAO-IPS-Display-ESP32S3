# Part 2: TinyML with the IMU and the microphone (Edge Impulse)

This part follows two chapters of the book
[TinyML Made Easy: Hands-On with Seeed Studio Devices](https://mjrovai.github.io/TinyML_Made_Easy_XIAO_ESP32S3_ebook/)
by Marcelo Rovai: *Motion Classification and Anomaly Detection* and *Keyword
Spotting (KWS)*. The book uses the XIAOML Kit; here the same labs run on the
XIAO IPS Display (ESP32-S3) boards. The vision chapters do not apply, because
these boards have no camera.

**Status: in progress.** Keyword spotting: a model was trained in the Edge Impulse
Studio and runs on the board, it classifies recorded clips correctly there, and a first
live test with a voice shows it detecting YES and NO. A measured accuracy per word is
still missing. Motion: the data logger is written and
tested; the recording and the Studio steps are still to do.

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
as a CSV file named after its class, such as `idle.003.csv`, in a folder of the
same name (`/idle/idle.003.csv`). No computer is
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
- **One folder per class,** because a small FAT16 card cannot hold hundreds of
  files in its root folder (see [A limit we hit](#a-limit-we-hit-the-root-folder-of-a-small-fat16-card)).
- **Debug commands** on the serial port: `c` next class, `r` record, `l` list
  files, `p` print the last file, `x` delete the last file.

<!-- sketch: part2_tinyml/01_imu_logger/01_imu_logger.ino -->
**Sketch:** [`part2_tinyml/01_imu_logger/01_imu_logger.ino`](https://github.com/Mjrovai/XIAO-IPS-Display-ESP32S3/blob/main/part2_tinyml/01_imu_logger/01_imu_logger.ino)

```cpp
/*
  Part 2, Test 01 - IMU data logger (CSV files on the microSD card)

  Records 10-second motion samples at exactly 50 Hz and saves each one as a CSV
  file on the card, named after its class and kept in a folder of the same name:
  /maritime/maritime.001.csv, /idle/idle.003.csv, and so on. No computer is needed while recording, so the board can ride in a bag, a
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
  snprintf(out, n, "/%s/%s.%03d.csv", LABELS[label], LABELS[label], index);
}

// One folder per class keeps the root folder of a small FAT16 card from filling up
// (it holds only about 512 entries, and a long name uses three of them).
static bool ensureDirs() {
  bool ok = true;
  for (int l = 0; l < NUM_LABELS; l++) {
    char d[24];
    snprintf(d, sizeof(d), "/%s", LABELS[l]);
    if (!SD.exists(d) && !SD.mkdir(d)) ok = false;
  }
  return ok;
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
    canvas.drawCentreString("Insert a FAT card, or", 86, 150, 1);
    canvas.drawCentreString("free space in its root", 86, 164, 1);
    canvas.drawCentreString("folder, then press RESET", 86, 178, 1);
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
  if (sdOk) {
    if (!ensureDirs()) { Serial.println("cannot create the class folders: root folder full?"); sdOk = false; }
    else scanExisting();
  }
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

**Status:** the model is trained and runs on the board. Replaying recorded clips on
the device gives the same answers as the same model on a computer (tests 04 and 05).
A first live test with a voice, on the board's own microphone, now works (see
[the live test](#the-live-test-with-a-voice)). Finding why the first attempt failed took a detour through a bug in
the optimized neural-network kernels (see [A problem we found](#a-problem-we-found-the-esp-nn-kernels-give-wrong-answers)).

The book uses four classes, **yes**, **no**, **noise**, and **unknown**, from the
Edge Impulse keyword-spotting pre-built dataset (derived from Pete Warden's
Speech Commands), 1-second clips at 16 kHz and 16 bits. Its model uses MFCC
features and a small 1D convolutional network.

**The dataset.** The two copies of this dataset in the author's Dropbox
(`XIAOML-Kit/keywords` and `2021/.../keywords2`) had the right folders and file
names, but every one of the 6,036 files was empty (0 bytes), so we downloaded the
original archive again from `https://cdn.edgeimpulse.com/datasets/keywords2.zip`.
It is 145,511,868 bytes, with SHA-256
`ab2375ceb0e5add85e47419bd2edf2b8d4378fc5513079044eef73850a39a918`. We keep it
outside the repository and outside Dropbox, in `~/datasets/edge-impulse/`.

`tools/kws_dataset_check.py` checks it. Result: the zip is intact, and there are
6,036 files, none empty or unreadable: **yes 1,500, no 1,500, unknown 1,500, and
noise 1,536**. Every file is 16 kHz, 16-bit, mono, and exactly 1.00 s, which is
the format the book needs. The level of a random sample of 300 files per class,
as RMS in dBFS over each 1-second clip:

| Class | 10th percentile | Median | 90th percentile | Clipped files (of 300) |
|---|---|---|---|---|
| yes | -35.9 | -25.2 | -18.6 | 16 |
| no | -33.3 | -24.4 | -18.0 | 9 |
| unknown | -35.1 | -24.3 | -17.1 | 14 |
| noise | -37.3 | -27.4 | -15.4 | 9 |

**A question that was open, and is now answered.** These clips are fairly loud, and
the speech this board records is quieter (see test 03). Whether that hurts the model
turned out not to matter: with this model, the answers do not change when the signal
is made 12 dB louder or 36 dB quieter (test 05).

The files are derived from Pete Warden's Speech Commands dataset
([arXiv:1804.03209](https://arxiv.org/abs/1804.03209)). They are not in this
repository; check the dataset's license before you share them.

### Test 02: audio front end (no model yet)

Captures the microphone continuously at 16 kHz and cuts it into 250 ms slices,
the way Edge Impulse's continuous inference works: a 1-second window made of
four slices, sliding forward one slice at a time. There is no classifier. This
test proves the part that differs from the book, the ESP32 core 3.x PDM driver
in place of the old `I2S.h`, before a trained model exists.

**Measured.** In a run of 400 slices (about 100 seconds) there were no overruns
and no read errors. Single intervals between slices alternate between 240 ms and
256 ms, because the driver hands over audio in 16 ms blocks, but the average over
400 slices was 250.03 ms, so no audio is lost. The main loop picked up each
slice within 1.6 ms of its completion while it also redrew the screen. We did
not test this sketch with a model; test 04 does, and it takes about 32 ms per
slice. The book says the KWS sketch needs PSRAM enabled; this model runs with PSRAM
disabled (see [the model](#the-model)).

#### How it works

- **Two buffers.** A capture task fills one 250 ms slice while the main loop
  reads the other. If a slice finishes before the last one was read, an
  *overrun* is counted. That counter stayed at zero.
- **The task runs on core 0**, so drawing on core 1 cannot delay the audio.
- **The first half second is discarded**, because the microphone filter is still
  settling (see Part 1, test 04).
- **The model hook is `process()`**: it receives each finished slice as 4000
  samples. Edge Impulse's `run_classifier_continuous()` goes there.

<!-- sketch: part2_tinyml/02_kws_audio_frontend/02_kws_audio_frontend.ino -->
**Sketch:** [`part2_tinyml/02_kws_audio_frontend/02_kws_audio_frontend.ino`](https://github.com/Mjrovai/XIAO-IPS-Display-ESP32S3/blob/main/part2_tinyml/02_kws_audio_frontend/02_kws_audio_frontend.ino)

```cpp
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
```
<!-- /sketch -->

### Test 03: record your own keywords

The dataset comes from many voices and many microphones. This board has one
specific microphone, and you have one specific pronunciation of "yes" and "no".
So it is worth recording your own clips, for two reasons: to add them to the
training data, and, more important, to keep some of them as a **test set**, so
the model is tested on what this board actually hears. The book also recommends
collecting words spoken by yourself.

The recorder listens to the microphone and, each time it hears a word, saves a
1-second clip with the word inside (300 ms before it starts, 700 ms after), as a
16 kHz, 16-bit, mono WAV named like the dataset: `yes.own.001.wav`,
`no.own.014.wav`, and so on, in a folder per class (`/yes`, `/no`, `/unknown`,
`/noise`). You say a word, wait for "SAVED", and say it again. No computer is
needed.

**How to use it**

1. Touch a class: YES, NO, UNKNOWN (any other word), or NOISE. For NOISE the
   board does not wait for a word; it saves a clip every 1.5 seconds, so you can
   let it record the background of your room.
2. Press USR1 to start listening, and again to stop.
3. Press USR2 to delete the last clip if it is bad (cut off, or a cough). Press
   it again to delete the one before.
4. If words are missed, or noise triggers it, change the margin with the `-` and
   `+` buttons. The screen shows the background level (blue line) and the trigger
   level (orange line).

The thumbnail shows the last clip's waveform, so you can check that the word is
centered, and the screen shows the clip's level in dBFS. The audio is saved as
captured, with no gain, so that level is the true level of this microphone.
Compare it with the dataset's clips, whose median is near -25 dBFS RMS.

**A suggested plan.** The numbers are our judgment, not a tested recipe: about 60
clips of YES, 60 of NO, 40 of other words for UNKNOWN, and a minute or two of
NOISE in the room where you will use the board. Keep about a third of the YES and
NO clips for the **test** set, and put the rest in training. Your clips will be a
small share of the training data next to the dataset's 1,500 per class, so do not
expect them to change the model much; their main value is the test.

**Measured.** The save path was tested with the NOISE class: six clips were saved,
each 32,044 bytes (a 44-byte header plus 32,000 bytes of audio) and each read back
from the card and compared with what was captured. **Word detection has not been
tested with a voice yet.** The thresholds (a two-frame trigger, a 15 dB margin
above the background) are first guesses.

![The recorder in two states: LISTENING with the previous clip yes.own.041.wav saved, and SAVED with yes.own.033.wav and its waveform](images/kws_recorder_states.jpg)

*The recorder while collecting YES clips. The waveform is the last clip, with the
word in the middle. In these photos the line under the level bar wrapped onto a
second line; the sketch now shortens it.*

**What a first session produced.** One session with a single speaker gave 166
clips: 50 YES, 50 NO, 50 UNKNOWN, and 16 NOISE (the recorder stopped saving after
the 16th NOISE clip, see the next section). The files are 16 kHz, 16-bit, mono,
exactly 1.00 s, and not empty. We reviewed them with `tools/kws_clip_review.py`.

| Class | Clips | RMS median (dBFS) | RMS 10th to 90th percentile | Peak median / 90th percentile |
|---|---|---|---|---|
| yes | 50 | -39.2 | -44.2 to -29.9 | -22.6 / -5.7 |
| no | 50 | -41.1 | -46.3 to -35.0 | -25.6 / -19.2 |
| unknown | 50 | -39.1 | -42.0 to -28.6 | -22.2 / -7.2 |
| noise | 16 | -64.9 | -70.9 to -31.5 | -42.1 / -20.2 |

- **The words are centered as designed.** The word starts at about 300 ms in the
  clip (median 300 for YES and NO, 320 for UNKNOWN).
- **This microphone is quieter than the dataset.** The medians for the three word
  classes are -39 to -41 dBFS, against -24 to -25 dBFS for the dataset's clips.
  That is **14 to 17 dB lower**, with the same measurement on both sides (the RMS of
  a whole 1-second clip). These recordings were made at one distance and one
  speaking volume, so the gap is one speaker's, not a property of the microphone.
  It does not matter for this model (test 05).
- **No gain is needed.** We had worried that raising the level by 14 dB would clip the
  louder clips (the 90th percentile of the peaks is already -5.7 dBFS for YES). Test 05
  shows the model is insensitive to the level, so the question does not arise.
- **Ten clips look suspicious** out of 150 word clips: `yes.own.003`, `022`,
  `038`, `039`, `047`; `no.own.026`; `unknown.own.001`, `022`, `036`, `046`. The
  reasons are a word that may be cut off, two separate bursts, or a very quiet
  clip (`yes.own.022` is at -62 dBFS). The review looks only at the signal. It
  cannot tell whether the word you said is the word in the label, so listen to
  these before you upload them.
- **The noise class was quiet and short.** The first 16 clips had a median of -65
  dBFS, against -27 for the dataset's noise clips. A second session added 50
  clips (see below).

#### How it works

- **A 2-second ring buffer** is filled by a capture task, as in test 02.
- **Frames of 20 ms** are analysed one by one. The background level is tracked
  slowly, only while it is quiet.
- **A word starts** when two frames in a row are more than the margin above the
  background. The clip is cut 300 ms before that point and 700 ms after it.
- **After a clip is saved,** the board waits for half a second of quiet before it
  listens again, so one word makes one clip.
- **Every clip is verified:** the file is written, read back, and compared with
  the audio in memory.
- **The save, delete, and card code** is the same as in the logger, including the
  shared SPI bus.

<!-- sketch: part2_tinyml/03_kws_recorder/03_kws_recorder.ino -->
**Sketch:** [`part2_tinyml/03_kws_recorder/03_kws_recorder.ino`](https://github.com/Mjrovai/XIAO-IPS-Display-ESP32S3/blob/main/part2_tinyml/03_kws_recorder/03_kws_recorder.ino)

```cpp
/*
  Part 2, Test 03 - Keyword recorder (your own voice, as WAV files on the microSD)

  Listens to the microphone and, every time it hears a word, saves a 1-second clip
  with the word in it (300 ms before the start, 700 ms after) as a 16 kHz, 16-bit,
  mono WAV named like the Edge Impulse dataset: yes.own.001.wav, no.own.014.wav,
  and so on. Say a word, wait for "SAVED", say it again. No computer is needed.
  Each class goes in its own folder (/yes, /no, /unknown, /noise): the root folder of
  a small FAT16 card holds only about 512 entries, and long names use three each, so
  roughly 165 files in the root fill it, even with the card nearly empty.

  Classes: YES, NO, UNKNOWN (any other word), NOISE. For NOISE the board does not
  wait for a word: it saves one clip every 1.5 seconds, so you can let it record
  the background of your room.

  Use it:
    1. Touch a class. Use the - and + buttons if words are missed or noise triggers it.
    2. Press USR1 to start listening, and again to stop.
    3. Press USR2 to delete the last clip if it is bad (cut off, or a cough). Press it
       again to delete the one before.
  The thumbnail shows the last clip's waveform, so you can see if the word is centered.

  Audio is saved as captured, with no gain, so the level on screen is the true level
  of this microphone. Edge Impulse's dataset clips have a median near -25 dBFS RMS.

  Debug commands on the serial port: s = start/stop, c = next class, x = delete the
  last clip (repeat to go further back), d<path> = delete one file, l = list files.
*/

#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <driver/i2s_pdm.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"
#include "bus/Bus_SPI.h"
#include "touch/Touch_AXS5106L.h"

static constexpr int SAMPLE_RATE = 16000;
static constexpr int CLIP_SAMPLES = 16000;       // 1 second, what the model expects
static constexpr int FRAME = 320;                // 20 ms analysis frames
static constexpr int PRE_FRAMES = 15;            // 300 ms before the word starts
static constexpr int POST_FRAMES = 35;           // 700 ms after
static constexpr int RING = 32000;               // 2 seconds of history
static constexpr int NOISE_PERIOD_FRAMES = 75;   // NOISE: one clip every 1.5 s

static const char *LABELS[] = {"yes", "no", "unknown", "noise"};
static const char *TITLES[] = {"YES", "NO", "UNKNOWN", "NOISE"};
static constexpr int NUM_CLASSES = 4;

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;
static constexpr gpio_num_t MIC_CLK = GPIO_NUM_1;   // D0
static constexpr gpio_num_t MIC_DATA = GPIO_NUM_2;  // D1
static constexpr uint8_t SD_CS = D6;
static constexpr uint8_t SD_MISO = D9;
static constexpr uint8_t BTN_USR1 = D19;
static constexpr uint8_t BTN_USR2 = D15;

Seeed_GFX display;
Seeed_Sprite canvas;
Touch_AXS5106L touch(-1, D7, Wire, 172, 320);
static i2s_chan_handle_t rx = nullptr;

// ---- Capture: a task fills a 2-second ring buffer ------------------------------
static int16_t ring[RING];
static volatile uint32_t written = 0;  // total samples captured since start

static void captureTask(void *) {
  static int16_t chunk[FRAME];
  size_t got = 0;
  for (int i = 0; i < 16; i++) i2s_channel_read(rx, chunk, sizeof(chunk), &got, 1000);  // let the filter settle
  for (;;) {
    if (i2s_channel_read(rx, chunk, sizeof(chunk), &got, 1000) != ESP_OK) continue;
    uint32_t w = written;
    size_t n = got / sizeof(int16_t);
    for (size_t i = 0; i < n; i++) ring[(w + i) % RING] = chunk[i];
    written = w + n;  // publish after the samples are in place
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

// ---- State -----------------------------------------------------------------------
enum Phase { P_STOPPED, P_ARMED, P_CAPTURING, P_COOLDOWN };
static Phase phase = P_STOPPED;
static int selected = 0;
static int nextIndex[NUM_CLASSES];
static bool sdOk = false;
static bool dirError = false;  // could not create the class folders (root folder full?)
static uint32_t sdMb = 0;

static uint32_t framesDone = 0;     // frames analysed so far
static float floorDb = -70.0f;      // running estimate of the background level
static float marginDb = 15.0f;      // how far above the floor counts as a word
static int hits = 0;                // consecutive frames above the trigger
static uint32_t onsetFrame = 0;
static uint32_t quietFrames = 0;
static uint32_t lastNoiseFrame = 0;
static float frameDbNow = -90.0f;

static char history[16][40];  // the last saved clips, newest at the end, so USR2 can undo several
static int historyCount = 0;
static char statusLine[48] = "stopped";
static float lastRmsDb = -90.0f, lastPeakDb = -90.0f;
static uint8_t thumb[160];
static bool haveThumb = false;
static int16_t clip[CLIP_SAMPLES];

// ---- SD card -------------------------------------------------------------------------
static bool initSd() {
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  SPIClass *spi = static_cast<Bus_SPI &>(display.panel().driver().bus()).spiInstance();
  if (!spi || !spiAttachMISO(spi->bus(), SD_MISO)) return false;
  const uint32_t clocks[] = {20000000, 10000000, 4000000};
  for (uint32_t f : clocks)
    if (SD.begin(SD_CS, *spi, f)) { sdMb = (uint32_t)(SD.cardSize() / (1024ULL * 1024ULL)); return true; }
  return false;
}

static void pathFor(int cls, int idx, char *out, size_t n) { snprintf(out, n, "/%s/%s.own.%03d.wav", LABELS[cls], LABELS[cls], idx); }

// One folder per class. Creating a folder needs a free entry in the root folder.
static void ensureDirs() {
  for (int c = 0; c < NUM_CLASSES; c++) {
    char d[24];
    snprintf(d, sizeof(d), "/%s", LABELS[c]);
    if (!SD.exists(d) && !SD.mkdir(d)) dirError = true;
  }
}

static void scanExisting() {
  for (int c = 0; c < NUM_CLASSES; c++) {
    int i = 1; char p[40];
    for (;; i++) { pathFor(c, i, p, sizeof(p)); if (!SD.exists(p)) break; }
    nextIndex[c] = i;
  }
}

// ---- WAV -------------------------------------------------------------------------------
static void putLE32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static void putLE16(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }

static void makeHeader(uint8_t *h, uint32_t dataBytes) {
  memcpy(h, "RIFF", 4); putLE32(h + 4, 36 + dataBytes); memcpy(h + 8, "WAVEfmt ", 8);
  putLE32(h + 16, 16); putLE16(h + 20, 1); putLE16(h + 22, 1);       // PCM, mono
  putLE32(h + 24, SAMPLE_RATE); putLE32(h + 28, SAMPLE_RATE * 2);    // rate, bytes per second
  putLE16(h + 32, 2); putLE16(h + 34, 16);                           // block align, bits
  memcpy(h + 36, "data", 4); putLE32(h + 40, dataBytes);
}

static float rmsDbOf(const int16_t *s, int n, float *peakDb) {
  float mean = 0; for (int i = 0; i < n; i++) mean += s[i]; mean /= n;
  float sq = 0; int peak = 0;
  for (int i = 0; i < n; i++) { float v = s[i] - mean; sq += v * v; if (abs(s[i]) > peak) peak = abs(s[i]); }
  float rms = sqrtf(sq / n); if (rms < 1) rms = 1;
  if (peakDb) *peakDb = 20.0f * log10f(fmaxf(peak, 1) / 32768.0f);
  return 20.0f * log10f(rms / 32768.0f);
}

// Write the clip, read it back, and check it. Returns true if the file is good.
static bool saveClip(const char *path) {
  uint8_t header[44];
  makeHeader(header, CLIP_SAMPLES * 2);
  File f = SD.open(path, FILE_WRITE);
  if (!f) return false;
  f.write(header, 44);
  f.write((const uint8_t *)clip, CLIP_SAMPLES * 2);
  f.close();
  f = SD.open(path);
  if (!f || f.size() != 44 + CLIP_SAMPLES * 2) { if (f) f.close(); return false; }
  uint8_t h2[44]; f.read(h2, 44);
  bool ok = memcmp(h2, header, 44) == 0;
  static int16_t back[CLIP_SAMPLES];
  f.read((uint8_t *)back, CLIP_SAMPLES * 2);
  f.close();
  return ok && memcmp(back, clip, CLIP_SAMPLES * 2) == 0;
}

static void makeThumb() {
  int per = CLIP_SAMPLES / 160;
  int peak = 1;
  for (int i = 0; i < CLIP_SAMPLES; i++) if (abs(clip[i]) > peak) peak = abs(clip[i]);
  for (int x = 0; x < 160; x++) {
    int m = 0;
    for (int i = 0; i < per; i++) { int v = abs(clip[x * per + i]); if (v > m) m = v; }
    thumb[x] = (uint8_t)((long)m * 24 / peak);  // 0..24 pixels above and below the middle
  }
  haveThumb = true;
}

// Copy the 1-second window that starts at startSample out of the ring, save it.
static void captureAndSave(uint32_t startSample) {
  for (int i = 0; i < CLIP_SAMPLES; i++) clip[i] = ring[(startSample + i) % RING];
  char path[40];
  pathFor(selected, nextIndex[selected], path, sizeof(path));
  lastRmsDb = rmsDbOf(clip, CLIP_SAMPLES, &lastPeakDb);
  bool ok = saveClip(path);
  if (ok) {
    if (historyCount == 16) { memmove(history[0], history[1], 15 * 40); historyCount = 15; }
    snprintf(history[historyCount++], 40, "%s", path);
    nextIndex[selected]++;
    makeThumb();
    snprintf(statusLine, sizeof(statusLine), "SAVED %s", path + 1);
  } else {
    snprintf(statusLine, sizeof(statusLine), "CANNOT SAVE: card/folder?");
  }
  Serial.printf("%s %s rms=%.1f dBFS peak=%.1f dBFS %s\n", ok ? "SAVED" : "FAILED", path, lastRmsDb, lastPeakDb,
                ok ? "(verified by reading it back)" : "");
}

static void deleteLast() {
  if (historyCount == 0) return;
  char path[40];
  snprintf(path, sizeof(path), "%s", history[--historyCount]);
  SD.remove(path);
  Serial.printf("deleted %s\n", path);
  scanExisting();
  snprintf(statusLine, sizeof(statusLine), "deleted %s", path + 1);
  if (historyCount == 0) haveThumb = false;
}

// ---- Analysis: one 20 ms frame at a time -----------------------------------------------
static void analyseFrame() {
  uint32_t start = framesDone * FRAME;
  int16_t buf[FRAME];
  for (int i = 0; i < FRAME; i++) buf[i] = ring[(start + i) % RING];
  float db = rmsDbOf(buf, FRAME, nullptr);
  frameDbNow = db;
  uint32_t thisFrame = framesDone;
  framesDone++;

  if (phase == P_STOPPED) return;

  if (selected == 3) {  // NOISE: a clip at fixed intervals, no word needed
    if (thisFrame - lastNoiseFrame >= NOISE_PERIOD_FRAMES && thisFrame >= (uint32_t)(CLIP_SAMPLES / FRAME)) {
      lastNoiseFrame = thisFrame;
      captureAndSave((framesDone - (CLIP_SAMPLES / FRAME)) * FRAME);
    }
    return;
  }

  bool above = db > floorDb + marginDb;
  switch (phase) {
    case P_ARMED:
      if (db < floorDb + 6.0f) floorDb += 0.02f * (db - floorDb);  // follow the background
      hits = above ? hits + 1 : 0;
      if (hits >= 2) {  // two frames in a row: a word is starting
        onsetFrame = thisFrame - 1;
        phase = P_CAPTURING;
        snprintf(statusLine, sizeof(statusLine), "word...");
      }
      break;
    case P_CAPTURING:
      if (framesDone >= onsetFrame + POST_FRAMES + 1) {
        int32_t startFrame = (int32_t)onsetFrame - PRE_FRAMES;
        if (startFrame < 0) startFrame = 0;
        captureAndSave((uint32_t)startFrame * FRAME);
        phase = P_COOLDOWN;
        quietFrames = 0;
      }
      break;
    case P_COOLDOWN:  // wait until it is quiet again, so one word makes one clip
      quietFrames = (db < floorDb + 6.0f) ? quietFrames + 1 : 0;
      if (quietFrames >= 25) { phase = P_ARMED; hits = 0; }
      break;
    default: break;
  }
}

// ---- Screen ----------------------------------------------------------------------------------
static void draw() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  char t[48];
  if (!sdOk) {
    canvas.drawCentreString("KWS RECORDER", 86, 4, 1);
    canvas.setTextColor(TFT_RED, TFT_BLACK);
    canvas.drawCentreString("NO SD CARD", 86, 130, 1);
    canvas.pushSprite(0, 0);
    return;
  }
  if (dirError) {
    canvas.drawCentreString("KWS RECORDER", 86, 4, 1);
    canvas.setTextColor(TFT_RED, TFT_BLACK);
    canvas.drawCentreString("CARD FULL", 86, 80, 1);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.drawCentreString("Cannot create the folders.", 86, 120, 1);
    canvas.drawCentreString("The root folder is full.", 86, 134, 1);
    canvas.drawCentreString("Copy the clips to a computer,", 86, 160, 1);
    canvas.drawCentreString("delete them from the card,", 86, 174, 1);
    canvas.drawCentreString("and press RESET.", 86, 188, 1);
    canvas.pushSprite(0, 0);
    return;
  }
  canvas.drawCentreString("KWS RECORDER", 86, 4, 1);
  for (int i = 0; i < NUM_CLASSES; i++) {
    int y = 26 + i * 34;
    bool sel = (i == selected);
    canvas.fillRoundRect(6, y, 160, 30, 6, sel ? TFT_GREEN : TFT_BLACK);
    canvas.drawRoundRect(6, y, 160, 30, 6, sel ? TFT_GREEN : TFT_WHITE);
    canvas.setTextColor(sel ? TFT_BLACK : TFT_WHITE, sel ? TFT_GREEN : TFT_BLACK);
    canvas.setCursor(14, y + 8);
    canvas.print(TITLES[i]);
    snprintf(t, sizeof(t), "%d", nextIndex[i] - 1);
    canvas.setCursor(166 - 12 * strlen(t) - 8, y + 8);
    canvas.print(t);
  }
  // Level meter with the background floor and the trigger line.
  int bx = 6, bw = 160, by = 168;
  auto xOf = [&](float db) { return bx + (int)(bw * constrain((db + 90.0f) / 90.0f, 0.0f, 1.0f)); };
  canvas.drawRect(bx, by, bw, 14, TFT_WHITE);
  uint16_t lc = frameDbNow > -20 ? TFT_RED : (frameDbNow > -40 ? TFT_YELLOW : TFT_GREEN);
  canvas.fillRect(bx + 1, by + 1, xOf(frameDbNow) - bx - 1, 12, lc);
  canvas.drawFastVLine(xOf(floorDb), by - 3, 20, TFT_BLUE);
  canvas.drawFastVLine(xOf(floorDb + marginDb), by - 3, 20, TFT_ORANGE);
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  snprintf(t, sizeof(t), "now %.0f floor %.0f trig %.0f", frameDbNow, floorDb, floorDb + marginDb);
  canvas.setCursor(6, by + 20);
  canvas.print(t);

  const char *ph = phase == P_STOPPED ? "STOPPED" : phase == P_ARMED ? "LISTENING" : phase == P_CAPTURING ? "WORD..." : "SAVED";
  canvas.setTextSize(2);
  canvas.setTextColor(phase == P_STOPPED ? TFT_LIGHTGREY : TFT_GREEN, TFT_BLACK);
  canvas.drawCentreString(selected == 3 && phase != P_STOPPED ? "RECORDING" : ph, 86, 204, 1);
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.drawCentreString(statusLine, 86, 226, 1);

  // Thumbnail of the last clip.
  canvas.drawRect(6, 238, 160, 52, TFT_DARKGREY);
  if (haveThumb) {
    for (int x = 0; x < 160; x++) canvas.drawFastVLine(6 + x, 264 - thumb[x], thumb[x] * 2 + 1, TFT_CYAN);
    snprintf(t, sizeof(t), "rms %.1f  peak %.1f dBFS", lastRmsDb, lastPeakDb);
    canvas.setCursor(6, 294);
    canvas.print(t);
  }
  // Margin buttons.
  canvas.fillRoundRect(6, 304, 40, 14, 4, TFT_BLUE);
  canvas.fillRoundRect(126, 304, 40, 14, 4, TFT_BLUE);
  canvas.setTextColor(TFT_WHITE, TFT_BLUE);
  canvas.drawCentreString("-", 26, 307, 1);
  canvas.drawCentreString("+", 146, 307, 1);
  snprintf(t, sizeof(t), "margin %.0f dB", marginDb);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.drawCentreString(t, 86, 307, 1);
  canvas.pushSprite(0, 0);
}

static void toggleListening() {
  if (phase == P_STOPPED) {
    phase = P_ARMED; hits = 0; floorDb = -70.0f;
    lastNoiseFrame = framesDone;
    snprintf(statusLine, sizeof(statusLine), "say %s", TITLES[selected]);
  } else {
    phase = P_STOPPED;
    snprintf(statusLine, sizeof(statusLine), "stopped");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(800);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) { Serial.println(display.lastResult().message); return; }
  canvas.createSprite(display, 172, 320);
  display.attachTouch(touch, display.panel().driver().bus());
  pinMode(BTN_USR1, INPUT_PULLUP);
  pinMode(BTN_USR2, INPUT_PULLUP);
  sdOk = initSd();
  if (sdOk) {
    ensureDirs();
    if (dirError) Serial.println("cannot create the class folders: the root folder is full");
    else scanExisting();
  }
  Serial.printf("SD %s, %lu MB\n", sdOk ? "ok" : "MISSING", (unsigned long)sdMb);
  if (!micBegin()) { Serial.println("mic init FAILED"); return; }
  xTaskCreatePinnedToCore(captureTask, "audio", 6144, nullptr, configMAX_PRIORITIES - 2, nullptr, 0);
  draw();
}

void loop() {
  static bool d1 = false, d2 = false, touchWasDown = false;
  static uint32_t lastDraw = 0;
  bool b1 = digitalRead(BTN_USR1) == LOW, b2 = digitalRead(BTN_USR2) == LOW;
  bool press1 = b1 && !d1, press2 = b2 && !d2;
  d1 = b1; d2 = b2;
  bool dirty = false;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == 's') { toggleListening(); dirty = true; }
    if (c == 'c') { selected = (selected + 1) % NUM_CLASSES; dirty = true; }
    if (c == 'x') { deleteLast(); dirty = true; }
    if (c == 'd') {  // d/yes.own.003.wav : delete one file by name (debug)
      String name = Serial.readStringUntil('\n'); name.trim();
      if (name.length() > 1 && SD.exists(name)) { SD.remove(name); Serial.printf("deleted %s\n", name.c_str()); scanExisting(); }
      else Serial.printf("not found: %s\n", name.c_str());
    }
    if (c == 'l') {  // the root, and how many files each folder holds
      File r = SD.open("/");
      for (File f = r.openNextFile(); f; f = r.openNextFile()) {
        if (f.isDirectory()) {
          int n = 0; for (File g = f.openNextFile(); g; g = f.openNextFile()) n++;
          Serial.printf("%s/  %d files\n", f.name(), n);
        } else Serial.printf("%s  %u bytes\n", f.name(), (unsigned)f.size());
      }
    }
  }
  if (!sdOk || dirError) { if (press1) {} delay(100); draw(); return; }
  if (press1) { toggleListening(); dirty = true; }
  if (press2) { deleteLast(); dirty = true; }

  // Touch is only read in this loop (the microphone does not use I2C).
  int32_t x, y;
  bool down = display.getTouch(&x, &y);
  if (down && !touchWasDown) {
    if (y >= 26 && y < 26 + NUM_CLASSES * 34) { selected = (y - 26) / 34; dirty = true; }
    else if (y >= 300) {
      if (x < 60) marginDb = fmaxf(6.0f, marginDb - 3.0f);
      else if (x > 112) marginDb = fminf(40.0f, marginDb + 3.0f);
      dirty = true;
    }
  }
  touchWasDown = down;

  // Analyse every frame that has arrived. If we fell more than 1.2 s behind, skip ahead.
  uint32_t availFrames = written / FRAME;
  if (availFrames > framesDone + (uint32_t)(1.2f * SAMPLE_RATE / FRAME)) { framesDone = availFrames - 1; Serial.println("analysis fell behind, skipping"); }
  while (framesDone < availFrames) analyseFrame();

  if (dirty || millis() - lastDraw > 200) { lastDraw = millis(); draw(); }
  delay(2);
}
```
<!-- /sketch -->

**A second session of noise.** After the fix, the recorder saved 50 more NOISE
clips into `/noise/` in one go. They are byte for byte identical to what was on
the card, all 16 kHz, 16-bit, mono, exactly 1.00 s, none clipped, and the review
flagged none. Their median level is -55.7 dBFS (10th to 90th percentile: -64.3 to
-42.1), still about 28 dB below the dataset's noise clips, with a few louder
events in the room. Together with the first 16 that makes 66 noise clips.

#### Preparing the upload

`tools/kws_prepare_upload.py` gathers the sessions into one folder, never moving
the originals. The recorder numbers its clips from 001 in every session, so when
two sessions use the same name, the later one gets its folder name in the file
name (`noise.own2.001.wav`); the part before the first dot is still the label, which
is how the Studio infers it. It sets aside the clips you want to listen to first
and splits the rest at random, about two thirds for training and one third for
testing:

| Class | Train | Test | Review first | Total |
|---|---|---|---|---|
| yes | 30 | 15 | 5 | 50 |
| no | 33 | 16 | 1 | 50 |
| unknown | 31 | 15 | 4 | 50 |
| noise | 44 | 22 | 0 | 66 |

The ten clips in *Review first* are the ones the review flagged. They are kept out
of the test set so that doubtful clips do not distort the measurement. The split
is random with a fixed seed, so it is repeatable, but it is not stratified by
anything else, and with 15 or 16 test clips per word a single mistake moves the
accuracy by about 6 points. Treat the result as a rough indication.

### A limit we hit: the root folder of a small FAT16 card

![The recorder showing WRITE FAILED while the NOISE class stands at 16 clips](images/kws_recorder_write_failed.jpg)

**Symptom.** After 166 clips the recorder showed `WRITE FAILED` and saved nothing
more. That is why there are only 16 NOISE clips.

**Cause.** It was not a full card: the 947 MB card was 99% free. A FAT16 volume
has a fixed-size root folder, typically 512 entries, and a file with a long name
uses several of them. Names like `unknown.own.001.wav` use three entries each, so
166 files take 498 entries, and with the three items already on the card and the
volume label, the count reaches 508. macOS adds a few more hidden items (such as
`.fseventsd` and `.Trashes`) when it mounts the card, which brings it to 512.

**How we checked.** With the card mounted on the Mac, creating even one new file
with a long name failed with *no space left on device*, while the volume reported
936 MiB free. The board gave the same answer when it tried to create the class
folders on that card. We did not read the root-folder size from the volume itself,
so "512 entries" is the usual value for FAT16, not something we measured on this
card.

**Fix.** The recorder and the logger now keep each class in its own folder.
Subfolders do not have the root's limit. Creating the folders needs a few free
entries in the root, so on a card whose root is already full, the sketches say so
on the screen (`CARD FULL`) and on the serial port, instead of failing quietly. To
recover such a card, copy the clips to a computer, delete them from the card, and
reset the board.

**Tested after the fix.** We cleared the card (the 166 clips were first checked
byte for byte against the copies on the Mac) and reset the board. It created the
class folders on its own. The logger saved a 500-row file into `/idle/` with no
lost samples, and the recorder saved six NOISE clips into `/noise/`, each read
back and verified; we removed those test files afterwards. We have not run a full
recording session with the fixed sketches yet.

### The model

The model was trained in the Edge Impulse Studio by the author (project
[XIAO IPS Display - KWS](https://studio.edgeimpulse.com/public/1129422/live), a
public project) from the dataset plus the recordings from test 03. The library it
generated is not in this repository. What it contains, read from its files:

- the input is 16,000 samples (1 second at 16 kHz); 13 MFCC coefficients over 50
  frames, 650 values, go into the network;
- four classes, in this order: `no`, `noise`, `unknown`, `yes` (alphabetical, so
  `yes` is index 3);
- the network is compiled with Edge Impulse's EON compiler and quantized to int8;
  the working memory it asks for is 6,265 bytes, and the model code is 56 KB, so it
  needs **no PSRAM** (the book asks for it, for the Sense board's sketch);
- it detects no anomalies.

The Studio's public page reports **91.5% on the validation set and 87.0% on the test
set**, and an estimate of 366 ms of latency, 15.4 KB of RAM and 30.8 KB of flash for
the XIAO ESP32-S3 Plus. We read those numbers from the page and did not reproduce
the Studio's test.

### Test 04: keyword spotting on the board

Runs the model on the microphone, continuously: the microphone is cut into 250 ms
slices (test 02), and every slice runs the classifier on the last second of audio.
The screen shows the detected word in large letters (YES in green, NO in red), a bar
for each of the four classes, the processing times, and the level. A word is shown
when YES or NO wins with a score of at least 0.80 (a first guess, not a tuned value),
and stays on screen for one second.

**Build it with ESP-NN turned off.** The file `build_opt.h` next to the sketch does
that. It contains one line:

```
-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=0
```

Both the Arduino IDE and `arduino-cli` read it. Without it, the model gives wrong
answers on this board (next section).

**Measured.** The sketch uses 597 KB of flash (28%) and 45 KB of RAM (13%), with
PSRAM disabled. For each slice, the signal processing takes about 16 to 17 ms and the
network about 15 ms, so roughly 32 ms of the 250 ms available. In a quiet room the
class `noise` wins nearly everywhere, but with less certainty than before the fix
(scores of 0.46 to 0.89), and `unknown` sometimes reaches 0.4 to 0.5. We have not
measured the live behavior with a voice.

#### How it works

- **The capture is the one from test 02:** two slice buffers filled by a task on core 0.
- **`numpy::int16_to_float`** hands the slice to Edge Impulse as floats in the int16
  range. The MFCC block then divides by 32768 itself.
- **`run_classifier_continuous()`** keeps the earlier slices, so each call classifies
  the last second. The first full window needs four slices; until then the screen
  says "warming up".
- **The labels come from the model,** so the screen shows whatever the four classes
  are called, in the model's own order.

<!-- sketch: part2_tinyml/04_kws_inference/04_kws_inference.ino -->
**Sketch:** [`part2_tinyml/04_kws_inference/04_kws_inference.ino`](https://github.com/Mjrovai/XIAO-IPS-Display-ESP32S3/blob/main/part2_tinyml/04_kws_inference/04_kws_inference.ino)

```cpp
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
```
<!-- /sketch -->

### A problem we found: the ESP-NN kernels give wrong answers

**What happened.** The first live attempt classified everything as `noise`. The
recording showed rhythmic bursts of speech, yet `yes` never went above 0.09. We
then fed recorded clips to the board over the USB cable (test 05) to take the
microphone and the room out of the picture. Even the dataset's own clips, which the
model had been trained on, came out as `noise`: **13 of 48 correct (27%)**, with 0 of
12 for `yes` and 0 of 12 for `unknown`.

**What we ruled out, in this order:**

1. *The clips arriving damaged.* The board returns a checksum of what it received;
   all of them matched.
2. *The way we feed the classifier.* The Edge Impulse source confirms the format:
   floats in the int16 range, as the sketch does.
3. *The model.* We built the same library, with the same SDK and the same compiled
   network, as a program for the Mac. On the same 48 clips it got **44 of 48 (91.7%)**,
   in line with the Studio's 87 to 91%.

**The cause.** On the ESP32-S3 the SDK turns on two optimized paths by default: ESP-DSP
(a fast FFT for the signal processing) and ESP-NN (optimized neural-network kernels).
We turned them off one at a time, with a compile-time flag, and scored the same 48
clips on the board each time:

| Build | Correct, of 48 |
|---|---|
| Default (ESP-DSP on, ESP-NN on) | 13 (27.1%) |
| ESP-DSP off, ESP-NN on | 13 (27.1%) |
| **ESP-NN off, ESP-DSP on** | **44 (91.7%)** |
| ESP-DSP off, ESP-NN off | 44 (91.7%) |
| The same model on the Mac | 44 (91.7%) |

With ESP-NN off, the board's confusion matrix matches the Mac's cell for cell.
ESP-DSP is fine, and keeping it makes the signal processing about twice as fast. The
cost of turning ESP-NN off is that the network takes about 15 ms per slice instead of
1 to 2 ms. The 1 to 2 ms were fast because they were wrong.

**How far this goes.** We found it with Arduino core 3.3.12, with this one model, and
with the Edge Impulse SDK inside the library we downloaded. We did not try other cores,
other SDK versions, or other models, and we do not know if the fault is in the ESP-NN
code, in how it is built here, or in how it meets this core. The book tells you to
stay on core 2.0.17 and not update it; that is consistent with a problem in newer
cores, but we did not test 2.0.17.

### Test 05: replay, a hardware-in-the-loop test

Classifies audio that a computer sends over the USB cable, with the same signal
processing and the same network as test 04. It is how the problem above was found,
and it is a good way to test any model on this board without depending on the room.
It has three commands: `WAVE` (a whole 1-second clip), and `SLCE` and `RSET` (the
continuous classifier, one 250 ms slice at a time, as in test 04).

The tools on the computer (standard library only, except the program for the Mac):

- `tools/kws_replay.py` sends whole clips and scores them, optionally made louder or
  quieter. With `--host` it runs the same clips through the program for the Mac.
- `tools/kws_stream_eval.py` places each clip between two stretches of noise and runs
  it in continuous mode, applying test 04's rule, on the board (`--port`) or on the
  Mac.
- `tools/ei_host_test/` holds the small program that runs the library on the Mac, and
  `build.py` builds it from the unzipped library.

<!-- sketch: part2_tinyml/05_kws_replay/05_kws_replay.ino -->
**Sketch:** [`part2_tinyml/05_kws_replay/05_kws_replay.ino`](https://github.com/Mjrovai/XIAO-IPS-Display-ESP32S3/blob/main/part2_tinyml/05_kws_replay/05_kws_replay.ino)

```cpp
/*
  Part 2, Test 05 - Keyword spotting replay (hardware-in-the-loop test)

  Classifies audio clips that a computer sends over the USB serial port, using the
  same Edge Impulse pre-processing and the same quantized network that run on the
  microphone in test 04. The microphone and the room are taken out of the loop, so
  you can measure the model on this board with controlled inputs: your own
  recordings, the dataset's clips, and the same clips made louder or quieter.

  Protocol: the computer sends the 4 bytes "WAVE" followed by 16000 samples of
  16-bit little-endian audio (32000 bytes, one second at 16 kHz). The board answers
  with one line:
    RESULT <label> <score> ... dsp <ms> nn <ms> sum <sum of samples> peak <max abs sample>
  tools/kws_replay.py does the sending and the counting.

  The classifier for WAVE is run_classifier(): it sees exactly one clip. Two more commands
  exercise the continuous classifier that test 04 uses:
    "RSET"                      starts a new stream (resets the continuous state)
    "SLCE" + 4000 samples       one 250 ms slice; answers  SLICE <label> <score> ...
  tools/kws_stream_eval.py --port sends a clip between two stretches of noise, slice by
  slice, and counts the words detected.
*/

#include <Arduino.h>
#include <XIAO_IPS_Display_-_KWS_inferencing.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"

static constexpr int N = EI_CLASSIFIER_RAW_SAMPLE_COUNT;  // 16000 samples
static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;

Seeed_GFX display;
Seeed_Sprite canvas;
static int16_t clip[N];
static int16_t sliceBuf[EI_CLASSIFIER_SLICE_SIZE];
static uint32_t received = 0;
static char lastLine[48] = "waiting for clips";

static int clipGetData(size_t offset, size_t length, float *out) {
  numpy::int16_to_float(&clip[offset], out, length);
  return 0;
}

static int sliceGetData(size_t offset, size_t length, float *out) {
  numpy::int16_to_float(&sliceBuf[offset], out, length);
  return 0;
}

static void draw() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(2);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.drawCentreString("KWS REPLAY", 86, 6, 1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  char t[32];
  snprintf(t, sizeof(t), "%lu clips", (unsigned long)received);
  canvas.drawCentreString(t, 86, 60, 1);
  canvas.setTextSize(1);
  canvas.drawCentreString(lastLine, 86, 110, 1);
  canvas.pushSprite(0, 0);
}

// Read exactly n bytes, or give up after timeoutMs without progress.
static bool readExact(uint8_t *dst, size_t n, uint32_t timeoutMs) {
  size_t got = 0;
  uint32_t last = millis();
  while (got < n) {
    int avail = Serial.available();
    if (avail > 0) {
      got += Serial.readBytes(dst + got, min((size_t)avail, n - got));
      last = millis();
    } else if (millis() - last > timeoutMs) {
      return false;
    }
  }
  return true;
}

void setup() {
  Serial.setRxBufferSize(40000);  // a whole clip fits, so the computer never waits on us
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(800);
  if (display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                    Config_Seeed_1inch47_Touch_JD9853A>()) {
    canvas.createSprite(display, 172, 320);
    draw();
  }
  run_classifier_init();
  Serial.printf("READY replay %s %d samples\n", EI_CLASSIFIER_PROJECT_NAME, N);
}

static void answerWave() {
  if (!readExact((uint8_t *)clip, N * sizeof(int16_t), 3000)) { Serial.println("ERROR timeout while reading the clip"); return; }
  signal_t signal;
  signal.total_length = N;
  signal.get_data = &clipGetData;
  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) { Serial.printf("ERROR classifier %d\n", (int)err); return; }
  Serial.print("RESULT");
  int top = 0;
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    Serial.printf(" %s %.3f", result.classification[i].label, result.classification[i].value);
    if (result.classification[i].value > result.classification[top].value) top = i;
  }
  // A checksum and the peak of what arrived, so the computer can tell whether the clip
  // was received intact.
  int32_t sum = 0; int peak = 0;
  for (int i = 0; i < N; i++) { sum += clip[i]; if (abs(clip[i]) > peak) peak = abs(clip[i]); }
  Serial.printf(" dsp %lu nn %lu sum %ld peak %d\n", (unsigned long)result.timing.dsp,
                (unsigned long)result.timing.classification, (long)sum, peak);
  received++;
  snprintf(lastLine, sizeof(lastLine), "%s %.2f", result.classification[top].label, result.classification[top].value);
  if (received % 10 == 0) draw();  // the screen is slow; refresh every tenth clip
}

static void answerSlice() {
  const int n = EI_CLASSIFIER_SLICE_SIZE;
  if (!readExact((uint8_t *)sliceBuf, n * sizeof(int16_t), 3000)) { Serial.println("ERROR timeout while reading the slice"); return; }
  signal_t signal;
  signal.total_length = n;
  signal.get_data = &sliceGetData;
  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR err = run_classifier_continuous(&signal, &result, false);
  if (err != EI_IMPULSE_OK) { Serial.printf("ERROR classifier %d\n", (int)err); return; }
  Serial.print("SLICE");
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) Serial.printf(" %s %.3f", result.classification[i].label, result.classification[i].value);
  Serial.println();
}

void loop() {
  // Keep the last 4 bytes; a command is a 4-letter marker: WAVE, SLCE, or RSET.
  static uint8_t win[4] = {0, 0, 0, 0};
  while (Serial.available()) {
    win[0] = win[1]; win[1] = win[2]; win[2] = win[3]; win[3] = (uint8_t)Serial.read();
    if (!memcmp(win, "WAVE", 4)) { memset(win, 0, 4); answerWave(); }
    else if (!memcmp(win, "SLCE", 4)) { memset(win, 0, 4); answerSlice(); }
    else if (!memcmp(win, "RSET", 4)) { memset(win, 0, 4); run_classifier_init(); Serial.println("RESET ok"); }
  }
}
```
<!-- /sketch -->

### What the model does on the board, measured with recorded clips

**Whole clips, on your own test clips** (68 clips never used in training: 16 `no`,
22 `noise`, 15 `unknown`, 15 `yes`): **58 correct (85.3%)**, identical on the board
and on the Mac. Recall by class: `no` 93.8%, `yes` 93.3%, `unknown` 80.0%, `noise`
77.3%.

**Continuous mode, as in the live sketch.** Each clip sits between two stretches of
your room noise, and a clip counts as a detection when YES or NO wins with at least
0.80 in some slice:

| Clips | `yes` detected | `no` detected | False alarms on `noise` | False alarms on `unknown` |
|---|---|---|---|---|
| Your 68 test clips | 93.3% (14 of 15) | 93.8% (15 of 16) | 0% (0 of 22) | 20% (3 of 15) |
| 400 dataset clips (100 per class) | 84% | 84% | 0% | 10% |

The board and the Mac gave **the same table, cell for cell, on all 468 clips.** The
dataset row is optimistic, because many of its clips were used in training; the first
row is the fairer one. With 15 or 16 clips per word, one clip moves a percentage by
about 6 points. On the Mac only, with your test clips, the threshold is a trade-off:
at 0.6, `no` is detected 100% of the time but the false alarms on `unknown` rise to
26.7%; at 0.9, `no` falls to 75% and those false alarms to 6.7%.

**The level does not matter.** The scores for a clip were identical, to three
decimals, from +12 dB louder to 36 dB quieter. On the board, whole-clip accuracy was
the same at +0, +6, and +12 dB. A likely reason, which we have read in the code but
not isolated with an experiment: the MFCC block normalizes the coefficients over a
sliding window (its `win_size` is 101), which removes a constant gain, and the
network's outputs are quantized in steps of about 0.004.

### The live test with a voice

Three live attempts, in order:

1. With ESP-NN on (the default), speech was clearly present, yet `yes` never went above
   0.09. This is what led to the problem described above.
2. With ESP-NN off, but no speech in the recording (the loudest slice was -48 dBFS), so
   it proves nothing.
3. **With ESP-NN off and a voice,** on the board's own microphone, over about 100 s. The
   loudest slice was -29 dBFS, and 67 of the 400 slices were louder than -45 dBFS.

What attempt 3 shows:

- **The model detects the words.** There were 27 stretches in which the winner was not
  `noise`. In 7 of them only `yes` reached 0.80, in 11 only `no`, in 4 both did, and in 5
  neither (the winner was `unknown`, or the scores were weak). A clean word gives a score
  of 1.00 for 3 to 5 slices in a row, which is 0.75 to 1.25 s.
- **Words that are not YES or NO mostly land in `unknown`,** with scores of 0.88 to 0.99.
- **No false alarms in the quiet.** Of the 397 one-second windows whose levels are all
  known, 168 were entirely quiet (every slice below -50 dBFS). In none of them did `yes`
  or `no` reach 0.80.
- **The label can flicker inside one word.** Several stretches switch between `yes`,
  `no`, and `unknown` for a slice or two, for example `yes yes yes yes unk unk no yes yes
  yes`. A rule that asks for two slices in a row would hide those flickers; we have not
  tried it.

What it does **not** show: an accuracy per word. The exact order and number of the words
spoken, with their times, were not recorded, so we cannot say how many `yes` were missed
or how many `no` were taken for `yes`. For that number, the test needs ground truth, for
example a cue (a beep) before every word, logged on the computer, so that each detection
can be matched to what was asked. The measurements on recorded clips above remain the
reliable numbers.

### Next steps for keyword spotting

1. **A live test with ground truth.** The first live test shows the model works, but not
   how well. Play a short beep before each word, log the time of every beep on the
   computer, ask for a known sequence (for example YES, NO, and other words in a random
   order), and match each detection to the word that was asked.
2. **Tune the detection rule.** The threshold of 0.80 is a guess. On your test clips
   it misses about 6% of the words and `unknown` triggers a word 20% of the time. A
   rule that asks for two slices in a row, or a different threshold for each word,
   might do better; test it on recorded clips first, with the tools above.
3. **Find out if the ESP-NN problem is the core.** Build the same replay sketch on core
   2.0.17 (the book's version), in a separate setup so that your installation is not
   changed. It would say whether the fault comes from the newer core.
4. **Wake-word use.** A single word is not a wake word; a real one needs a model made
   for it, trained to reject speech that is not the word.
