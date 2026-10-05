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

**Status:** the audio front end and a recorder for your own keywords are written,
the dataset is downloaded and checked, and a first session of your own clips (166)
is recorded and reviewed. Training in the Edge Impulse Studio and the
model are not done yet.

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

**A risk to check.** These clips are fairly loud. Our microphone measured lower
levels for speech in Part 1, test 04, but that was the RMS of 64 ms blocks, which
is not the same measurement as the RMS of a whole 1-second clip, so the two are
not directly comparable. Whether speech picked up by this board is quieter than
the training data, and whether that hurts the model, is something to measure once
there is a model and recordings from the board.

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
not test it with a model, so the processing time of a real classifier is not
known yet. The book says the KWS sketch needs PSRAM enabled; this test does not
use it, and we have not yet checked how much memory a real model needs on this
board.

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
  a whole 1-second clip). It is the risk we noted earlier, now measured. These
  recordings were made at one distance and one speaking volume, so the gap is one
  speaker's, not a property of the microphone.
- **A fixed gain is not a free fix.** Raising everything by 14 dB would match the
  medians, but the louder clips would clip: the 90th percentile of the peaks is
  already -5.7 dBFS for YES and -7.2 for UNKNOWN. Whether a gain helps the model
  is something to test, not assume (see the next steps).
- **Ten clips look suspicious** out of 150 word clips: `yes.own.003`, `022`,
  `038`, `039`, `047`; `no.own.026`; `unknown.own.001`, `022`, `036`, `046`. The
  reasons are a word that may be cut off, two separate bursts, or a very quiet
  clip (`yes.own.022` is at -62 dBFS). The review looks only at the signal. It
  cannot tell whether the word you said is the word in the label, so listen to
  these before you upload them.
- **The noise class is quiet and short.** Its median is -65 dBFS, against -27 for
  the dataset's noise clips, and there are only 16 of them. More noise recorded
  in the room where the board will be used would help.

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
reset the board. The fixed sketches compile; we have not run the full recording
session with them yet.

### Next steps for keyword spotting

1. Upload the four dataset folders to a new Edge Impulse project, with *Data acquisition,
   Upload existing data*, inferring the label from the file name and letting the
   Studio split training and test data (as in the book).
   Add your own clips from test 03 in the same way, sending about a third of them
   to the *Testing* category.
2. Build the impulse with the book's settings: 1-second windows, MFCC features, and a small 1D convolutional
   network (two Conv1D and pooling blocks with 8 and 16 filters, dropout 0.25,
   learning rate 0.005, 100 epochs, noise augmentation).
3. Test, then deploy as an Arduino library (quantized, int8).
   Also test the model on your own clips, kept out of training, and compare it with
   the same clips made louder by 6 dB and by 12 dB (limited so they do not clip)
   to learn whether a gain on the microphone helps. That is an experiment to run,
   not a result.
4. Replace `process()` with the Edge Impulse classifier, and show the detected
   word and its confidence on the display.
