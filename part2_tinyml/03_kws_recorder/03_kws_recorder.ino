/*
  Part 2, Test 03 - Keyword recorder (your own voice, as WAV files on the microSD)

  Listens to the microphone and, every time it hears a word, saves a 1-second clip
  with the word in it (300 ms before the start, 700 ms after) as a 16 kHz, 16-bit,
  mono WAV named like the Edge Impulse dataset: yes.own.001.wav, no.own.014.wav,
  and so on. Say a word, wait for "SAVED", say it again. No computer is needed.

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

static void pathFor(int cls, int idx, char *out, size_t n) { snprintf(out, n, "/%s.own.%03d.wav", LABELS[cls], idx); }

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
    snprintf(statusLine, sizeof(statusLine), "WRITE FAILED");
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
  snprintf(t, sizeof(t), "now %.0f  floor %.0f  trigger %.0f dBFS", frameDbNow, floorDb, floorDb + marginDb);
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
  if (sdOk) scanExisting();
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
    if (c == 'l') { File r = SD.open("/"); for (File f = r.openNextFile(); f; f = r.openNextFile()) Serial.printf("%s %u\n", f.name(), (unsigned)f.size()); }
  }
  if (!sdOk) { delay(100); return; }
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
