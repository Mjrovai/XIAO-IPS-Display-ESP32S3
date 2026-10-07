/*
  Part 3, Test 09 - The VerneBot GRU on the board

  Runs the character-level GRU of the VerneBot project (4,095,867 parameters, 123 characters,
  trained on ten Jules Verne novels) and writes text with it, one character at a time. The weights
  are read from the microSD card into PSRAM as 8-bit integers with one scale per row (4.2 MB),
  made by tools/verne_gru.py from the project's float16 export. The arithmetic is the GRU of the
  project's JavaScript demo (Keras with reset_after = True): the weights are 8-bit, the numbers
  that flow through the network are 32-bit floats.

  File on the card, in the folder /slm:   verne_rnn_int8.bin
  Build with PSRAM = OPI PSRAM.

  Use it:
    USR1  write a new text (or stop the one being written)
    USR2  change the beginning of the text (the seed), then press USR1
    Touch: drag up or down to scroll, tap to switch between big and small letters.
  Serial (115200): the text as it is written, then a line with the speed. Commands:
    g <seed>   greedy writing of 64 characters (the most likely character every time); used to compare
               with tools/verne_gru.py greedy --int8
    t <seed>   the 8 most likely next characters after the seed; compare with tests/reference.json
*/

#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <math.h>
#include <Seeed_GFX.h>
#include "board/boards/XIAO_LCD_Board.h"
#include "driver/tft/Driver_JD9853A.h"
#include "panel/Panel_TFT.h"
#include "bus/Bus_SPI.h"
#include "touch/Touch_AXS5106L.h"

static constexpr int8_t LCD_RST_PIN = 13;
static constexpr int8_t LCD_BL_PIN = 12;
static constexpr uint8_t SD_CS = D6;
static constexpr uint8_t SD_MISO = D9;
static constexpr uint8_t BTN_USR1 = D19;
static constexpr uint8_t BTN_USR2 = D15;

Seeed_GFX display;
Seeed_Sprite canvas;
Touch_AXS5106L touch(-1, D7, Wire, 172, 320);

static constexpr int V = 123, D = 256, U = 1024, CONTEXT = 120;
static constexpr int WRITE_CHARS = 300;          // how many characters one press of USR1 writes

// Declared before the functions, so that the Arduino build places its prototypes after it.
struct Mat { int8_t *w; float *s; int rows, cols; };   // out[i] = s[i] * sum_j w[i][j] * x[j]
typedef bool (*TickFn)(const char *piece, int count);   // return false to stop

// ---- The model in memory ------------------------------------------------------------------------------
static char vocab[V][5];                         // each character of the vocabulary, as UTF-8
static uint8_t vocabLen[V];
static float *wte, *biasIn, *biasRec, *denseB;   // biasIn, biasRec: 3 x 1024 each, gates in the order z, r, h
static Mat Wx, Wh, Wd;                           // Wx, Wh: 3072 rows (z, r, h)
static float *h, *gx, *gh, *logits;
static float lastCharsPerSec = 0;

static bool readAll(File &f, void *dst, size_t n) {
  uint8_t *p = (uint8_t *)dst;
  while (n) {
    int got = f.read(p, min(n, (size_t)16384));
    if (got <= 0) return false;
    p += got; n -= got;
  }
  return true;
}

static bool readMat(File &f, Mat &m, int rows, int cols) {
  m.rows = rows; m.cols = cols;
  m.s = (float *)ps_malloc(rows * sizeof(float));
  m.w = (int8_t *)ps_malloc((size_t)rows * cols);
  return m.s && m.w && readAll(f, m.s, rows * 4) && readAll(f, m.w, (size_t)rows * cols);
}

// Print the folders in the root of the card and the files of /slm, to see what is there when the model is not found.
static void listCard() {
  File d = SD.open("/");
  if (d && d.isDirectory())
    for (File e = d.openNextFile(); e; e = d.openNextFile()) Serial.printf("  /%s%s\n", e.name(), e.isDirectory() ? "/" : "");
  File m = SD.open("/slm");
  if (m && m.isDirectory())
    for (File e = m.openNextFile(); e; e = m.openNextFile()) Serial.printf("  /slm/%s  %u\n", e.name(), (unsigned)e.size());
  File m2 = SD.open("/slm/slm");
  if (m2 && m2.isDirectory())
    for (File e = m2.openNextFile(); e; e = m2.openNextFile()) Serial.printf("  /slm/slm/%s  %u\n", e.name(), (unsigned)e.size());
}

static const char *loadModel() {
  File f = SD.open("/slm/verne_rnn_int8.bin", FILE_READ);
  if (!f) return "no /slm/verne_rnn_int8.bin";
  char magic[4]; uint32_t hdr[7];
  if (!readAll(f, magic, 4) || memcmp(magic, "VGRU", 4) || !readAll(f, hdr, 28)) return "bad file";
  if (hdr[1] != V || hdr[2] != D || hdr[3] != U) return "unexpected model shape";
  for (int i = 0; i < V; i++) {
    uint8_t n; if (!readAll(f, &n, 1) || n > 4) return "bad vocabulary";
    vocabLen[i] = n; memset(vocab[i], 0, 5);
    if (!readAll(f, vocab[i], n)) return "bad vocabulary";
  }
  wte = (float *)ps_malloc(V * D * 4); biasIn = (float *)ps_malloc(3 * U * 4); biasRec = (float *)ps_malloc(3 * U * 4); denseB = (float *)ps_malloc(V * 4);
  h = (float *)ps_malloc(U * 4); gx = (float *)ps_malloc(3 * U * 4); gh = (float *)ps_malloc(3 * U * 4); logits = (float *)ps_malloc(V * 4);
  if (!wte || !biasIn || !biasRec || !denseB || !h || !gx || !gh || !logits) return "no PSRAM (use PSRAM=opi)";
  if (!readAll(f, wte, V * D * 4) || !readAll(f, biasIn, 3 * U * 4) || !readAll(f, biasRec, 3 * U * 4) || !readAll(f, denseB, V * 4)) return "read failed";
  if (!readMat(f, Wx, 3 * U, D) || !readMat(f, Wh, 3 * U, U) || !readMat(f, Wd, V, U)) return "read failed or out of PSRAM";
  f.close();
  return nullptr;
}

// ---- The GRU ------------------------------------------------------------------------------------------------
static void mv(float *out, const Mat &m, const float *x) {
  for (int i = 0; i < m.rows; i++) {
    const int8_t *w = m.w + (size_t)i * m.cols;
    float a0 = 0, a1 = 0;
    for (int j = 0; j + 1 < m.cols; j += 2) { a0 += (float)w[j] * x[j]; a1 += (float)w[j + 1] * x[j + 1]; }
    out[i] = (a0 + a1) * m.s[i];
  }
}

static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

static float *step(int c) {          // one character in, the scores of the next one out
  mv(gx, Wx, wte + c * D);
  mv(gh, Wh, h);
  for (int k = 0; k < U; k++) {
    float z = sigm(gx[k] + gh[k] + biasIn[k] + biasRec[k]);
    float r = sigm(gx[U + k] + gh[U + k] + biasIn[U + k] + biasRec[U + k]);
    float hh = tanhf(gx[2 * U + k] + biasIn[2 * U + k] + r * (gh[2 * U + k] + biasRec[2 * U + k]));
    h[k] = z * h[k] + (1.0f - z) * hh;
  }
  mv(logits, Wd, h);
  for (int i = 0; i < V; i++) logits[i] += denseB[i];
  return logits;
}

// ---- Text in and out ----------------------------------------------------------------------------------------
static int charIndex(char c) {       // the vocabulary index of a one-byte character, or -1
  for (int i = 0; i < V; i++) if (vocabLen[i] == 1 && vocab[i][0] == c) return i;
  return -1;
}

// The screen font is ASCII: show the typographic characters of the books as their plain versions.
static const char *screenForm(int i) {
  static char one[2]; const char *v = vocab[i];
  if (vocabLen[i] == 1) return v;
  const char *plain = nullptr;
  if (!strcmp(v, "\xE2\x80\x98") || !strcmp(v, "\xE2\x80\x99") || !strcmp(v, "\xE2\x80\xB2")) plain = "'";
  else if (!strcmp(v, "\xE2\x80\x9C") || !strcmp(v, "\xE2\x80\x9D") || !strcmp(v, "\xE2\x80\xB3")) plain = "\"";
  else if (!strcmp(v, "\xE2\x80\x94")) plain = "-";
  else if (!strncmp(v, "\xC3\xA0", 2) || !strncmp(v, "\xC3\xA2", 2)) plain = "a";
  else if (!strncmp(v, "\xC3\xA8", 2) || !strncmp(v, "\xC3\xA9", 2) || !strncmp(v, "\xC3\xAA", 2) || !strncmp(v, "\xC3\xAB", 2)) plain = "e";
  else if (!strncmp(v, "\xC3\xAF", 2)) plain = "i";
  else if (!strncmp(v, "\xC3\xB4", 2)) plain = "o";
  else if (!strncmp(v, "\xC3\xBB", 2) || !strncmp(v, "\xC3\xBC", 2)) plain = "u";
  else if (!strncmp(v, "\xC3\xA7", 2)) plain = "c";
  else if (!strncmp(v, "\xC3\xB1", 2)) plain = "n";
  else if (!strncmp(v, "\xC3\x80", 2)) plain = "A";
  else if (!strncmp(v, "\xC3\x8A", 2)) plain = "E";
  if (plain) return plain;
  one[0] = '?'; one[1] = 0; return one;
}

static uint32_t rngS = 1;
static float randF() { rngS ^= rngS << 13; rngS ^= rngS >> 17; rngS ^= rngS << 5; return (rngS >> 8) / 16777216.0f; }

static int sampleChar(const float *lg, float temperature) {
  if (temperature <= 0) { int b = 0; for (int i = 1; i < V; i++) if (lg[i] > lg[b]) b = i; return b; }
  static float p[V]; float mx = lg[0];
  for (int i = 1; i < V; i++) if (lg[i] > mx) mx = lg[i];
  float sum = 0;
  for (int i = 0; i < V; i++) { p[i] = expf((lg[i] - mx) / temperature); sum += p[i]; }
  float r = randF() * sum, acc = 0;
  for (int i = 0; i < V; i++) { acc += p[i]; if (r < acc) return i; }
  return V - 1;
}

// Feed the seed through the network from a zero state (the last 120 characters, as in the demo), then write.
static int writeText(const char *seed, int count, float temperature, uint32_t seedValue, TickFn tick, uint32_t *usOut) {
  memset(h, 0, U * 4);
  int ids[CONTEXT], n = 0;
  for (const char *p = seed; *p; p++) { int i = charIndex(*p); if (i >= 0) { if (n == CONTEXT) { memmove(ids, ids + 1, (CONTEXT - 1) * sizeof(int)); n--; } ids[n++] = i; } }
  if (n == 0) ids[n++] = charIndex(' ');
  float *lg = nullptr;
  for (int i = 0; i < n; i++) lg = step(ids[i]);
  rngS = seedValue ? seedValue : 1;
  uint32_t us = 0; int made = 0;
  for (int i = 0; i < count; i++) {
    uint32_t t0 = micros();
    int c = sampleChar(lg, temperature);
    Serial.write((const uint8_t *)vocab[c], vocabLen[c]);
    if (tick && !tick(screenForm(c), i + 1)) break;
    lg = step(c);
    us += micros() - t0; made++;
    lastCharsPerSec = us ? made * 1e6f / us : 0;
  }
  if (usOut) *usOut = us;
  return made;
}

// ---- Screen and buttons -------------------------------------------------------------------------------------
static const char *PROMPTS[] = {"THE FLYING SUBMARINE", "Captain Nemo", "The balloon rose", "It was a dark night"};
static constexpr int NUM_PROMPTS = 4;
static int promptIndex = 0;
static char story[4096];
static int storyLen = 0;
static uint32_t lastDrawMs = 0;
static bool wasPressed1 = false, wasPressed2 = false;

static bool initSd() {
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  // The card shares the display's SPI bus: reuse its SPI object and attach MISO (Part 1, test 08).
  SPIClass *spi = static_cast<Bus_SPI &>(display.panel().driver().bus()).spiInstance();
  if (!spi || !spiAttachMISO(spi->bus(), SD_MISO)) return false;
  const uint32_t clocks[] = {20000000, 10000000, 4000000};
  for (uint32_t f : clocks) if (SD.begin(SD_CS, *spi, f)) return true;
  return false;
}


// What is on the screen: the story, wrapped to the width of the screen, and the part of it that is visible.
static int textSize = 2;          // 2 = big letters (14 per line), 1 = small (28 per line)
static bool followEnd = true;     // keep the newest text in view, until the user scrolls
static int scrollTop = 0;         // first visible line
static int curTokens = 0;
static bool curFinished = true;
static char (*lines)[30] = nullptr;
static constexpr int MAX_LINES = 450;

static int wrapStory(int cols) {
  int nl = 0, col = 0, wl = 0;
  char word[32];
  auto flushWord = [&]() {
    if (!wl) return;
    if (col + wl > cols && col > 0) { lines[nl][col] = 0; if (nl < MAX_LINES - 1) nl++; col = 0; }
    for (int i = 0; i < wl; i++) lines[nl][col++] = word[i];
    wl = 0;
  };
  for (int i = 0; i < storyLen && nl < MAX_LINES - 1; i++) {
    char c = story[i];
    if (c == ' ' || c == '\n') {
      flushWord();
      if (c == '\n') { lines[nl][col] = 0; nl++; col = 0; }
      else if (col > 0 && col < cols) lines[nl][col++] = ' ';
    } else if (wl < cols) word[wl++] = c;
  }
  flushWord();
  lines[nl][col] = 0;
  return nl + 1;
}

static void drawStory() {
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.setCursor(4, 4);
  canvas.print("VerneBot GRU, 4M weights");
  canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  char hdr[48];
  snprintf(hdr, sizeof(hdr), "seed: %s", PROMPTS[promptIndex]);
  canvas.setCursor(4, 16);
  canvas.print(hdr);
  const int cols = textSize == 2 ? 14 : 28, lineH = textSize == 2 ? 18 : 9, top = 30, area = 270;
  const int maxLines = area / lineH;
  int total = wrapStory(cols);
  int lastTop = total > maxLines ? total - maxLines : 0;
  if (followEnd) scrollTop = lastTop;
  if (scrollTop >= lastTop) { scrollTop = lastTop; followEnd = true; }  // scrolled back to the end
  if (scrollTop < 0) scrollTop = 0;
  canvas.setTextSize(textSize);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  for (int i = 0; i < maxLines && scrollTop + i < total; i++) {
    canvas.setCursor(4, top + i * lineH);
    canvas.print(lines[scrollTop + i]);
  }
  if (total > maxLines) {  // a thin bar on the right edge shows where we are in the story
    int barH = max(8, area * maxLines / total), barY = top + (area - barH) * scrollTop / max(1, lastTop);
    canvas.fillRect(169, barY, 3, barH, TFT_DARKGREY);
  }
  canvas.setTextSize(1);
  canvas.setTextColor(curFinished ? TFT_GREEN : TFT_YELLOW, TFT_BLACK);
  if (curFinished && curTokens == 0) snprintf(hdr, sizeof(hdr), "USR1: write the text");
  else snprintf(hdr, sizeof(hdr), "%s %d chars  %.1f ch/s", curFinished ? "done" : "...", curTokens, lastCharsPerSec);
  canvas.setCursor(4, 306);
  canvas.print(hdr);
  canvas.pushSprite(0, 0);
}

static void showStory(bool finished, int tokens) { curFinished = finished; curTokens = tokens; drawStory(); }

// Touch: drag to scroll, tap to change the size of the letters. Returns true if the screen changed.
static bool pollTouch() {
  static bool down = false, moved = false;
  static int32_t y0 = 0, lastY = 0;
  static uint32_t t0 = 0;
  int32_t x, y;
  bool now = display.getTouch(&x, &y);
  bool redraw = false;
  const int lineH = textSize == 2 ? 18 : 9;
  if (now && !down) { y0 = lastY = y; moved = false; t0 = millis(); }
  else if (now && down) {
    if (abs(y - y0) > 8) moved = true;
    int steps = (y - lastY) / lineH;  // a drag down moves the text down: show earlier lines
    if (steps) { scrollTop -= steps; lastY += steps * lineH; followEnd = false; redraw = true; }
  } else if (!now && down && !moved && millis() - t0 < 500) {
    textSize = 3 - textSize;
    followEnd = true;
    redraw = true;
  }
  down = now;
  return redraw;
}

static bool pressed(uint8_t pin) { return digitalRead(pin) == LOW; }

static bool liveTick(const char *piece, int count) {
  size_t n = strlen(piece);
  if (storyLen + (int)n < (int)sizeof(story) - 1) { memcpy(story + storyLen, piece, n); storyLen += n; story[storyLen] = 0; }
  bool touched = pollTouch();
  if (touched || millis() - lastDrawMs > 250) { showStory(false, count); lastDrawMs = millis(); }
  bool b1 = pressed(BTN_USR1);
  if (b1 && !wasPressed1) { wasPressed1 = true; return false; }
  if (!b1) wasPressed1 = false;
  return true;
}

static bool quietTick(const char *, int) { return true; }

static void writeLive() {
  storyLen = 0; story[0] = 0; followEnd = true;
  strncpy(story, PROMPTS[promptIndex], sizeof(story) - 1);
  storyLen = strlen(story);
  Serial.printf("\n--- seed \"%s\" ---\n%s", PROMPTS[promptIndex], PROMPTS[promptIndex]);
  uint32_t us = 0;
  int made = writeText(PROMPTS[promptIndex], WRITE_CHARS, 0.7f, (uint32_t)esp_random() ^ millis(), liveTick, &us);
  Serial.printf("\n--- %d characters, %.2f characters per second ---\n", made, us ? made * 1e6f / us : 0.0f);
  followEnd = false;
  showStory(true, made);
}

static void handleSerial() {
  static char line[160];
  static int len = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c != '\n' && c != '\r') { if (len < (int)sizeof(line) - 1) line[len++] = c; continue; }
    line[len] = 0;
    const char *arg = line[1] == ' ' ? line + 2 : line + 1;
    if (line[0] == 'g') {
      Serial.print("GREEDY ");
      uint32_t us = 0; int made = writeText(arg, 64, 0.0f, 1, quietTick, &us);
      Serial.printf("\nGREEDY done: %d characters, %.2f characters per second\n", made, us ? made * 1e6f / us : 0.0f);
    } else if (line[0] == 't') {
      memset(h, 0, U * 4);
      float *lg = nullptr;
      for (const char *p = arg; *p; p++) { int i = charIndex(*p); if (i >= 0) lg = step(i); }
      if (lg) {
        bool used[V] = {false};
        Serial.print("TOP8 ");
        for (int k = 0; k < 8; k++) {
          int b = -1;
          for (int i = 0; i < V; i++) if (!used[i] && (b < 0 || lg[i] > lg[b])) b = i;
          used[b] = true;
          Serial.print(vocab[b][0] == '\n' ? "\\n" : vocab[b]);
        }
        Serial.println();
      }
    }
    len = 0;
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  Serial.setRxBufferSize(1024);
  delay(800);
  pinMode(BTN_USR1, INPUT_PULLUP);
  pinMode(BTN_USR2, INPUT_PULLUP);
  if (!display.begin<Board_XIAO_1inch47_Touch_Display<LCD_RST_PIN, LCD_BL_PIN>,
                     Config_Seeed_1inch47_Touch_JD9853A>()) { Serial.println(display.lastResult().message); return; }
  canvas.createSprite(display, 172, 320);
  display.attachTouch(touch, display.panel().driver().bus());
  lines = (char (*)[30])ps_malloc(MAX_LINES * 30);
  Serial.printf("PSRAM: %u bytes free\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  const char *err = nullptr;
  if (!initSd()) err = "no SD card";
  else err = loadModel();
  if (err) {
    Serial.printf("ERROR: %s\n", err);
    Serial.println("on the card:");
    listCard();
    canvas.fillScreen(TFT_BLACK);
    canvas.setTextColor(TFT_RED, TFT_BLACK);
    canvas.setTextSize(2);
    canvas.drawCentreString("GRU ERROR", 86, 40, 1);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.drawCentreString(err, 86, 90, 1);
    canvas.drawCentreString("Needs PSRAM=opi and", 86, 120, 1);
    canvas.drawCentreString("/slm/verne_rnn_int8.bin", 86, 134, 1);
    canvas.pushSprite(0, 0);
    for (;;) delay(1000);
  }
  Serial.printf("model loaded: %d characters, GRU of %d units; PSRAM free now %u bytes\n", V, U, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.setTextSize(2);
  canvas.drawCentreString("VERNEBOT", 86, 60, 1);
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.drawCentreString("a GRU of 4 million", 86, 100, 1);
  canvas.drawCentreString("parameters, by character", 86, 114, 1);
  canvas.drawCentreString("USR1: write a text", 86, 150, 1);
  canvas.drawCentreString("USR2: change the seed", 86, 166, 1);
  canvas.drawCentreString("drag: scroll", 86, 200, 1);
  canvas.drawCentreString("tap: letter size", 86, 214, 1);
  canvas.pushSprite(0, 0);
}

void loop() {
  handleSerial();
  bool b1 = pressed(BTN_USR1), b2 = pressed(BTN_USR2);
  if (b2 && !wasPressed2) {
    promptIndex = (promptIndex + 1) % NUM_PROMPTS;
    snprintf(story, sizeof(story), "%s", PROMPTS[promptIndex]);
    storyLen = strlen(story);
    followEnd = true;
    showStory(true, 0);
  }
  wasPressed2 = b2;
  if (pollTouch()) showStory(curFinished, curTokens);
  if (b1 && !wasPressed1) { wasPressed1 = true; writeLive(); }
  if (!b1) wasPressed1 = false;
  delay(20);
}
