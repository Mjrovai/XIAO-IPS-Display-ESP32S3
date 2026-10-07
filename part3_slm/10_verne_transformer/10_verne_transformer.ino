/*
  Part 3, Test 10 - The VerneBot Transformer on the board

  Runs the character-level Transformer of the VerneBot project (4,046,336 parameters, window of 256
  characters, 5 blocks of 4 heads, trained on ten Jules Verne novels) and writes text with it, one
  character at a time. The weights are read from the microSD card into PSRAM as 8-bit integers with one
  scale per row (4.4 MB), made by tools/verne_transformer.py from the project's float16 export. The arithmetic
  is the one of the project's JavaScript demo: pre-norm, learned positions, the exact (erf) GELU, an output
  head tied to the embedding, and a cache of keys and values that never wraps. When the window is full, the
  oldest 64 characters are dropped and the rest are run through the network again at their new positions.

  File on the card, in the folder /slm:   verne_tx_int8.bin
  Build with PSRAM = OPI PSRAM. The weights, the cache, and the tables take about 7 MB of the 8 MB.

  Use it:
    USR1  write a new text (or stop the one being written)
    USR2  change the beginning of the text (the seed), then press USR1
    Touch: drag up or down to scroll, tap to switch between big and small letters.
  Serial (115200): the text as it is written, then a line with the speed. Commands:
    g <seed>   greedy writing of 64 characters; compare with tools/verne_transformer.py greedy --int8
    G <seed>   greedy writing of 300 characters (it crosses the window, so the cache is rebuilt once)
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

static constexpr int V = 123, D = 256, NL = 5, NH = 4, DH = 64, FF = 1024;
static constexpr int WRITE_CHARS = 300;          // how many characters one press of USR1 writes
static constexpr int REFRESH_EVERY = 64;         // characters dropped when the window is full (as in the demo)

// Declared before the functions, so that the Arduino build places its prototypes after them.
struct Mat { int8_t *w; float *s; int rows, cols; };   // out[i] = s[i] * sum_j w[i][j] * x[j]
typedef bool (*TickFn)(const char *piece, int count);   // return false to stop
struct Layer {
  float *ln1g, *ln1b, *bq, *bk, *bv, *bo, *ln2g, *ln2b, *bfc, *bproj;
  Mat wq, wk, wv, wo, fc, proj;
};

// ---- The model in memory ------------------------------------------------------------------------------
static int CTX = 256;                            // the window, read from the file
static char vocab[V][5];                         // each character of the vocabulary, as UTF-8
static uint8_t vocabLen[V];
static float *wte, *wpe, *lnfg, *lnfb;
static Layer layers[NL];
static float *kc, *vc;                           // the cache: NL x CTX x D, never wraps
static float *xv, *nv, *qv, *kv, *vv, *att, *ao, *tmp, *ff1, *logits;
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
  File f = SD.open("/slm/verne_tx_int8.bin", FILE_READ);
  if (!f) return "no /slm/verne_tx_int8.bin";
  char magic[4]; uint32_t hdr[7];
  if (!readAll(f, magic, 4) || memcmp(magic, "VTRF", 4) || !readAll(f, hdr, 28)) return "bad file";
  if (hdr[1] != V || hdr[2] != D || hdr[3] != NL || hdr[4] != NH || hdr[5] != FF || hdr[6] > 256) return "unexpected model shape";
  CTX = hdr[6];
  for (int i = 0; i < V; i++) {
    uint8_t n; if (!readAll(f, &n, 1) || n > 4) return "bad vocabulary";
    vocabLen[i] = n; memset(vocab[i], 0, 5);
    if (!readAll(f, vocab[i], n)) return "bad vocabulary";
  }
  auto fl = [&](int n) { return (float *)ps_malloc((size_t)n * 4); };
  wte = fl(V * D); wpe = fl(CTX * D); lnfg = fl(D); lnfb = fl(D);
  if (!wte || !wpe || !lnfg || !lnfb) return "no PSRAM (use PSRAM=opi)";
  if (!readAll(f, wte, V * D * 4) || !readAll(f, wpe, CTX * D * 4) || !readAll(f, lnfg, D * 4) || !readAll(f, lnfb, D * 4)) return "read failed";
  for (int l = 0; l < NL; l++) {
    Layer &L = layers[l];
    float **v1[10] = {&L.ln1g, &L.ln1b, &L.bq, &L.bk, &L.bv, &L.bo, &L.ln2g, &L.ln2b, &L.bfc, &L.bproj};
    const int sz[10] = {D, D, D, D, D, D, D, D, FF, D};
    for (int i = 0; i < 10; i++) { *v1[i] = fl(sz[i]); if (!*v1[i] || !readAll(f, *v1[i], sz[i] * 4)) return "read failed or out of PSRAM"; }
    if (!readMat(f, L.wq, D, D) || !readMat(f, L.wk, D, D) || !readMat(f, L.wv, D, D) || !readMat(f, L.wo, D, D) ||
        !readMat(f, L.fc, FF, D) || !readMat(f, L.proj, D, FF)) return "read failed or out of PSRAM";
  }
  f.close();
  kc = fl(NL * CTX * D); vc = fl(NL * CTX * D);
  xv = fl(D); nv = fl(FF); qv = fl(D); kv = fl(D); vv = fl(D); att = fl(CTX); ao = fl(D); tmp = fl(D); ff1 = fl(FF); logits = fl(V);
  if (!kc || !vc || !xv || !nv || !qv || !kv || !vv || !att || !ao || !tmp || !ff1 || !logits) return "out of PSRAM for the cache";
  return nullptr;
}

// ---- The Transformer ----------------------------------------------------------------------------------------
static void mv(float *out, const Mat &m, const float *x) {
  for (int i = 0; i < m.rows; i++) {
    const int8_t *w = m.w + (size_t)i * m.cols;
    float a0 = 0, a1 = 0;
    for (int j = 0; j + 1 < m.cols; j += 2) { a0 += (float)w[j] * x[j]; a1 += (float)w[j + 1] * x[j + 1]; }
    out[i] = (a0 + a1) * m.s[i];
  }
}

static void layerNorm(float *o, const float *x, const float *g, const float *b) {
  float m = 0, v = 0;
  for (int i = 0; i < D; i++) m += x[i];
  m /= D;
  for (int i = 0; i < D; i++) { float d = x[i] - m; v += d * d; }
  float inv = 1.0f / sqrtf(v / D + 1e-5f);
  for (int i = 0; i < D; i++) o[i] = (x[i] - m) * inv * g[i] + b[i];
}

static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }   // the exact GELU

static float *step(int c, int pos) {          // one character at position pos in, the scores of the next one out
  for (int i = 0; i < D; i++) xv[i] = wte[c * D + i] + wpe[pos * D + i];
  const float scale = 1.0f / sqrtf((float)DH);
  for (int l = 0; l < NL; l++) {
    Layer &L = layers[l];
    layerNorm(nv, xv, L.ln1g, L.ln1b);
    mv(qv, L.wq, nv); mv(kv, L.wk, nv); mv(vv, L.wv, nv);
    for (int i = 0; i < D; i++) { qv[i] += L.bq[i]; kv[i] += L.bk[i]; vv[i] += L.bv[i]; }
    float *kl = kc + (size_t)l * CTX * D, *vl = vc + (size_t)l * CTX * D;
    memcpy(kl + (size_t)pos * D, kv, D * 4);
    memcpy(vl + (size_t)pos * D, vv, D * 4);
    for (int h = 0; h < NH; h++) {
      const int off = h * DH;
      float mx = -1e30f;
      for (int t = 0; t <= pos; t++) {
        const float *kt = kl + (size_t)t * D + off;
        float s = 0;
        for (int i = 0; i < DH; i++) s += qv[off + i] * kt[i];
        s *= scale; att[t] = s; if (s > mx) mx = s;
      }
      float sum = 0;
      for (int t = 0; t <= pos; t++) { att[t] = expf(att[t] - mx); sum += att[t]; }
      float inv = 1.0f / sum;
      for (int i = 0; i < DH; i++) ao[off + i] = 0;
      for (int t = 0; t <= pos; t++) {
        const float *vt = vl + (size_t)t * D + off, w = att[t] * inv;
        for (int i = 0; i < DH; i++) ao[off + i] += w * vt[i];
      }
    }
    mv(tmp, L.wo, ao);
    for (int i = 0; i < D; i++) xv[i] += tmp[i] + L.bo[i];                 // first residual
    layerNorm(nv, xv, L.ln2g, L.ln2b);
    mv(ff1, L.fc, nv);
    for (int i = 0; i < FF; i++) ff1[i] = gelu(ff1[i] + L.bfc[i]);         // the bias goes in before the GELU
    mv(tmp, L.proj, ff1);
    for (int i = 0; i < D; i++) xv[i] += tmp[i] + L.bproj[i];              // second residual
  }
  layerNorm(nv, xv, lnfg, lnfb);
  for (int i = 0; i < V; i++) {                                            // the head is the embedding table
    float s = 0; const float *e = wte + i * D;
    for (int j = 0; j < D; j++) s += e[j] * nv[j];
    logits[i] = s;
  }
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


// Called while the cache is being rebuilt, to show it on the screen. Return false to stop.
static bool (*rebuildHook)(int done, int total) = nullptr;
static uint32_t rebuildUs = 0;
static int rebuilds = 0;

// Write count characters after the seed. Same logic as the demo: the seed keeps at most CTX-1 characters, and when
// the window is full the newest CTX-REFRESH_EVERY characters are run again at positions 0, 1, 2, ...
static int writeText(const char *seed, int count, float temperature, uint32_t seedValue, TickFn tick, uint32_t *usOut) {
  int *hist = (int *)ps_malloc((CTX + 2) * sizeof(int));
  int n = 0;
  for (const char *p = seed; *p; p++) { int i = charIndex(*p); if (i >= 0) { if (n == CTX - 1) { memmove(hist, hist + 1, (CTX - 2) * sizeof(int)); n--; } hist[n++] = i; } }
  if (n == 0) hist[n++] = charIndex(' ');
  float *lg = nullptr;
  for (int p = 0; p < n; p++) lg = step(hist[p], p);
  int pos = n;
  rngS = seedValue ? seedValue : 1;
  uint32_t us = 0; int made = 0; rebuildUs = 0; rebuilds = 0;
  for (int i = 0; i < count; i++) {
    uint32_t t0 = micros();
    int c = sampleChar(lg, temperature);
    Serial.write((const uint8_t *)vocab[c], vocabLen[c]);
    if (tick && !tick(screenForm(c), i + 1)) break;
    if (n == CTX) { memmove(hist, hist + 1, (CTX - 1) * sizeof(int)); n--; }
    hist[n++] = c;
    if (pos + 1 >= CTX) {
      int keep = max(1, CTX - REFRESH_EVERY);
      memmove(hist, hist + n - keep, keep * sizeof(int)); n = keep;
      uint32_t r0 = micros(); bool ok = true;
      for (int p = 0; p < keep; p++) {
        lg = step(hist[p], p);
        if (rebuildHook && !rebuildHook(p + 1, keep)) { ok = false; break; }
      }
      uint32_t rd = micros() - r0;
      rebuildUs += rd; rebuilds++; pos = keep;
      us += micros() - t0 - rd;
      if (!ok) { made++; break; }
    } else {
      lg = step(c, pos); pos++;
      us += micros() - t0;
    }
    made++;
    lastCharsPerSec = us ? made * 1e6f / us : 0;
  }
  free(hist);
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
static char footerNote[40] = "";      // shown instead of the speed, for example while the cache is rebuilt

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
  canvas.setCursor(10, 4);   // a little to the right: the rounded corner of the screen hides the first letter
  canvas.print("VerneBot Transformer, 4M");
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
  else if (footerNote[0]) snprintf(hdr, sizeof(hdr), "%s", footerNote);
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

static bool rebuildTick(int done, int total) {
  if (done % 8 == 0 || done == total) { snprintf(footerNote, sizeof(footerNote), "rebuilding cache %d/%d", done, total); showStory(false, curTokens); }
  bool b1 = pressed(BTN_USR1);
  if (b1 && !wasPressed1) { wasPressed1 = true; return false; }
  if (!b1) wasPressed1 = false;
  return true;
}

static void writeLive() {
  storyLen = 0; story[0] = 0; followEnd = true;
  strncpy(story, PROMPTS[promptIndex], sizeof(story) - 1);
  storyLen = strlen(story);
  Serial.printf("\n--- seed \"%s\" ---\n%s", PROMPTS[promptIndex], PROMPTS[promptIndex]);
  uint32_t us = 0;
  rebuildHook = rebuildTick;
  int made = writeText(PROMPTS[promptIndex], WRITE_CHARS, 0.7f, (uint32_t)esp_random() ^ millis(), liveTick, &us);
  rebuildHook = nullptr; footerNote[0] = 0;
  Serial.printf("\n--- %d characters, %.2f characters per second, %d cache rebuilds of %.1f s ---\n", made, us ? made * 1e6f / us : 0.0f, rebuilds, rebuilds ? rebuildUs / 1e6f / rebuilds : 0.0f);
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
    } else if (line[0] == 'G') {
      Serial.print("GREEDY ");
      uint32_t us = 0; int made = writeText(arg, 300, 0.0f, 1, quietTick, &us);
      Serial.printf("\nGREEDY done: %d characters, %.2f characters per second, %d cache rebuilds of %.1f s\n", made, us ? made * 1e6f / us : 0.0f, rebuilds, rebuilds ? rebuildUs / 1e6f / rebuilds : 0.0f);
    } else if (line[0] == 't') {
      float *lg = nullptr; int pos = 0;
      for (const char *p = arg; *p; p++) { int i = charIndex(*p); if (i >= 0 && pos < CTX - 1) lg = step(i, pos++); }
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
    canvas.drawCentreString("TX ERROR", 86, 40, 1);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.drawCentreString(err, 86, 90, 1);
    canvas.drawCentreString("Needs PSRAM=opi and", 86, 120, 1);
    canvas.drawCentreString("/slm/verne_tx_int8.bin", 86, 134, 1);
    canvas.pushSprite(0, 0);
    for (;;) delay(1000);
  }
  Serial.printf("model loaded: %d characters, %d blocks, window %d; PSRAM free now %u bytes\n", V, NL, CTX, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.setTextSize(2);
  canvas.drawCentreString("VERNEBOT", 86, 60, 1);
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.drawCentreString("a Transformer of 4 million", 86, 100, 1);
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
