# Part 3: a tiny language model on the board

A language model with 260,000 parameters generates short children's stories on the
ESP32-S3, one token at a time, and the text appears on the 1.47" screen as it is
written. It is far too small to be useful, and that is the point of the lesson: it
shows, in a few hundred lines of C++, everything a large language model does, with a model
small enough to read, run, and check by hand.

![The board showing a story it generated, in big letters, with 512 tokens done at 12.5 tokens per second](../images/slm_story_screen.jpg)

*The screen after a story of 512 tokens. The last line shows the speed measured while
the model was running.*

## What you need

- The 1.47" touch board, with **PSRAM turned on**: in the Arduino IDE, Tools, PSRAM,
  OPI PSRAM (with `arduino-cli`, the board string ends in `:PSRAM=opi`). The default is
  *Disabled*, and then the sketch stops with a message, because the weights do not fit
  in the internal RAM.
- A microSD card (FAT) with a folder `slm` holding two files, in the root of the card:
  `/slm/stories260K.bin` (the weights) and `/slm/tok512.bin` (the tokenizer).

The two files are Andrej Karpathy's, from
[huggingface.co/karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas), folder
`stories260K`. They are not in this repository; download them from there. The copies used
for the measurements below had these sizes and SHA-256 sums:

| File | Bytes | SHA-256 |
|---|---|---|
| `stories260K.bin` | 1,056,540 | `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696` |
| `tok512.bin` | 6,227 | `037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312` |

## Credits

The model, the tokenizer, and the file format come from Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c) project (MIT license) and its
[tinyllamas](https://huggingface.co/karpathy/tinyllamas) models, trained on the
TinyStories data set. The sketch in this repository is a port of llama2.c's inference code
to the Arduino environment, and the Python check in `tools/slm_reference.py` is a second,
independent implementation of the same model. Both were written by Claude Sonnet 5.5
(Anthropic) under the author's direction, and the author ran, checked, and photographed
them on the board.

## The model

Read from the header of the file:

| | |
|---|---|
| Embedding size (`dim`) | 64 |
| Layers | 5 |
| Attention heads | 8 (4 for the keys and values) |
| Feed-forward size | 172 |
| Vocabulary | 512 tokens |
| Context | 512 tokens |
| Parameters | 264,128, stored as 32-bit floats |

The file is 1,056,540 bytes: 28 bytes of header and 1,056,512 bytes of weights, which
is exactly 264,128 values of 4 bytes. The match with the count we computed from the header
is how we know the file format was read correctly.

**Memory.** After loading, 1,824,316 bytes of the PSRAM are in use (6.44 MB of the 8.26 MB
that the board reports are still free). The weights are 1.06 MB; the cache of the attention
layers (the keys and values of every position, 5 layers by 512 positions) is 655,360 bytes;
the rest is the tokenizer and the working buffers. The sketch leaves most of the internal heap free (about 340 KB at the benchmark).

## Test 07: tell a story

The sketch loads the weights from the card into PSRAM and generates a story. The first
token is the "beginning of text" mark; then each step runs the whole network once to get
a score for each of the 512 tokens, picks one, and feeds it back in.

**Controls**

- **USR1** tells a new story, or stops the one being told.
- **USR2** changes the beginning of the story (the *prompt*): none, "Once upon a time",
  "One day, a little girl", "Tom and Lily", or "The big dog". The new beginning appears in
  big letters; press USR1 to continue it.
- **Touch:** drag up or down to scroll, and tap to switch between big letters (14 per
  line) and small letters (28 per line). While a story is being written the screen follows
  the newest text; scroll up to read, and scroll back to the end to follow again.
- **Serial** (115200): the story as text, then a line with the speed. Two commands: `b`
  runs a benchmark, and `g <text>` decodes greedily and prints the token numbers (used for
  the check below).

The words are chosen with a temperature of 0.9 and top-p of 0.9: the model draws among the
most likely tokens, so every story is different.

<!-- sketch: part3_slm/07_slm_stories/07_slm_stories.ino -->
**Sketch:** [`part3_slm/07_slm_stories/07_slm_stories.ino`](https://github.com/Mjrovai/XIAO-IPS-Display-ESP32S3/blob/main/part3_slm/07_slm_stories/07_slm_stories.ino)

```cpp
/*
  Part 3, Test 07 - A tiny language model on the board (stories260K)

  Runs Andrej Karpathy's stories260K model: 260 thousand parameters, 5 layers, trained on
  very short children's stories, with a 512-token vocabulary. The weights (1 MB, float32)
  are read from the microSD card into PSRAM, and the story is generated one token at a
  time on the ESP32-S3 itself. The inference code is a port of llama2.c (Andrej Karpathy,
  MIT license, https://github.com/karpathy/llama2.c) to the Arduino environment.

  Files on the card, in the folder /slm:  stories260K.bin  and  tok512.bin
  (from https://huggingface.co/karpathy/tinyllamas, folder stories260K).

  Build with PSRAM = OPI PSRAM (the FQBN ends in :PSRAM=opi). The weights and the cache
  of the attention layers take about 1.7 MB, far more than the internal RAM.

  Use it:
    USR1  start a new story (or stop the one being told)
    USR2  change the beginning of the story (the prompt), then press USR1
    Touch drag up or down: scroll the story. Tap: switch between big and small letters.
  Serial (115200): the story as text, then a line with tokens per second.
    b          benchmark: 200 tokens, fixed seed, nothing drawn on the screen
    g <text>   greedy decoding (temperature 0) of 64 tokens from <text>, printing the
               token numbers; used to compare with tools/slm_reference.py
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

// Declared here, before the functions, so that the Arduino build can place its prototypes after them.
struct Stats { int tokens; float tokPerSec; uint32_t forwardUs; };
typedef bool (*TickFn)(const char *piece, int tokensSoFar);  // return false to stop
struct ProbIndex { float prob; int index; };

// ---- The model (checkpoint format of llama2.c) -------------------------------------------------
struct Config { int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len; };
static Config cfg;
static float *weightsBlock = nullptr;
static size_t weightsBytes = 0;
static float *tokEmb, *rmsAtt, *wq, *wk, *wv, *wo, *rmsFfn, *w1, *w2, *w3, *rmsFinal, *wcls;

// Run state: the activations and the cache of keys and values of every layer.
static float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits, *keyCache, *valueCache;

// ---- Tokenizer ---------------------------------------------------------------------------------
static char **vocab = nullptr;
static float *vocabScores = nullptr;
static int maxTokenLength = 0;

static bool readAll(File &f, void *dst, size_t n) { return f.read((uint8_t *)dst, n) == (int)n; }

static bool loadTokenizer(const char *path) {
  File f = SD.open(path, FILE_READ);
  if (!f) return false;
  if (!readAll(f, &maxTokenLength, 4)) return false;
  vocab = (char **)ps_malloc(cfg.vocab_size * sizeof(char *));
  vocabScores = (float *)ps_malloc(cfg.vocab_size * sizeof(float));
  for (int i = 0; i < cfg.vocab_size; i++) {
    int len;
    if (!readAll(f, &vocabScores[i], 4) || !readAll(f, &len, 4)) return false;
    vocab[i] = (char *)ps_malloc(len + 1);
    if (!readAll(f, vocab[i], len)) return false;
    vocab[i][len] = 0;
  }
  f.close();
  return true;
}

static int strLookup(const char *s) {
  for (int i = 0; i < cfg.vocab_size; i++)
    if (!strcmp(vocab[i], s)) return i;
  return -1;
}

// Text to tokens: BOS, then the letters, then the best pairs merged until none is left.
static int encode(const char *text, int *tokens) {
  int n = 0;
  tokens[n++] = 1;  // BOS
  if (text[0]) tokens[n++] = strLookup(" ");
  char buf[8];
  int len = 0;
  for (const char *c = text; *c; c++) {
    if ((*c & 0xC0) != 0x80) len = 0;
    buf[len++] = *c;
    buf[len] = 0;
    if (*(c + 1) && (*(c + 1) & 0xC0) == 0x80 && len < 4) continue;
    int id = strLookup(buf);
    if (id != -1) tokens[n++] = id;
    else for (int i = 0; i < len; i++) tokens[n++] = (unsigned char)buf[i] + 3;  // byte fallback
    len = 0;
  }
  char pair[64];
  for (;;) {
    float best = -1e10f;
    int bestId = -1, bestIdx = -1;
    for (int i = 0; i < n - 1; i++) {
      snprintf(pair, sizeof(pair), "%s%s", vocab[tokens[i]], vocab[tokens[i + 1]]);
      int id = strLookup(pair);
      if (id != -1 && vocabScores[id] > best) { best = vocabScores[id]; bestId = id; bestIdx = i; }
    }
    if (bestIdx == -1) break;
    tokens[bestIdx] = bestId;
    for (int i = bestIdx + 1; i < n - 1; i++) tokens[i] = tokens[i + 1];
    n--;
  }
  return n;
}

// Token to text. Returns the piece to print (a one-character string for a raw byte).
static const char *decode(int prev, int token) {
  static char one[2];
  const char *piece = vocab[token];
  if (prev == 1 && piece[0] == ' ') piece++;  // no space after the start of the text
  unsigned int byteVal;
  if (sscanf(piece, "<0x%02X>", &byteVal) == 1) { one[0] = (char)byteVal; one[1] = 0; return one; }
  return piece;
}

// ---- The transformer -----------------------------------------------------------------------------
static void rmsnorm(float *o, const float *in, const float *weight, int size) {
  float ss = 0;
  for (int j = 0; j < size; j++) ss += in[j] * in[j];
  ss = 1.0f / sqrtf(ss / size + 1e-5f);
  for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * in[j]);
}

static void softmax(float *v, int size) {
  float m = v[0];
  for (int i = 1; i < size; i++) if (v[i] > m) m = v[i];
  float sum = 0;
  for (int i = 0; i < size; i++) { v[i] = expf(v[i] - m); sum += v[i]; }
  for (int i = 0; i < size; i++) v[i] /= sum;
}

// out (d) = w (d, n) times in (n)
static void matmul(float *out, const float *in, const float *w, int n, int d) {
  for (int i = 0; i < d; i++) {
    float val = 0;
    const float *row = w + (size_t)i * n;
    for (int j = 0; j < n; j++) val += row[j] * in[j];
    out[i] = val;
  }
}

static float *forward(int token, int pos) {
  const int dim = cfg.dim, hidden = cfg.hidden_dim, hs = dim / cfg.n_heads;
  const int kvDim = (dim * cfg.n_kv_heads) / cfg.n_heads, kvMul = cfg.n_heads / cfg.n_kv_heads;
  memcpy(x, tokEmb + (size_t)token * dim, dim * sizeof(float));
  for (int l = 0; l < cfg.n_layers; l++) {
    rmsnorm(xb, x, rmsAtt + l * dim, dim);
    size_t loff = (size_t)l * cfg.seq_len * kvDim;
    float *k = keyCache + loff + (size_t)pos * kvDim;
    float *v = valueCache + loff + (size_t)pos * kvDim;
    matmul(q, xb, wq + (size_t)l * dim * dim, dim, dim);
    matmul(k, xb, wk + (size_t)l * dim * kvDim, dim, kvDim);
    matmul(v, xb, wv + (size_t)l * dim * kvDim, dim, kvDim);
    // Rotary position encoding: rotate each pair of values of q (and of k) by an angle that grows with pos.
    for (int i = 0; i < dim; i += 2) {
      int headDim = i % hs;
      float freq = 1.0f / powf(10000.0f, headDim / (float)hs);
      float val = pos * freq, fcr = cosf(val), fci = sinf(val);
      int rotn = i < kvDim ? 2 : 1;
      for (int r = 0; r < rotn; r++) {
        float *vec = r == 0 ? q : k;
        float v0 = vec[i], v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }
    // Attention, head by head: compare q with every key so far, then mix the values.
    for (int h = 0; h < cfg.n_heads; h++) {
      float *qh = q + h * hs;
      float *a = att + h * cfg.seq_len;
      for (int t = 0; t <= pos; t++) {
        float *kt = keyCache + loff + (size_t)t * kvDim + (h / kvMul) * hs;
        float score = 0;
        for (int i = 0; i < hs; i++) score += qh[i] * kt[i];
        a[t] = score / sqrtf((float)hs);
      }
      softmax(a, pos + 1);
      float *xbh = xb + h * hs;
      memset(xbh, 0, hs * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        float *vt = valueCache + loff + (size_t)t * kvDim + (h / kvMul) * hs;
        for (int i = 0; i < hs; i++) xbh[i] += a[t] * vt[i];
      }
    }
    matmul(xb2, xb, wo + (size_t)l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++) x[i] += xb2[i];
    // Feed-forward block with the SwiGLU activation.
    rmsnorm(xb, x, rmsFfn + l * dim, dim);
    matmul(hb, xb, w1 + (size_t)l * dim * hidden, dim, hidden);
    matmul(hb2, xb, w3 + (size_t)l * dim * hidden, dim, hidden);
    for (int i = 0; i < hidden; i++) hb[i] = hb[i] * (1.0f / (1.0f + expf(-hb[i]))) * hb2[i];
    matmul(xb, hb, w2 + (size_t)l * dim * hidden, hidden, dim);
    for (int i = 0; i < dim; i++) x[i] += xb[i];
  }
  rmsnorm(x, x, rmsFinal, dim);
  matmul(logits, x, wcls, dim, cfg.vocab_size);
  return logits;
}

// ---- Choosing the next token -----------------------------------------------------------------------
static uint64_t rngState = 1;
static uint32_t randU32() {
  rngState ^= rngState >> 12; rngState ^= rngState << 25; rngState ^= rngState >> 27;
  return (rngState * 0x2545F4914F6CDD1DULL) >> 32;
}
static float randF32() { return (randU32() >> 8) / 16777216.0f; }

static float lastTokPerSec = 0;
static ProbIndex *probIndex = nullptr;
static int cmpProb(const void *a, const void *b) {
  float pa = ((const ProbIndex *)a)->prob, pb = ((const ProbIndex *)b)->prob;
  return pa > pb ? -1 : (pa < pb ? 1 : 0);
}

static int sampleToken(float *lg, float temperature, float topp) {
  const int n = cfg.vocab_size;
  if (temperature == 0.0f) {
    int best = 0;
    for (int i = 1; i < n; i++) if (lg[i] > lg[best]) best = i;
    return best;
  }
  for (int i = 0; i < n; i++) lg[i] /= temperature;
  softmax(lg, n);
  float coin = randF32();
  if (topp <= 0 || topp >= 1) {  // plain sampling
    float cdf = 0;
    for (int i = 0; i < n; i++) { cdf += lg[i]; if (coin < cdf) return i; }
    return n - 1;
  }
  int n0 = 0;
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) if (lg[i] >= cutoff) { probIndex[n0].index = i; probIndex[n0].prob = lg[i]; n0++; }
  qsort(probIndex, n0, sizeof(ProbIndex), cmpProb);
  float cum = 0;
  int last = n0 - 1;
  for (int i = 0; i < n0; i++) { cum += probIndex[i].prob; if (cum > topp) { last = i; break; } }
  float r = coin * cum, cdf = 0;
  for (int i = 0; i <= last; i++) { cdf += probIndex[i].prob; if (r < cdf) return probIndex[i].index; }
  return probIndex[last].index;
}

// ---- Loading --------------------------------------------------------------------------------------------
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

// Print the folders in the root of the card and the files of /slm, to see what is there when the model is not found.
static void listCard() {
  File d = SD.open("/");
  if (d && d.isDirectory())
    for (File e = d.openNextFile(); e; e = d.openNextFile()) Serial.printf("  /%s%s\n", e.name(), e.isDirectory() ? "/" : "");
  File m = SD.open("/slm");
  if (m && m.isDirectory())
    for (File e = m.openNextFile(); e; e = m.openNextFile()) Serial.printf("  /slm/%s  %u\n", e.name(), (unsigned)e.size());
}

static const char *loadModel() {
  File f = SD.open("/slm/stories260K.bin", FILE_READ);
  if (!f) return "no /slm/stories260K.bin";
  if (!readAll(f, &cfg, sizeof(cfg))) return "bad header";
  const bool shared = cfg.vocab_size > 0;
  cfg.vocab_size = abs(cfg.vocab_size);
  weightsBytes = f.size() - sizeof(cfg);
  weightsBlock = (float *)ps_malloc(weightsBytes);
  if (!weightsBlock) return "no PSRAM (use PSRAM=opi)";
  size_t got = 0;
  while (got < weightsBytes) {  // read in pieces so the watchdog is fed
    int n = f.read((uint8_t *)weightsBlock + got, min((size_t)16384, weightsBytes - got));
    if (n <= 0) return "read failed";
    got += n;
  }
  f.close();
  const int dim = cfg.dim, hidden = cfg.hidden_dim, L = cfg.n_layers, hs = dim / cfg.n_heads;
  const int kvDim = (dim * cfg.n_kv_heads) / cfg.n_heads;
  float *p = weightsBlock;
  tokEmb = p; p += (size_t)cfg.vocab_size * dim;
  rmsAtt = p; p += (size_t)L * dim;
  wq = p; p += (size_t)L * dim * dim;
  wk = p; p += (size_t)L * dim * kvDim;
  wv = p; p += (size_t)L * dim * kvDim;
  wo = p; p += (size_t)L * dim * dim;
  rmsFfn = p; p += (size_t)L * dim;
  w1 = p; p += (size_t)L * dim * hidden;
  w2 = p; p += (size_t)L * hidden * dim;
  w3 = p; p += (size_t)L * dim * hidden;
  rmsFinal = p; p += dim;
  p += (size_t)cfg.seq_len * hs;  // the two tables of the old position encoding, not used
  wcls = shared ? tokEmb : p;
  // Run state (the keys and values of every position and layer are what the cache keeps).
  x = (float *)ps_malloc(dim * 4); xb = (float *)ps_malloc(dim * 4); xb2 = (float *)ps_malloc(dim * 4);
  hb = (float *)ps_malloc(hidden * 4); hb2 = (float *)ps_malloc(hidden * 4); q = (float *)ps_malloc(dim * 4);
  att = (float *)ps_malloc((size_t)cfg.n_heads * cfg.seq_len * 4);
  logits = (float *)ps_malloc(cfg.vocab_size * 4);
  keyCache = (float *)ps_malloc((size_t)L * cfg.seq_len * kvDim * 4);
  valueCache = (float *)ps_malloc((size_t)L * cfg.seq_len * kvDim * 4);
  probIndex = (ProbIndex *)ps_malloc(cfg.vocab_size * sizeof(ProbIndex));
  if (!keyCache || !valueCache || !att || !probIndex || !logits) return "out of memory";
  if (!loadTokenizer("/slm/tok512.bin")) return "no /slm/tok512.bin";
  return nullptr;
}

// ---- Telling a story ------------------------------------------------------------------------------------
static Stats generate(const char *prompt, int steps, float temperature, float topp, uint64_t seed,
                      TickFn tick, bool printIds) {
  rngState = seed;
  int *promptTokens = (int *)ps_malloc((strlen(prompt) + 4) * sizeof(int));
  int nPrompt = encode(prompt, promptTokens);
  if (steps > cfg.seq_len) steps = cfg.seq_len;
  int token = promptTokens[0], prev = 0, pos = 0, produced = 0;
  uint32_t forwardUs = 0;
  while (pos < steps) {
    uint32_t t0 = micros();
    float *lg = forward(token, pos);
    int next = pos < nPrompt - 1 ? promptTokens[pos + 1] : sampleToken(lg, temperature, topp);
    forwardUs += micros() - t0;
    pos++;
    if (next == 1 && pos >= nPrompt) break;  // BOS after the prompt: the story ended
    produced++;
    lastTokPerSec = forwardUs ? produced * 1e6f / forwardUs : 0.0f;  // the running speed, shown on the screen
    if (printIds) Serial.printf("%d ", next);
    const char *piece = decode(token, next);
    if (tick && !tick(piece, produced)) break;
    prev = token;
    token = next;
  }
  free(promptTokens);
  (void)prev;
  return {produced, forwardUs ? produced * 1e6f / forwardUs : 0.0f, forwardUs};
}

// ---- Screen and buttons ----------------------------------------------------------------------------------
static const char *PROMPTS[] = {"", "Once upon a time", "One day, a little girl", "Tom and Lily", "The big dog"};
static constexpr int NUM_PROMPTS = 5;
static int promptIndex = 0;
static char story[4096];
static int storyLen = 0;
static uint32_t lastDrawMs = 0;
static bool wasPressed1 = false, wasPressed2 = false;

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
  canvas.print("stories260K on the ESP32-S3");
  canvas.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  char hdr[48];
  snprintf(hdr, sizeof(hdr), "prompt: %s", PROMPTS[promptIndex][0] ? PROMPTS[promptIndex] : "(none)");
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
  if (curFinished && curTokens == 0) snprintf(hdr, sizeof(hdr), "USR1: tell the story");
  else snprintf(hdr, sizeof(hdr), "%s %d tokens  %.1f tok/s", curFinished ? "done" : "...", curTokens, lastTokPerSec);
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

// Called after every token: collect the text, redraw about four times a second, watch USR1 to stop.
static int tokensShown = 0;
static bool liveTick(const char *piece, int tokens) {
  size_t n = strlen(piece);
  if (storyLen + (int)n < (int)sizeof(story) - 1) { memcpy(story + storyLen, piece, n); storyLen += n; story[storyLen] = 0; }
  Serial.print(piece);
  tokensShown = tokens;
  bool touched = pollTouch();
  if (touched || millis() - lastDrawMs > 250) { showStory(false, tokens); lastDrawMs = millis(); }
  bool b1 = pressed(BTN_USR1);
  if (b1 && !wasPressed1) { wasPressed1 = true; return false; }
  if (!b1) wasPressed1 = false;
  return true;
}

static bool quietTick(const char *, int) { return true; }

static void tellStory() {
  storyLen = 0; story[0] = 0;
  followEnd = true;
  const char *prompt = PROMPTS[promptIndex];
  strncpy(story, prompt, sizeof(story) - 1);
  storyLen = strlen(story);
  Serial.printf("\n--- story, prompt \"%s\" ---\n", prompt);
  Stats s = generate(prompt, cfg.seq_len, 0.9f, 0.9f, (uint64_t)esp_random() << 16 ^ millis(), liveTick, false);
  lastTokPerSec = s.tokPerSec;
  Serial.printf("\n--- %d tokens, %.1f tok/s ---\n", s.tokens, s.tokPerSec);
  followEnd = false;  // when it is done, stay where the reader is
  showStory(true, s.tokens);
}

static void benchmark() {
  Stats s = generate("", 200, 0.9f, 0.9f, 1, quietTick, false);
  Serial.printf("BENCH tokens %d forward %lu ms  %.2f tok/s  free PSRAM %u  free heap %u\n", s.tokens,
                (unsigned long)(s.forwardUs / 1000), s.tokPerSec, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                (unsigned)ESP.getFreeHeap());
}

static void handleSerial() {
  static char line[160];
  static int len = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c != '\n' && c != '\r') { if (len < (int)sizeof(line) - 1) line[len++] = c; continue; }
    line[len] = 0;
    if (line[0] == 'b') benchmark();
    else if (line[0] == 'g') {
      Serial.print("IDS ");
      Stats s = generate(line[1] == ' ' ? line + 2 : line + 1, 64 + 0, 0.0f, 0.0f, 1, quietTick, true);
      Serial.printf("\nGREEDY tokens %d %.2f tok/s\n", s.tokens, s.tokPerSec);
    }
    len = 0;
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
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
    canvas.drawCentreString("SLM ERROR", 86, 40, 1);
    canvas.setTextSize(1);
    canvas.setTextColor(TFT_WHITE, TFT_BLACK);
    canvas.drawCentreString(err, 86, 90, 1);
    canvas.drawCentreString("Needs PSRAM=opi and", 86, 120, 1);
    canvas.drawCentreString("/slm/stories260K.bin", 86, 134, 1);
    canvas.drawCentreString("/slm/tok512.bin on the card", 86, 148, 1);
    canvas.pushSprite(0, 0);
    for (;;) delay(1000);
  }
  Serial.printf("model: dim %d, hidden %d, layers %d, heads %d, kv heads %d, vocab %d, context %d; weights %u bytes\n",
                cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len,
                (unsigned)weightsBytes);
  Serial.printf("PSRAM after loading: %u bytes free\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  canvas.fillScreen(TFT_BLACK);
  canvas.setTextColor(TFT_CYAN, TFT_BLACK);
  canvas.setTextSize(2);
  canvas.drawCentreString("STORIES", 86, 60, 1);
  canvas.setTextSize(1);
  canvas.setTextColor(TFT_WHITE, TFT_BLACK);
  canvas.drawCentreString("260K parameters", 86, 100, 1);
  canvas.drawCentreString("USR1: tell a story", 86, 140, 1);
  canvas.drawCentreString("USR2: change the start", 86, 156, 1);
  canvas.drawCentreString("drag: scroll", 86, 190, 1);
  canvas.drawCentreString("tap: letter size", 86, 204, 1);
  canvas.pushSprite(0, 0);
}

void loop() {
  handleSerial();
  bool b1 = pressed(BTN_USR1), b2 = pressed(BTN_USR2);
  if (b2 && !wasPressed2) {
    promptIndex = (promptIndex + 1) % NUM_PROMPTS;
    // Show the new beginning in big letters, where the story will appear.
    snprintf(story, sizeof(story), "%s", PROMPTS[promptIndex][0] ? PROMPTS[promptIndex] : "(no beginning: the model makes it up)");
    storyLen = strlen(story);
    followEnd = true;
    showStory(true, 0);
  }
  wasPressed2 = b2;
  if (pollTouch()) showStory(curFinished, curTokens);
  if (b1 && !wasPressed1) { wasPressed1 = true; tellStory(); }
  if (!b1) wasPressed1 = false;
  delay(20);
}
```
<!-- /sketch -->

#### How it works

- **The tokenizer** turns text into numbers. The 512 tokens are letters and common pieces
  of words (the file `tok512.bin` has them with a score each). The text is cut into letters
  and then the best-scoring pairs are merged until none is left. The reverse is a table
  lookup. A token such as `<0x0A>` is a raw byte (here, a new line).
- **One step of the network** (`forward`): look up the embedding of the token (64 numbers),
  then for each of the 5 layers: normalize; compute the query, key, and value; rotate them by
  an angle that depends on the position (RoPE); store the key and value in the cache; let
  each of the 8 heads compare its query with all the keys so far and mix the values by
  those scores (attention); add the result back; then a feed-forward block with the SwiGLU
  activation, added back too. At the end, the 64 numbers are multiplied by the embedding
  table to give 512 scores.
- **The cache** is why a step does not recompute the past: the keys and values of earlier
  positions are kept. It is also why a step takes longer the further the story goes, as
  the attention has more positions to look at.
- **Choosing** the token: with a temperature of 0 it is always the highest score; with a
  higher temperature it is drawn from the softmax of the scores, cutting off the unlikely
  tail (top-p).
- **Everything is plain float code**, no libraries and no optimization of the inner loops,
  to be readable.

## Is it right? A check against a second implementation

A port can run and still be wrong, and a 260K-parameter model would write odd stories
either way. So the check does not look at the text. `tools/slm_reference.py` runs the same
model in Python (numpy), written separately, with greedy decoding (always the highest
score). The command `g ` on the board does the same.

For each of the five beginnings of the sketch (none, "Once upon a time", "One day, a
little girl", "Tom and Lily", and "The big dog"), the first 64 tokens were **the same on the
board and in numpy, 64 of 64 in every case**. This checks the whole path (file format,
tokenizer, attention, the cache, and the feed-forward block) for those five prompts. It does
not check the sampling (temperature and top-p), since greedy decoding does not use it, and
it does not look past 64 tokens.

Greedy decoding also shows a known weakness of small models: with "The big dog" it falls
into a loop and repeats "He liked to play with his ball." This is why the sketch draws the
words at random instead.

```bash
python3 tools/slm_reference.py ~/datasets/slm/stories260K.bin ~/datasets/slm/tok512.bin "" --steps 64
```

## Speed

All numbers are the time spent in the network only, not drawing on the screen.

| What | Tokens per second |
|---|---|
| Greedy, the first 64 tokens | 30.5 |
| Benchmark (`b`): 200 tokens, fixed seed | 21.7 |
| A full story of 512 tokens | 12.5 to 12.6 |

The speed falls as the story grows, because each step attends to all the positions before
it. The first token needs only one position; the 512th needs 512. We did not try to make it
faster: the weights are read from PSRAM, which is slower than the internal RAM; the code does
not use the second core; and the weights are 32-bit floats, where 8-bit weights would be
four times smaller. These are the places to look.

## What the stories are like

A model this small writes grammatical sentences in the style of a children's story and
loses the thread within a paragraph. This is a story the board wrote from the empty
prompt (the model's own words, with an ordinary run of the sketch):

> Once upon a time, there was a little girl named Lily. She loved to jump, but it was her
> favorite ice cream. One day, Lily was very impressive and decided to clean up her room.
> She asked her mommy, "Why are you sad?"

The first sentence is a typical opening for this kind of story. After that
the sentences are fine one by one, but they do not follow from each other ("it was her
favorite ice cream"; "very impressive"). Larger models with the same design do much better;
this one shows the machinery and its limits. The training data set is TinyStories, made of
very simple stories, which is why such a small model can write anything readable at all.

## Next steps

1. **Make it faster.** Try the second core, keep the hot buffers in internal RAM, or
   quantize the weights to 8 bits. Check every change against `tools/slm_reference.py`.
2. **Compare the other model sizes** of the same family (a model of 15 million
   parameters would need about 60 MB as 32-bit floats, which does not fit in the 8 MB of PSRAM).
3. **Let the user type the beginning**, with a serial command, instead of the five fixed
   prompts.
