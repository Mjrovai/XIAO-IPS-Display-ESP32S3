/*
  Part 3, Test 08 - How fast would a 4-million-parameter character model run on the board?

  The VerneBot models of the book "Edge AI Engineering" (a GRU network and a Transformer, about 4
  million parameters each, 123 characters) are too big for the stories260K sketch: 16 MB as 32-bit
  floats. This sketch measures the speed they would have WITHOUT any trained weights: it builds
  random 8-bit weights with exactly the shapes of the two models, and times the generation of one
  character at a time. The speed of a network does not depend on the values of its weights, only
  on its shapes, so the numbers are the real speed; the text it "writes" is meaningless.

  Shapes (they reproduce the parameter counts of the chapter exactly):
    GRU:          embedding 123 x 256, a GRU layer of 1024 units (Keras layout, reset_after),
                  a dense layer 1024 -> 123.                                   4,095,867 weights
    Transformer:  5 blocks, 4 heads of 64, width 256, feed-forward 1024, learned positions,
                  LayerNorm, GELU, output tied to the embedding, window 256.   4,046,336 weights

  Two ways of doing the arithmetic are timed:
    float:  8-bit weights times floating-point activations
    int8:   8-bit weights times 8-bit activations, summed in 32-bit integers, four bytes per load
  and a third test reads the 4 MB of weights from the PSRAM and does nothing else, which is the
  ceiling set by the memory.

  Build with PSRAM = OPI PSRAM. No display, no card: the results go to the serial port (115200)
  at start-up and each time you send a character.
*/

#include <Arduino.h>
#include <math.h>

static constexpr int V = 123, D = 256, H = 4, HD = 64, L = 5, FF = 1024, WIN = 256, U = 1024;

// Declared before the functions so that the Arduino build places its prototypes after it.
struct Mat { int8_t *w; float *s; int rows, cols; };   // out[i] = s[i] * sum_j w[i][j] * x[j]

// ---- Random 8-bit weights ---------------------------------------------------------------------------------
static uint32_t rngS = 2463534242u;
static inline uint32_t rnd() { rngS ^= rngS << 13; rngS ^= rngS >> 17; rngS ^= rngS << 5; return rngS; }


static Mat makeMat(int rows, int cols) {
  Mat m; m.rows = rows; m.cols = cols;
  size_t n = (size_t)rows * cols;
  m.w = (int8_t *)ps_malloc(n);
  m.s = (float *)ps_malloc(rows * sizeof(float));
  if (!m.w || !m.s) { Serial.println("out of PSRAM"); for (;;) delay(1000); }
  uint32_t *p = (uint32_t *)m.w;
  for (size_t i = 0; i < n / 4; i++) p[i] = rnd();
  for (int i = 0; i < rows; i++) m.s[i] = 1.0f / (127.0f * sqrtf((float)cols)) * 2.0f;  // keeps activations tame
  return m;
}

// ---- The two kinds of matrix-vector product -----------------------------------------------------------
static void mvFloat(float *out, const Mat &m, const float *x) {
  for (int i = 0; i < m.rows; i++) {
    const int8_t *w = m.w + (size_t)i * m.cols;
    float a0 = 0, a1 = 0;
    int j = 0;
    for (; j + 1 < m.cols; j += 2) { a0 += (float)w[j] * x[j]; a1 += (float)w[j + 1] * x[j + 1]; }
    out[i] = (a0 + a1) * m.s[i];
  }
}

static float quantize(int8_t *q, const float *x, int n) {   // returns the scale of x
  float mx = 1e-9f;
  for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > mx) mx = a; }
  float sc = mx / 127.0f, inv = 127.0f / mx;
  for (int j = 0; j < n; j++) q[j] = (int8_t)lrintf(x[j] * inv);
  return sc;
}

static void mvInt8(float *out, const Mat &m, const int8_t *xq, float sx) {
  const uint32_t *xq4 = (const uint32_t *)xq;
  for (int i = 0; i < m.rows; i++) {
    const uint32_t *w4 = (const uint32_t *)(m.w + (size_t)i * m.cols);
    int32_t acc = 0;
    for (int j = 0; j < m.cols / 4; j++) {
      uint32_t a = w4[j], b = xq4[j];
      acc += (int8_t)a * (int8_t)b + (int8_t)(a >> 8) * (int8_t)(b >> 8)
           + (int8_t)(a >> 16) * (int8_t)(b >> 16) + (int8_t)(a >> 24) * (int8_t)(b >> 24);
    }
    out[i] = (float)acc * m.s[i] * sx;
  }
}

// One entry point for both: mode 0 = float, mode 1 = int8. scratch is an int8 buffer for the quantized input.
static int mode = 0;
static int8_t xqBuf[U + 16] __attribute__((aligned(4)));
static void mv(float *out, const Mat &m, const float *x) {
  if (mode == 0) mvFloat(out, m, x);
  else { float sx = quantize(xqBuf, x, m.cols); mvInt8(out, m, xqBuf, sx); }
}

// ---- The GRU model ------------------------------------------------------------------------------------
static void freeMat(Mat &m) { free(m.w); free(m.s); m.w = nullptr; m.s = nullptr; }

static Mat gWx, gWh, gWd;
static float *gEmb, *gBx, *gBh, *gBd, *gH, *gGx, *gGh, *gLog;

static void gruInit() {
  gWx = makeMat(3 * U, D); gWh = makeMat(3 * U, U); gWd = makeMat(V, U);
  gEmb = (float *)ps_malloc(V * D * 4); gBx = (float *)ps_malloc(3 * U * 4); gBh = (float *)ps_malloc(3 * U * 4);
  gBd = (float *)ps_malloc(V * 4); gH = (float *)ps_malloc(U * 4); gGx = (float *)ps_malloc(3 * U * 4);
  gGh = (float *)ps_malloc(3 * U * 4); gLog = (float *)ps_malloc(V * 4);
  for (int i = 0; i < V * D; i++) gEmb[i] = ((int32_t)rnd() >> 8) / 8388608.0f * 0.5f;
  for (int i = 0; i < 3 * U; i++) { gBx[i] = 0.01f; gBh[i] = 0.01f; }
  for (int i = 0; i < V; i++) gBd[i] = 0;
  for (int i = 0; i < U; i++) gH[i] = 0;
}

static void gruFree() {
  freeMat(gWx); freeMat(gWh); freeMat(gWd);
  free(gEmb); free(gBx); free(gBh); free(gBd); free(gH); free(gGx); free(gGh); free(gLog);
}

static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

static int gruStep(int c) {     // one character in, the most likely next character out
  mv(gGx, gWx, gEmb + c * D);
  mv(gGh, gWh, gH);
  for (int k = 0; k < U; k++) {
    float r = sigm(gGx[k] + gBx[k] + gGh[k] + gBh[k]);
    float z = sigm(gGx[U + k] + gBx[U + k] + gGh[U + k] + gBh[U + k]);
    float hh = tanhf(gGx[2 * U + k] + gBx[2 * U + k] + r * (gGh[2 * U + k] + gBh[2 * U + k]));
    gH[k] = z * gH[k] + (1.0f - z) * hh;
  }
  mv(gLog, gWd, gH);
  int best = 0;
  for (int i = 0; i < V; i++) { gLog[i] += gBd[i]; if (gLog[i] > gLog[best]) best = i; }
  return best;
}

// ---- The Transformer model ----------------------------------------------------------------------------
static Mat tQkv[L], tPrj[L], tFc1[L], tFc2[L];
static float *tTok, *tPos, *tKc, *tVc, *tX, *tN, *tQkvO, *tAtt, *tAo, *tF1, *tF2, *tLog, *tLnW, *tLnB;

static void trfInit() {
  for (int l = 0; l < L; l++) {
    tQkv[l] = makeMat(3 * D, D); tPrj[l] = makeMat(D, D); tFc1[l] = makeMat(FF, D); tFc2[l] = makeMat(D, FF);
  }
  tTok = (float *)ps_malloc(V * D * 4); tPos = (float *)ps_malloc(WIN * D * 4);
  tKc = (float *)ps_malloc((size_t)L * WIN * D * 4); tVc = (float *)ps_malloc((size_t)L * WIN * D * 4);
  tX = (float *)ps_malloc(D * 4); tN = (float *)ps_malloc(FF * 4); tQkvO = (float *)ps_malloc(3 * D * 4);
  tAtt = (float *)ps_malloc(H * WIN * 4); tAo = (float *)ps_malloc(D * 4); tF1 = (float *)ps_malloc(FF * 4);
  tF2 = (float *)ps_malloc(D * 4); tLog = (float *)ps_malloc(V * 4);
  if (!tKc || !tVc || !tTok) { Serial.println("out of PSRAM"); for (;;) delay(1000); }
  for (int i = 0; i < V * D; i++) tTok[i] = ((int32_t)rnd() >> 8) / 8388608.0f * 0.5f;
  for (int i = 0; i < WIN * D; i++) tPos[i] = ((int32_t)rnd() >> 8) / 8388608.0f * 0.1f;
}

static void trfFree() {
  for (int l = 0; l < L; l++) { freeMat(tQkv[l]); freeMat(tPrj[l]); freeMat(tFc1[l]); freeMat(tFc2[l]); }
  free(tTok); free(tPos); free(tKc); free(tVc); free(tX); free(tN); free(tQkvO); free(tAtt); free(tAo); free(tF1); free(tF2); free(tLog);
}

static void layerNorm(float *o, const float *x) {
  float m = 0, v = 0;
  for (int i = 0; i < D; i++) m += x[i];
  m /= D;
  for (int i = 0; i < D; i++) { float d = x[i] - m; v += d * d; }
  float inv = 1.0f / sqrtf(v / D + 1e-5f);
  for (int i = 0; i < D; i++) o[i] = (x[i] - m) * inv;     // scale 1 and shift 0: the cost is the same
}

static inline float gelu(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }

static int trfStep(int c, int p) {   // character c at position p (0..WIN-1)
  for (int i = 0; i < D; i++) tX[i] = tTok[c * D + i] + tPos[p * D + i];
  for (int l = 0; l < L; l++) {
    layerNorm(tN, tX);
    mv(tQkvO, tQkv[l], tN);
    float *q = tQkvO, *k = tQkvO + D, *v = tQkvO + 2 * D;
    memcpy(tKc + ((size_t)l * WIN + p) * D, k, D * 4);
    memcpy(tVc + ((size_t)l * WIN + p) * D, v, D * 4);
    for (int h = 0; h < H; h++) {
      float *a = tAtt + h * WIN, mx = -1e30f;
      for (int t = 0; t <= p; t++) {
        const float *kt = tKc + ((size_t)l * WIN + t) * D + h * HD;
        float s = 0;
        for (int i = 0; i < HD; i++) s += q[h * HD + i] * kt[i];
        s *= 0.125f; a[t] = s; if (s > mx) mx = s;
      }
      float sum = 0;
      for (int t = 0; t <= p; t++) { a[t] = expf(a[t] - mx); sum += a[t]; }
      float inv = 1.0f / sum;
      for (int i = 0; i < HD; i++) tAo[h * HD + i] = 0;
      for (int t = 0; t <= p; t++) {
        const float *vt = tVc + ((size_t)l * WIN + t) * D + h * HD;
        float w = a[t] * inv;
        for (int i = 0; i < HD; i++) tAo[h * HD + i] += w * vt[i];
      }
    }
    mv(tF2, tPrj[l], tAo);
    for (int i = 0; i < D; i++) tX[i] += tF2[i];
    layerNorm(tN, tX);
    mv(tF1, tFc1[l], tN);
    for (int i = 0; i < FF; i++) tF1[i] = gelu(tF1[i]);
    mv(tF2, tFc2[l], tF1);
    for (int i = 0; i < D; i++) tX[i] += tF2[i];
  }
  layerNorm(tN, tX);
  int best = 0;
  for (int i = 0; i < V; i++) {                         // output tied to the embedding table
    float s = 0; const float *e = tTok + i * D;
    for (int j = 0; j < D; j++) s += e[j] * tN[j];
    tLog[i] = s; if (s > tLog[best]) best = i;
  }
  return best;
}

// ---- Measurements -------------------------------------------------------------------------------------
static volatile int sink = 0;

static void benchRnn(const char *name) {
  gruStep(5);                                            // warm up
  const int N = 200;
  uint32_t t0 = micros(); int c = 5;
  for (int i = 0; i < N; i++) c = gruStep(c);
  uint32_t dt = micros() - t0; sink = c;
  Serial.printf("RESULT GRU          %-5s %8.1f ms per character  %6.2f characters per second\n", name, dt / 1000.0f / N, N * 1e6f / dt);
}

static void benchTrf(const char *name, int window) {
  int c = 5;
  uint32_t tEarly = 0, tLate = 0, tAll = 0;
  for (int p = 0; p < window; p++) {
    uint32_t t0 = micros(); c = trfStep(c, p); uint32_t d = micros() - t0;
    tAll += d;
    if (p < 16) tEarly += d;
    if (p >= window - 16) tLate += d;
  }
  sink = c;
  Serial.printf("RESULT Transformer  %-5s window %3d: first 16 positions %6.1f ms, last 16 %6.1f ms, whole window %6.2f characters per second\n",
                name, window, tEarly / 16000.0f, tLate / 16000.0f, window * 1e6f / tAll);
}

static void benchMemory() {
  size_t n = 4 * 1024 * 1024;
  uint32_t *buf = (uint32_t *)ps_malloc(n);
  for (size_t i = 0; i < n / 4; i++) buf[i] = rnd();
  uint32_t acc = 0, t0 = micros();
  for (int rep = 0; rep < 2; rep++) for (size_t i = 0; i < n / 4; i++) acc += buf[i];
  uint32_t dt = micros() - t0; sink = acc;
  float mbs = 2.0f * n / 1048576.0f / (dt / 1e6f);
  Serial.printf("RESULT memory ceiling: reading the PSRAM with 32-bit loads gives %.1f MB/s; 4 MB of weights per character would allow at most %.1f characters per second\n",
                mbs, mbs / 4.0f);
  free(buf);
}

static void runAll() {
  Serial.printf("\nRESULT --- start, free PSRAM %u bytes, CPU %u MHz ---\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), (unsigned)getCpuFrequencyMhz());
  // The two models do not fit in the PSRAM together (about 4 MB of weights each), so one at a time.
  gruInit();
  Serial.printf("RESULT GRU weights built, PSRAM free %u bytes\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  mode = 0; benchRnn("float"); mode = 1; benchRnn("int8");
  gruFree();
  trfInit();
  Serial.printf("RESULT Transformer weights and cache built, PSRAM free %u bytes\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  mode = 0; benchTrf("float", 120); benchTrf("float", 256);
  mode = 1; benchTrf("int8", 120);  benchTrf("int8", 256);
  trfFree();
  benchMemory();
  Serial.println("RESULT --- done ---");
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(1500);
  Serial.printf("PSRAM free at start: %u bytes\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  runAll();
}

void loop() {
  if (Serial.available()) { while (Serial.available()) Serial.read(); runAll(); }
  delay(50);
}
