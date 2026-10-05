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
