// Runs the Edge Impulse KWS model on WAV clips, on a computer.
// Usage: host_test GAIN_DB < list_of_wav_paths     (one path per line)
// Prints: path <TAB> label score label score ...
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

static std::vector<int16_t> clip;
static float gainLinear = 1.0f;

static int get_data(size_t offset, size_t length, float *out) {
  for (size_t i = 0; i < length; i++) {
    float v = clip[offset + i] * gainLinear;
    out[i] = v > 32767.f ? 32767.f : (v < -32768.f ? -32768.f : v);   // int16-range floats, as on the board
  }
  return 0;
}

static bool loadWav(const std::string &path, std::vector<int16_t> &out) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  std::vector<uint8_t> b; uint8_t buf[4096]; size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
  fclose(f);
  size_t p = 12;
  while (p + 8 <= b.size()) {
    uint32_t sz; memcpy(&sz, &b[p + 4], 4);
    if (!memcmp(&b[p], "data", 4)) { size_t cnt = std::min<size_t>(sz, b.size() - p - 8) / 2; out.resize(cnt); memcpy(out.data(), &b[p + 8], cnt * 2); return true; }
    p += 8 + sz + (sz & 1);
  }
  return false;
}

static const int16_t *streamData = nullptr;
static int streamGet(size_t offset, size_t length, float *out) {
  for (size_t i = 0; i < length; i++) {
    float v = streamData[offset + i] * gainLinear;
    out[i] = v > 32767.f ? 32767.f : (v < -32768.f ? -32768.f : v);
  }
  return 0;
}

// Continuous mode, like the live sketch: noise + clip + noise, cut into 250 ms slices.
// Prints one line per clip: path <TAB> then, for each slice after the first full window, the four scores.
static int runStream(const char *noisePath) {
  std::vector<int16_t> noise;
  if (!loadWav(noisePath, noise) || noise.size() != 16000) { fprintf(stderr, "bad noise clip\n"); return 1; }
  std::string path;
  while (std::getline(std::cin, path)) {
    std::vector<int16_t> c;
    if (!loadWav(path, c) || c.size() != 16000) { printf("%s\tERROR\n", path.c_str()); continue; }
    std::vector<int16_t> st; st.insert(st.end(), noise.begin(), noise.end()); st.insert(st.end(), c.begin(), c.end());
    st.insert(st.end(), noise.begin(), noise.end());
    run_classifier_init();  // reset the continuous state for each clip
    printf("%s\t", path.c_str());
    const int slice = EI_CLASSIFIER_SLICE_SIZE;
    for (size_t off = 0; off + slice <= st.size(); off += slice) {
      streamData = st.data() + off;
      signal_t signal; signal.total_length = slice; signal.get_data = &streamGet;
      ei_impulse_result_t r = {0};
      if (run_classifier_continuous(&signal, &r, false) != EI_IMPULSE_OK) continue;
      size_t idx = off / slice;
      if (idx < EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW) continue;  // not a full window yet
      printf("[");
      for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) printf("%s %.3f%s", r.classification[i].label, r.classification[i].value, i + 1 < EI_CLASSIFIER_LABEL_COUNT ? " " : "");
      printf("] ");
    }
    printf("\n");
  }
  return 0;
}

int main(int argc, char **argv) {
  gainLinear = powf(10.f, (argc > 1 ? atof(argv[1]) : 0.0) / 20.f);
  run_classifier_init();
  if (argc > 3 && !strcmp(argv[2], "stream")) return runStream(argv[3]);
  std::string path;
  while (std::getline(std::cin, path)) {
    clip.clear();
    if (!loadWav(path, clip) || clip.size() != EI_CLASSIFIER_RAW_SAMPLE_COUNT) { printf("%s\tERROR size %zu\n", path.c_str(), clip.size()); continue; }
    signal_t signal; signal.total_length = clip.size(); signal.get_data = &get_data;
    ei_impulse_result_t result = {0};
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    if (err != EI_IMPULSE_OK) { printf("%s\tERROR %d\n", path.c_str(), (int)err); continue; }
    printf("%s\t", path.c_str());
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) printf("%s %.3f ", result.classification[i].label, result.classification[i].value);
    printf("\n");
  }
  return 0;
}
