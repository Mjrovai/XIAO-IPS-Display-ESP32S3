// Minimal POSIX porting layer for the Edge Impulse SDK, so the model library can run on a computer.
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"

EI_IMPULSE_ERROR ei_run_impulse_check_canceled() { return EI_IMPULSE_OK; }
void ei_serial_set_baudrate(int) {}
EI_IMPULSE_ERROR ei_sleep(int32_t ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); return EI_IMPULSE_OK; }
uint64_t ei_read_timer_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
uint64_t ei_read_timer_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void ei_putchar(char c) { putchar(c); }
char ei_getchar() { return getchar(); }
void ei_printf(const char *format, ...) { va_list a; va_start(a, format); vfprintf(stderr, format, a); va_end(a); }
void ei_printf_float(float f) { fprintf(stderr, "%f", f); }
void *ei_malloc(size_t size) { return malloc(size); }
void *ei_calloc(size_t n, size_t size) { return calloc(n, size); }
void ei_free(void *p) { free(p); }
