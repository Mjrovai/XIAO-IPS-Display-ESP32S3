/*
  Extra - Button bounce probe
  Records every raw edge on USR1 (D19) and USR2 (D15) with a microsecond
  timestamp, with no debouncing at all, so contact bounce is visible.
  Each press should be one falling edge (DOWN) and, on release, one rising
  edge (UP). Extra edges close together are bounce.

  Serial output, one line per edge:
    <micros>,<button>,<DOWN|UP>,<microseconds since the previous edge on this button>

  Edges are captured in interrupts, so even bounce shorter than a millisecond
  is recorded. No display is used, so nothing else competes for time.
*/

#include <Arduino.h>

static constexpr uint8_t PIN_USR1 = D19;
static constexpr uint8_t PIN_USR2 = D15;

struct Edge {
  uint32_t t;      // micros() at the edge
  uint8_t button;  // 1 or 2
  uint8_t level;   // 0 = pressed (LOW), 1 = released (HIGH)
};

static constexpr size_t QSIZE = 512;
static volatile Edge queue[QSIZE];
static volatile uint16_t head = 0;  // written by the interrupts
static uint16_t tail = 0;           // read by loop()

// One interrupt handler per button; both just store the time and the level.
static void IRAM_ATTR record(uint8_t button, uint8_t pin) {
  uint16_t h = head;
  queue[h].t = micros();
  queue[h].button = button;
  queue[h].level = digitalRead(pin);
  head = (h + 1) % QSIZE;
}
static void IRAM_ATTR isrUsr1() { record(1, PIN_USR1); }
static void IRAM_ATTR isrUsr2() { record(2, PIN_USR2); }

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(1500);
  pinMode(PIN_USR1, INPUT_PULLUP);
  pinMode(PIN_USR2, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_USR1), isrUsr1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_USR2), isrUsr2, CHANGE);
  Serial.println("micros,button,edge,us_since_previous_edge");
}

void loop() {
  static uint32_t last[3] = {0, 0, 0};
  while (tail != head) {
    Edge e;
    e.t = queue[tail].t;
    e.button = queue[tail].button;
    e.level = queue[tail].level;
    tail = (tail + 1) % QSIZE;
    uint32_t dt = last[e.button] ? e.t - last[e.button] : 0;
    last[e.button] = e.t;
    Serial.printf("%lu,USR%u,%s,%lu\n", (unsigned long)e.t, e.button,
                  e.level ? "UP" : "DOWN", (unsigned long)dt);
  }
  delay(1);
}
