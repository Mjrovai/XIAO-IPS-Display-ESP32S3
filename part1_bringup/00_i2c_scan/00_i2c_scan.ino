/*
  Test 00 - I2C scan
  Lists every device on the board's I2C bus (SDA=D4, SCL=D5).
  Expected on the 1.47" touch version: touch controller (0x63) and the IMU.
*/

#include <Wire.h>

void setup() {
  Serial.begin(115200);
  // Give the USB serial port time to enumerate before the first print.
  delay(1500);
  // The touch controller and the IMU share this bus: SDA = D4, SCL = D5.
  Wire.begin(D4, D5);
  // 400 kHz fast mode; both devices support it.
  Wire.setClock(400000);
}

void loop() {
  Serial.println("Scanning I2C bus...");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    // A device that acknowledges its address makes endTransmission() return 0.
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  device at 0x%02X\n", addr);
      found++;
    }
  }
  Serial.printf("Done: %d device(s)\n\n", found);
  // Repeat, so a serial monitor opened late still sees the result.
  delay(3000);
}
