#include <Wire.h>

void setup() {
  Wire.begin(0x27);          // Initialize I2C as slave with address 0x27
  Wire.onReceive(receiveEvent); // Register receive event
  Serial.begin(9600);
}

void loop() {
  // Nothing here
}

void receiveEvent(int bytes) {
  while (Wire.available()) {
    int received = Wire.read();
    Serial.print("Received: ");
    Serial.println(received);
  }
}
