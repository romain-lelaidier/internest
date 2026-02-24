#include <Wire.h>
#include <LoRa.h>

const int ledPin = 6;
volatile bool x;
volatile bool newI2CData = false;
volatile byte i2cBuffer[32];
volatile byte bufferIndex = 0;
int packet_number = 0;

void setup() {
  Serial.begin(9600);
  //while(!Serial);
  // Join I2C bus as slave with address 8
  Wire.begin(0x8);
  // Call receiveEvent when data received                
  Wire.onReceive(receiveEvent);
  
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  if (!LoRa.begin(868E6)) {
    Serial.println("Starting LoRa failed!");
    while (1);
  }
  Serial.println("Server initialised !");
}

void receiveEvent(int howMany) {
  while (Wire.available() && bufferIndex < sizeof(i2cBuffer)) {
    i2cBuffer[bufferIndex++] = Wire.read();
  }
  newI2CData = true; // Signal main loop
}

void loop() {
  if (newI2CData) {
    blink_led();
    // Pause LoRa if needed, or process data quickly
    LoRa.beginPacket();
    for (byte i = 0; i < bufferIndex; i++) {
      LoRa.write(i2cBuffer[i]);
      Serial.print(i2cBuffer[i]); Serial.print(", ");
    }
  if (LoRa.endPacket(true)) {
    Serial.print("Packet sent n° "); Serial.println(packet_number);
    packet_number++;
  } else {
    Serial.println("Failed to send packet.");
  }
    bufferIndex = 0; // Reset buffer
    newI2CData = false;
  }
}

void blink_led() {
    x = !x;
    if(x == true) {
      digitalWrite(ledPin, HIGH);
    }
    else {
      digitalWrite(ledPin, LOW);
    }
}  

