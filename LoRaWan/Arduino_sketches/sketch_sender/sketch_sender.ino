/*
This script should be uploaded on the LoRa Arduino card that is connected to the Rasp Pi through I2C connection,
and that sends data through LoRa to the other card, which is connected to the PC
*/

#include <LoRa.h>
#include <Wire.h>  // For I2C data transfer


const int ledPin = 6;  // Blink indicator : when a packet is received.
volatile bool x;
volatile bool newI2CData = false;
volatile byte i2cBuffer[32];
volatile byte bufferIndex = 0;
int packet_number = 0;

void setup() {
  Serial.begin(9600);
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
  if (newI2CData) {  // When the buffer is full, sends the packet through LoRa
    blink_led();
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

