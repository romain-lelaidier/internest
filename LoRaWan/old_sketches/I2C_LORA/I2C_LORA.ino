// Include the Wire library for I2C
#include <Wire.h>
#include <LoRa.h>
 
const int ledPin = 6;
volatile bool x;
byte buffer[32];

void setup() {
  Serial.begin(9600);
  while(!Serial);
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
 
// Function that executes whenever data is received from master
void receiveEvent(int howMany) {
  blink_led();
  byte byte_number = 0;
  while (Wire.available()) { // loop through all but the last
    if (byte_number >= 32) {Serial.println("buffer is overloaded. Stop streaming."); while(1);}
    buffer[byte_number] = Wire.read(); // receive byte as a character
    byte_number++;
  }
  byte message[byte_number];
  for (int i=0; i <= byte_number; i++) {
    message[i] = buffer[i];
  }
  LoRa.beginPacket();
  LoRa.write(message, byte_number);
  if (LoRa.endPacket(true)) {
    Serial.println("Packet sent!");
  } else {
    Serial.println("Failed to send packet.");
  }
}

void loop() {
  delay(100);
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
