/*
This script should be uploaded on the LoRa Arduino card that is connected to the PC through Serial,
and that receives data through LoRa from the other card, which is connected to the Rasp Pi
*/

#include <SPI.h>  // Serial communnication
#include <LoRa.h>  // LoRa communication

bool x;
bool verbose=false;  // false : real mode ; true : debug mode.
// It is necessary to not print anything during packet Serial communication, for it would interfere with the message
int led = 2;
int internal_led = 6;  // Blink indicator : blinks each time it receives a packet.
int packet_number = 0;  // To be improved and harmonised to the whole communication channel : keep track of packet number. 
// Actually, the packet number should be encoded in a byte (or several ones) in the packet

void setup() {
  pinMode(led, OUTPUT);
  Serial.begin(9600);
  while (!Serial);

  if (verbose) {Serial.println("LoRa Receiver");}

  if (!LoRa.begin(868E6)) {
    if (verbose) {Serial.println("Starting LoRa failed!");}
    while (1);
  }
}

void loop() {
  // try to parse packet
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    blink_led();
    // read packet
    while (LoRa.available()) {
      byte new_byte = LoRa.read();
      if (verbose) {Serial.print((byte)new_byte); Serial.print(", ");}
      else {Serial.write(new_byte);}
    }

    if (verbose) {
      Serial.print("Received packet n° "); Serial.print(packet_number);
      Serial.print(" of size "); Serial.println(packetSize);
      Serial.print(" with RSSI = ");
      Serial.println(LoRa.packetRssi());  // The RSSI is an indicator of the strength of the signal
    }
    packet_number++;
  }
}

void blink_led() {
    x = !x;
    if(x == true) {
      digitalWrite(led, HIGH);
      digitalWrite(internal_led, HIGH);
    }
    else {
      digitalWrite(led, LOW);
      digitalWrite(internal_led, LOW);
    }
}  