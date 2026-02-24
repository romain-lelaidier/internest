#include <SPI.h>
#include <LoRa.h>

bool x;
bool verbose=false;  // false : real mode ; true : debug mode
int led = 2;
int internal_led = 6;
int packet_number = 0;

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
      Serial.println(LoRa.packetRssi());
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