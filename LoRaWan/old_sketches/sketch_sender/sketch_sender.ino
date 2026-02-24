#include <SPI.h>
#include <LoRa.h>

// Buffer setup
const int interruptPin = 7; // Digital pin for interrupt
const int buffer_size_bytes = 2; // Number of bytes (8 bits per byte)
const int buffer_size_bits = 8 * buffer_size_bytes; // Number of bits (0s and 1s)
byte packed_buffer1[buffer_size_bytes];
byte packed_buffer2[buffer_size_bytes];
bool bit_buffer1[buffer_size_bits];
bool bit_buffer2[buffer_size_bits];
volatile bool current_is_b1 = true;
volatile int counter = 0;
volatile int buffer_number = 0;
bool b1_last_loop = true;

// Pack bits into bytes
void packBitsToBytes(const bool* bitBuffer, byte* packedBuffer, int numBits) {
  for (int i = 0; i < numBits; i += 8) {
    byte currentByte = 0;
    for (int j = 0; j < 8; j++) {
      if (i + j < numBits && bitBuffer[i + j]) {
        currentByte |= (1 << (7 - j)); // Set bit to 1
      }
    }
    packedBuffer[i / 8] = currentByte;
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("LoRa Sender");

  if (!LoRa.begin(868E6)) {
    Serial.println("Starting LoRa failed!");
    while (1);
  }

  pinMode(interruptPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(interruptPin), fill_buffer, CHANGE);
}

void loop() {
  delay(5);
  if ((b1_last_loop && !current_is_b1) || (!b1_last_loop && current_is_b1)) {
    Serial.print("Packet number: "); Serial.println(buffer_number);

    LoRa.beginPacket();
    if (b1_last_loop) {
      packBitsToBytes(bit_buffer2, packed_buffer2, buffer_size_bits);
      LoRa.write(packed_buffer2, buffer_size_bytes);
    } else {
      packBitsToBytes(bit_buffer1, packed_buffer1, buffer_size_bits);
      LoRa.write(packed_buffer1, buffer_size_bytes);
    }

    if (LoRa.endPacket(true)) {
      Serial.println("Packet sent!");
      b1_last_loop = current_is_b1;
    } else {
      Serial.println("Failed to send packet.");
    }
  }
}

void fill_buffer() {
  bool value = digitalRead(interruptPin); // Read from a digital pin (e.g., D3)
  if (current_is_b1) {
    bit_buffer1[counter] = value;
  } else {
    bit_buffer2[counter] = value;
  }
  counter++;
  if (counter >= buffer_size_bits) {
    counter = 0;
    current_is_b1 = !current_is_b1;
    buffer_number++;
  }
}
