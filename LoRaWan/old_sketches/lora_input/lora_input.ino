int pin = 13;
bool last_is_even = true;

void setup() {
  pinMode(pin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  Serial.println("im working");
  int even = (millis() / 100) % 2;
  if (even == 0) {
    digitalWrite(pin, HIGH);
    if (!last_is_even) {
      last_is_even = true;
    }
  }
  else {
    digitalWrite(pin, LOW);
    if (last_is_even) {
      last_is_even = false;
    }
  }

}

// void send_signal(byte signal[], float sending_time) {
//   number_of_bytes = len(signal);
//   for (i=0; i <= number_of_bytes; i++) {
//     
//   }
// }
