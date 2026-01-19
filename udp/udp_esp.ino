#include <WiFi.h>
#include <WiFiUdp.h>
WiFiUDP udp;

char packetBuffer[255];
unsigned int localPort = 9999;
char *serverip = "192.168.1.46";
unsigned int serverport = 8888;

const char *ssid = "******";
const char *password = "********";

void setup() {
  Serial.begin(115200);
  // Connect to Wifi network.
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(F("."));
  }
  udp.begin(localPort);
  Serial.printf("UDP Client : %s:%i \n", WiFi.localIP().toString().c_str(), localPort);
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    Serial.print(" Received packet from : "); Serial.println(udp.remoteIP());
    int len = udp.read(packetBuffer, 255);
    Serial.printf("Data : %s\n", packetBuffer);
    Serial.println();
  }
  delay(500);
  Serial.print("[Client Connected] "); Serial.println(WiFi.localIP());
  udp.beginPacket(serverip, serverport);
  char buf[30];
  unsigned long testID = millis();
  sprintf(buf, "ESP32 send millis: %lu", testID);
  udp.printf(buf);
  udp.endPacket();
}
