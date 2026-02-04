#include "driver/ledc.h"
#include "wifi.h"
#include "time.h"

#define GPIO_LED 0
#define GPIO_BUZZER 5

void led_indicator_service(void *pvParameters);

void beep(int frequency, int duration_ms);