// Include I2S driver
#include <driver/i2s.h>
#include <lwip/sockets.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "wifi.h"
#include "udp.h"

void audio_init();
void i2s_read_task(void *pvParameters);
void send_audio_task(void *pvParameters);