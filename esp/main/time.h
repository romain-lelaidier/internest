#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <sys/time.h>
#include <math.h>
#include <driver/gpio.h>

#include "esp_timer.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "freertos/task.h"

#include "udp.h"

bool is_synced(void);
int64_t micros(void);
int64_t synced_micros(void);
void sync_time_service(void);
void sync_time(void* pvParameters);