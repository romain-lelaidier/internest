#ifndef WIFI_H
#define WIFI_H

// #include "freertos/event_groups.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"

enum WiFiState {
    WIFI_OFF,
    WIFI_STARTED,
    WIFI_CONNECTED
};

extern enum WiFiState wifi_state;
extern uint8_t mac[6];

void wifi_init_sta(void);
void wifi_read_mac();

#endif