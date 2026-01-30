#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "wifi.h"
#include "audio_packet.h"
#include "time.h"

static const char *TAG = "main";

#define LED_PIN 0

void led_indicator_service(void *pvParameters) {
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED_PIN, 1);

    bool level = true, new_level = false;
    int64_t t;
    while (1) {
        if (wifi_state != WIFI_CONNECTED) {
            new_level = false;
        } else {
            if (is_synced()) {
                t = synced_micros() % 400000;
                if (level == false && t < 200000) {
                    new_level = true;
                    gpio_set_level(LED_PIN, 1);
                }
                if (level == true && t >= 200000) {
                    new_level = false;
                    gpio_set_level(LED_PIN, 0);
                }
            } else {
                new_level = true;
            }
        }
        if (new_level != level) {
            gpio_set_level(LED_PIN, new_level);
            level = new_level;
        }
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
}

void app_main(void) {
    //Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Debugging LED indicator task
    // xTaskCreate(led_indicator_service, "led_indicator", 4096, NULL, 7, NULL);

    // Initialize WiFi first (needed for lwIP stack)
    wifi_init_sta();

    uint8_t mac[6];
    wifi_read_mac(mac);

    // Set up I2S
    audio_init();
    
    // Create separate tasks for I2S reading and UDP sending
    xTaskCreate(i2s_read_task, "i2s_read", 4096, NULL, 5, NULL);
    xTaskCreate(send_audio_task, "send_audio", 4096, NULL, 4, NULL);
    xTaskCreate(led_indicator_service, "led_indicator", 4096, NULL, 3, NULL);

    // Time synchronization task
    xTaskCreate(sync_time, "sync_time", 4096, NULL, 6, NULL);

    // Keep main task alive
    while (1) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}
