#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "wifi.h"
#include "audio_packet.h"
#include "time.h"
#include "outputs.h"

static const char *TAG = "main";

void app_main(void) {
    //Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Debugging LED indicator task
    xTaskCreate(led_indicator_service, "led_indicator", 4096, NULL, 7, NULL);

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

    // while (1) {
    //     double f = 1380 + 50 * cos((double) micros() / 1000.0);
    //     printf("%lf\n", f);
    //     beep(f, 100);  // Beep at 1000Hz for 500ms
    //     vTaskDelay(pdMS_TO_TICKS(10));  // Wait 1 second before beeping again
    // }

    // Keep main task alive
    while (1) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}
