#include "outputs.h"

void led_indicator_service(void *pvParameters) {
    gpio_reset_pin(GPIO_LED);
    gpio_set_direction(GPIO_LED, GPIO_MODE_OUTPUT);
    gpio_set_level(GPIO_LED, 1);

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
                    gpio_set_level(GPIO_LED, 1);
                }
                if (level == true && t >= 200000) {
                    new_level = false;
                    gpio_set_level(GPIO_LED, 0);
                }
            } else {
                new_level = true;
            }
        }
        if (new_level != level) {
            gpio_set_level(GPIO_LED, new_level);
            level = new_level;
        }
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
}