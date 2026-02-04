#include "outputs.h"

void beep(int frequency, int duration_ms) {
    ledc_timer_config_t timer_conf = {
        .speed_mode       = LEDC_LOW_SPEED_MODE,
        .duty_resolution  = LEDC_TIMER_13_BIT,
        .timer_num        = LEDC_TIMER_0,
        .freq_hz          = frequency,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_conf));

    ledc_channel_config_t channel_conf = {
        .gpio_num       = GPIO_BUZZER,
        .speed_mode     = LEDC_LOW_SPEED_MODE,
        .channel        = LEDC_CHANNEL_0,
        .intr_type      = LEDC_INTR_DISABLE,
        .timer_sel      = LEDC_TIMER_0,
        .duty           = 6000,  // 50% duty cycle
        .hpoint         = 0
    };
    ESP_ERROR_CHECK(ledc_channel_config(&channel_conf));

    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    ESP_ERROR_CHECK(ledc_stop(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, 0));
}

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