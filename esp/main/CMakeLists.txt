idf_component_register(SRCS "wifi.c" "audio_packet.c" "udp.c" "time.c" "buzzer.c" "outputs.c" "main.c"
                    PRIV_REQUIRES esp_wifi nvs_flash esp_driver_gpio driver esp_adc esp_timer 
                    INCLUDE_DIRS ".")
