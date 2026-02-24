// main.c
#include "time.h"

#define SYNC_UDP_IP_ADDR    CONFIG_ESP_UDP_SYNC_IP
#define SYNC_UDP_PORT       CONFIG_ESP_UDP_SYNC_PORT
#define SYNC_N 256*2

#define TAG "time"

bool synced = false;
int64_t d_us, d_us_n = -1, d_us_x = -1;
double ud_us;
int64_t xm = 0;

int64_t micros(void) {
    return esp_timer_get_time();
    // static struct timeval tv_now;
    // gettimeofday(&tv_now, NULL);
    // return (int64_t)tv_now.tv_sec * 1000000L + (int64_t)tv_now.tv_usec;
}

bool is_synced(void) {
    return synced;
}

int64_t synced_micros(void) {
    if (!synced) return -1;
    return d_us + micros();
}

static void udp_clock_sync(void *pvParameters) {
    UDPSocket* sck = udp_create_socket(SYNC_UDP_IP_ADDR, atoi(SYNC_UDP_PORT));
    udp_connect_socket(sck, 1 * 1000000);

    char rx_buffer[20];
    char payload[8];
    strcpy(payload, (const char*) "/time/");

    int si, len;
    int64_t xs_us, xr_us, ym_us, dt_sr_us;
    // int64_t est_d_us[SYNC_N];
    // double est_ud_us[SYNC_N];

    bool level;

    si = 0;
    while(1) {

        payload[6] = 0;
        payload[7] = (char) (si%255);

        /* envoi de la requête */
        xs_us = micros();
        udp_send_socket(sck, payload, 8);

        while (!gpio_get_level(3)) {}
        xm = micros();

        len = udp_receive_socket(sck, rx_buffer, sizeof(rx_buffer) - 1);
        xr_us = micros();
        /* réponse reçue ou timeout écoulé */

        if (len >= 0) {
            /* réponse reçue */
            char id0 = rx_buffer[6];
            int id1 = (int) rx_buffer[7];
            if (id0 == 0 && si%255 == id1) {
                dt_sr_us = xr_us - xs_us;
                memcpy(&ym_us, rx_buffer + 8, 8);
                printf("%lld, %lld, %lld, %lld,\n", xs_us, xm, xr_us, ym_us);
                // est_d_us[si] = ym_us - (xs_us + xr_us) / 2;
                // est_ud_us[si] = ((double) dt_sr_us) / sqrt(12.0);
                if (d_us_n == -1 || ym_us - xr_us > d_us_n) d_us_n = ym_us - xr_us;
                if (d_us_x == -1 || ym_us - xs_us < d_us_x) d_us_x = ym_us - xs_us;
                si++;
            } else {
                printf("%c, %d ???? %i, %c\n", id0, id1, si, (char) si);
            }
        }

        if (si % 32 == 0) {
            vTaskDelay(1000 / portTICK_PERIOD_MS);
        }

    }

    // int64_t n_received = 0;
    // double ud_us_n = est_ud_us[0];

    // for (si = 1; si < SYNC_N; si++) {
    //     ud_us_n = fmin(ud_us_n, est_ud_us[si]);
    // }

    // printf("u(d)_min = %lf\n", ud_us_n);

    // double sum_inv_sqr = 0.0;
    // double sum_xoi_sqr = 0.0;

    // for (si = 0; si < SYNC_N; si++) {
    //     if (est_ud_us[si] <= 2 * ud_us_n) {
    //         n_received++;
    //         sum_inv_sqr += 1.0 / (est_ud_us[si] * est_ud_us[si]);
    //         sum_xoi_sqr += ((double) est_d_us[si]) / (est_ud_us[si] * est_ud_us[si]);
    //     }
    // }

    // d_us = (int64_t) (sum_xoi_sqr / sum_inv_sqr);
    // ud_us = 1.0 / sqrt(sum_inv_sqr);
    // synced = true;

    // printf("DT = %lld +- %lf (%lld used)\n", d_us, ud_us, n_received);
    // printf("Certitude interval : [ %lld, %lld ] (l = %lld)\n", d_us_n, d_us_x, d_us_x - d_us_n);

    // vTaskDelete(NULL);
}

static void udp_clock_measure(void* pvParameters) {
    static bool level, nlevel;
    level = false;
    while (1) {
        nlevel = gpio_get_level(3);
        if (nlevel != level) {
            level = nlevel;
            if (level) {
                xm = micros();
            }
            // ESP_LOGI(TAG, "gpio 3 : %d (%lld)", nlevel, micros());
        }
        // vTaskDelay(30 / portTICK_PERIOD_MS);
    }
}

void sync_time_service(void) {
    udp_clock_sync(NULL);
    // TaskHandle_t xSync;
    // xTaskCreate(udp_clock_sync, "udp_clock_sync", 4096, NULL, 5, &xSync);
}
