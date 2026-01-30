// main.c
#include "time.h"
#include <float.h>

#define UDP_IP              CONFIG_ESP_UDP_IP
#define SYNC_UDP_PORT       atoi(CONFIG_ESP_UDP_SYNC_PORT)
#define SYNC_PER_SALVE      atoi(CONFIG_ESP_SYNC_PER_SALVE)
#define SYNC_PERIOD         atoi(CONFIG_ESP_SYNC_PERIOD)
#define SYNC_REFRESH_TIME   atoi(CONFIG_ESP_SYNC_REFRESH_TIME)

#define TAG "time"

// #define GPIO_MEASURE false

#ifdef GPIO_MEASURE
#define MEASURE_IN_PIN      5
#define MEASURE_OUT_PIN     6
#endif

struct Delta {
    int64_t xs0, tau0;
    double alpha, beta_1, beta_2;
};

bool synced = false;
struct Delta delta;

int64_t micros(void) {
    return esp_timer_get_time();
}

bool is_synced(void) {
    return synced;
}

int64_t synced_micros(void) {
    if (!synced) return -1;
    int64_t m = micros();
    return m + delta.tau0 + (int64_t) (delta.alpha * (double) (m - delta.xs0) + (delta.beta_1 + delta.beta_2) / 2.0);
}

typedef struct MeasurePoint {
    double xs;
    double qd[2];
    bool used[2];
    struct MeasurePoint* prev;
    struct MeasurePoint* next;
} m_point;

m_point* create_point(double xs, double q, double d) {
    m_point* mp = malloc(sizeof(m_point));
    mp->xs = xs;
    mp->qd[0] = q;
    mp->qd[1] = d;
    mp->used[0] = true;
    mp->used[1] = true;
    mp->prev = NULL;
    mp->next = NULL;
    return mp;
}

double interpolate(int u, m_point* i, double* a) {
    if (i->used[u]) {
        return i->qd[u];
    }
    m_point* il = i;
    m_point* ir = i;
    while (!il->used[u]) {
        il = il->prev;
    }
    while (!ir->used[u]) {
        ir = ir->next;
    }
    double aw = (double)(ir->qd[u] - il->qd[u]) / (double)(ir->xs - il->xs);
    if (a != NULL) *a = aw;
    return il->qd[u] + aw * (i->xs - il->xs);
}

void convex_aggregate(int u, m_point* i, m_point** i1, m_point** i2, double* D_star, m_point** t_star, double *alpha) {
    double interpolated_i2, nD, s = u ? 1.0 : -1.0;
    double au, a1u;
    bool can_stop = false;
    m_point* nt;
    while (!can_stop) {
        au = (i->qd[u] - (*i1)->qd[u]) / (i->xs - (*i1)->xs);
        interpolated_i2 = (*i1)->qd[u] + au * ((*i2)->xs - (*i1)->xs);
        if (interpolated_i2 == (*i2)->qd[u] || (interpolated_i2 > (*i2)->qd[u]) ^ u) {
            // the previous point is now in the extended convex hull
            // we can remove it and backpropagate the new convex hull
            
            // removing i2 from the u enveloppe
            (*i2)->used[u] = false;
            if (!(*i2)->used[!u]) {
                // i2 is not used at all, we can totally remove it from the chain
                (*i2)->prev->next = (*i2)->next;
                (*i2)->next->prev = (*i2)->prev;
                free(*i2);
            }

            // the segment [i1, i] in domain u including point i has just been added.
            // we must find the point t that minimizes the vertical distance between this segment
            // and the other domain 1-u, and update D and alpha (a) accordingly.
            nt = *i1;
            nD = s * ((*i1)->qd[u] - interpolate(1-u, *i1, &a1u));
            if (nD < *D_star) {
                *D_star = nD;
                *t_star = nt;
                *alpha = a1u;
            }
            nt = i;
            nD = s * (i->qd[u] - interpolate(1-u, i, &a1u));
            if (nD < *D_star) {
                *D_star = nD;
                *t_star = nt;
                *alpha = a1u;
            }
            nt = (*i1)->next;
            while (nt != NULL) {
                while (!nt->used[1-u]) {
                    nt = nt->next;
                }
                nD = s * ((*i1)->qd[u] + au * (nt->xs - (*i1)->xs) - nt->qd[1-u]);
                if (nD < *D_star) {
                    *D_star = nD;
                    *t_star = nt;
                    *alpha = au;
                }
                nt = nt->next;
            }

            *i2 = (*i1);
            *i1 = (*i1)->prev;
            while ((*i1) != NULL && !(*i1)->used[u]) {
                *i1 = (*i1)->prev;
            }
            if ((*i1) == NULL) {
                can_stop = true;
            }
        } else {
            can_stop = true;
        }
    }
}

void add_point(struct Delta* d, int64_t xs, int64_t ym, int64_t xr) {
    static m_point *first = NULL, *last, *di1, *di2, *qi1, *qi2, *t_star;
    static double D_star;

    if (d == NULL) {
        // cleaning previous queue
        while (first != NULL && first->next != NULL) {
            first = first->next;
            free(first->prev);
        }
        if (first != NULL) {
            free(first);
        }
        first = NULL;
        last = NULL;
        return;
    }

    if (first == NULL) {
        d->xs0 = xs;
        d->tau0 = ym - xs;
        // d->xs0 = 0;
        // d->tau0 = 0;
    }
    
    m_point* mp = create_point((double)(xs - d->xs0), (double)(ym - xr - d->tau0), (double)(ym - xs - d->tau0));
    // printf("%lf, %lf, %lf,\n", mp->xs, mp->qd[0], mp->qd[1]);

    if (first == NULL) {
        // first node
        first = mp;
        last = mp;
        D_star = mp->qd[1] - mp->qd[0];
        t_star = mp;
        return;
    }

    // adding node at the end
    last->next = mp;
    mp->prev = last;
    last = mp;

    if (first->next->next == NULL) {
        // 2 elements in list
        qi1 = first;
        di1 = first;
        qi2 = first->next;
        di2 = first->next;
        return;
    }

    convex_aggregate(1, mp, &di1, &di2, &D_star, &t_star, &d->alpha);
    convex_aggregate(0, mp, &qi1, &qi2, &D_star, &t_star, &d->alpha);
    di1 = di2;
    di2 = mp;
    qi1 = qi2;
    qi2 = mp;

    d->beta_1 = interpolate(0, t_star, NULL) - d->alpha * t_star->xs;
    d->beta_2 = interpolate(1, t_star, NULL) - d->alpha * t_star->xs;
}

void udp_clock_sync(void) {
    UDPSocket* sck = udp_create_socket(UDP_IP, SYNC_UDP_PORT);
    udp_connect_socket(sck, 1 * 1000000);

    char rx_buffer[20];
    char payload[8];
    strcpy(payload, (const char*) "/time/");

    int i, si, len;
    struct Delta d;
    int64_t xs, ym, xr, xm = -1, ym_est = -1;

#ifdef GPIO_MEASURE
    bool can_break;
#endif

    si = 0;
    i = 1;
    add_point(NULL, 0, 0, 0);
    while(1) {

        payload[6] = 0;
        payload[7] = (char) (i%255);

#ifdef GPIO_MEASURE
        can_break = false;
        while (!can_break) {
            can_break = !gpio_get_level(MEASURE_IN_PIN);
        }
        gpio_set_level(MEASURE_OUT_PIN, 0);
#endif

        /* envoi de la requête */
        xs = micros();
        udp_send_socket(sck, payload, 8);
        
#ifdef GPIO_MEASURE        
        can_break = false;
        while (!can_break) {
            can_break = gpio_get_level(MEASURE_IN_PIN);
        }
        xm = micros();
        ym_est = synced_micros();
        gpio_set_level(MEASURE_OUT_PIN, 1);
#endif

        len = udp_receive_socket(sck, rx_buffer, sizeof(rx_buffer) - 1);
        xr = micros();
        /* réponse reçue ou timeout écoulé */

        if (len >= 0) {
            /* réponse reçue */
            char id0 = rx_buffer[6];
            int id1 = (int) rx_buffer[7];
            if (id0 == 0 && i%255 == id1) {
                memcpy(&ym, rx_buffer + 8, 8);
                add_point(&d, xs, ym, xr);
                // printf("%lld, %lld, %lld, %lld, %lld,\n", xs, xr, ym, xm, ym_est);
                i++;
            } else {
                printf("%c, %d ???? %i, %c\n", id0, id1, si, (char) si);
            }
        }

        if (i % SYNC_PER_SALVE == 0) {
            si++;
            
            if (fabs(d.alpha) < 0.0001) {
                memcpy(&delta, &d, sizeof(struct Delta));
                printf("# alpha = %.*e, beta in [%lf, %lf] (l=%lf)\n", DECIMAL_DIG, d.alpha, d.beta_1, d.beta_2, d.beta_2-d.beta_1);
                synced = true;
                vTaskDelay(SYNC_PERIOD / portTICK_PERIOD_MS);
            }

            // if (fabs(d.beta_2 - d.beta_1 - delta_beta) > 0.5) {
            //     delta_beta = d.beta_2 - d.beta_1;
            //     delta_beta_count = 0;
            // } else {
            //     delta_beta_count++;
            //     if (delta_beta_count == 10) {
            //         printf("SYNCED\n");
            //         synced = true;
            //         break;
            //     }
            // }
            
        }

    }
}

void sync_time(void* pvParameters) {
    while (1) {
        udp_clock_sync();
        vTaskDelay(SYNC_REFRESH_TIME / portTICK_PERIOD_MS);
    }
}

void sync_time_service(void) {
    // udp_clock_sync(NULL);
    TaskHandle_t xSync;
    xTaskCreate(sync_time, "sync_time", 4096, NULL, 5, &xSync);
}
