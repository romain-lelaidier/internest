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

typedef struct delta_builder {
    int64_t xs0, tau0;
    double alpha, beta_1, beta_2;
} delta_builder_t;

typedef struct m_point {
    double xs;
    double qd[2];
    bool used[2];
    int i;
    struct m_point* prev;
    struct m_point* next;
} m_point_t;

m_point_t *first = NULL, *last = NULL;
int n = 0;

bool synced = false;
delta_builder_t delta;

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

m_point_t* create_point(double xs, double q, double d) {
    m_point_t* mp = malloc(sizeof(m_point_t));
    mp->xs = xs;
    mp->qd[0] = q;
    mp->qd[1] = d;
    mp->used[0] = true;
    mp->used[1] = true;
    mp->prev = NULL;
    mp->next = NULL;
    mp->i = n++;
    return mp;
}

void convex_aggregate(int u, m_point_t* i, m_point_t** ip, m_point_t** ipp) {
    double interpolated_ip;
    bool can_stop = false;
    while (!can_stop) {
        interpolated_ip = (*ipp)->qd[u] + ((*ip)->xs - (*ipp)->xs) * (i->qd[u] - (*ipp)->qd[u]) / (i->xs - (*ipp)->xs);
        if ((interpolated_ip > (*ip)->qd[u]) ^ u) {
            // the previous point (ip) is now useless because inside the extended convex hull
            // we can remove it and backpropagate the new convex hull

            // removing ip from the u enveloppe
            (*ip)->used[u] = false;
            // if (!(*ip)->used[!u]) {
            //     // ip is not used at all, we can totally remove it from the chain
            //     (*ip)->prev->next = (*ip)->next;
            //     (*ip)->next->prev = (*ip)->prev;
            //     free(*ip);
            // }

            *ip = *ipp;
            *ipp = (*ipp)->prev;
            while ((*ipp) != NULL && !(*ipp)->used[u]) {
                *ipp = (*ipp)->prev;
            }
            if ((*ipp) == NULL) {
                // printf("# NULL???\n");
                can_stop = true;
            }
        } else {
            can_stop = true;
        }
    }
}

void add_point(delta_builder_t* d, int64_t xs, int64_t ym, int64_t xr) {
    static m_point_t *dipp, *dip, *qipp, *qip;

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
    
    m_point_t *mp = create_point((double)(xs - d->xs0), (double)(ym - xr - d->tau0), (double)(ym - xs - d->tau0));
    printf("%lld, %lld, %lld,\n", xs - d->xs0, ym - xr - d->tau0, ym - xs - d->tau0);

    if (first == NULL) {
        // first node
        first = mp;
        last = mp;
        return;
    }

    // adding node at the end
    last->next = mp;
    mp->prev = last;
    last = mp;

    if (first->next->next == NULL) {
        // 2 elements in list
        qip = first->next;
        qipp = first;
        dip = first->next;
        dipp = first;
        return;
    }

    convex_aggregate(0, mp, &qip, &qipp);
    qipp = qip;
    qip = mp;
    convex_aggregate(1, mp, &dip, &dipp);
    dipp = dip;
    dip = mp;
}

double interpolate(int u, m_point_t* i) {
    if (i->used[u]) {
        return i->qd[u];
    }
    m_point_t* il = i;
    m_point_t* ir = i;
    while (!il->used[u]) {
        il = il->prev;
    }
    while (!ir->used[u]) {
        ir = ir->next;
    }
    return il->qd[u] + (i->xs - il->xs) * (ir->qd[u] - il->qd[u]) / (ir->xs - il->xs);
}

void compute_alphas(delta_builder_t* d) {
    m_point_t *t = first;
    m_point_t* t_star = t;
    double d_star = t_star->qd[1] - t_star->qd[0];
    double d_t, t_star_ext_q, t_star_ext_d;

    double alpha_bl, alpha_br, alpha_tl, alpha_tr, alpha_min, alpha_max;
    m_point_t *ibl, *ibr, *itl, *itr;

    // get point where the smallest distance is reached
    t = first;
    while (t != NULL) {
        d_t = interpolate(1, t) - interpolate(0, t);
        if (d_t < d_star) {
            d_star = d_t;
            t_star = t;
        }
        t = t->next;
    }

    // compute alpha
    t_star_ext_q = interpolate(0, t_star);
    t_star_ext_d = interpolate(1, t_star);
    ibl = t_star->prev;
    itl = t_star->prev;
    ibr = t_star->next;
    itr = t_star->next;
    while (!ibl->used[0]) ibl = ibl->prev;
    while (!itl->used[1]) itl = itl->prev;
    while (!ibr->used[0]) ibr = ibr->next;
    while (!itr->used[1]) itr = itr->next;
    alpha_bl = (t_star_ext_q - ibl->qd[0]) / (t_star->xs - ibl->xs);
    alpha_tl = (t_star_ext_d - itl->qd[1]) / (t_star->xs - itl->xs);
    alpha_br = (ibr->qd[0] - t_star_ext_q) / (ibr->xs - t_star->xs);
    alpha_tr = (itr->qd[1] - t_star_ext_d) / (itr->xs - t_star->xs);

    alpha_min = fmax(alpha_tl, alpha_br);
    alpha_max = fmin(alpha_bl, alpha_tr);

    d->alpha = (alpha_min + alpha_max) * 0.5;
    d->beta_1 = t_star_ext_q - d->alpha * t_star->xs;
    d->beta_2 = t_star_ext_d - d->alpha * t_star->xs;
}

void udp_clock_sync(void) {
    UDPSocket* sck = udp_create_socket(UDP_IP, SYNC_UDP_PORT);
    udp_connect_socket(sck, 1 * 1000000);

    char rx_buffer[21];
    char payload[2];
    // strcpy(payload, (const char*) "/time/");

    int si, len;
    int16_t i;
    int64_t xs, ym, xr, xm = -1, ym_est = -1;

#ifdef GPIO_MEASURE
    bool can_break;
#endif

    si = 0;
    i = 1;
    add_point(NULL, 0, 0, 0);
    while(1) {

        // payload[0] = 0;
        // payload[1] = (char) (i%255);
        memcpy(payload, &i, 2);

#ifdef GPIO_MEASURE
        can_break = false;
        while (!can_break) {
            can_break = !gpio_get_level(MEASURE_IN_PIN);
        }
        gpio_set_level(MEASURE_OUT_PIN, 0);
#endif

        /* envoi de la requête */
        xs = micros();
        udp_send_socket(sck, payload, 2);

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

        if (len == 20 && memcmp(rx_buffer, rx_buffer + 10, 10) == 0) {
            /* réponse reçue */
            if (memcmp(rx_buffer, &i, 2) == 0) {
                memcpy(&ym, rx_buffer + 2, 8);
                add_point(&delta, xs, ym, xr);
                i++;
            } else {
                memcpy(&i, rx_buffer, 2);
                printf("# skipped\n");
                i++;
            }
        } else {
            printf("# error !\n");
            i++;
        }

        if (i % SYNC_PER_SALVE == 0) {
            si++;

            printf("# End of salve\n");
            compute_alphas(&delta);

            if (delta.beta_2 - delta.beta_1 <= 0) {
                synced = false;
                add_point(NULL, 0, 0, 0);
                printf("Negative DB -> restarting sync\n");
                // return;
            } else {
                if (si >= 3) {
                    synced = true;
                }
                printf("# alpha = %.*e, beta in [%lf, %lf] (l=%lf)\n", DECIMAL_DIG, delta.alpha, delta.beta_1, delta.beta_2, delta.beta_2-delta.beta_1);
                vTaskDelay(SYNC_PERIOD / portTICK_PERIOD_MS);
            }

            
            // if (fabs(d.alpha) < 0.0001) {
            //     memcpy(&delta, &d, sizeof(struct Delta));
            //     synced = true;
            // }

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
