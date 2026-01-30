#ifndef UDP_H
#define UDP_H

#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <sys/time.h>
#include <math.h>
#include <driver/gpio.h>

#include "esp_log.h"
#include "esp_wifi.h"
#include "freertos/task.h"

enum UDPSocketState {
    UDPSOCKET_OFF,
    UDPSOCKET_INITIALIZED,
    UDPSOCKET_CONNECTED
};

typedef struct UDPSocketStruct UDPSocket;

UDPSocket* udp_create_socket(const char* ip, int port);
bool udp_connect_socket(UDPSocket* sck, int timeout_us);
bool udp_send_socket(UDPSocket* sck, const void *dataptr, size_t size);
int udp_receive_socket(UDPSocket* sck, const void* dataptr, size_t size);

#endif