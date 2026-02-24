#include "udp.h"
#define TAG "udp"

#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>

typedef struct UDPSocketStruct {
    enum UDPSocketState state;
    struct sockaddr_in* dest_addr;
    struct sockaddr_storage* source_addr;
    socklen_t socklen;
    int addr_family;
    int ip_protocol;
    int socket;
    /* handler */
    bool connected;
    bool initialized;
    /* customizable */
    const char* ip;
    int port;
} UDPSocket;

UDPSocket* udp_create_socket(const char* ip, int port) {
    UDPSocket* sck = malloc(sizeof(UDPSocket));
    sck->state = UDPSOCKET_OFF;
    sck->dest_addr = malloc(sizeof(struct sockaddr_in));
    sck->source_addr = malloc(sizeof(struct sockaddr_storage));
    sck->socklen = sizeof(*(sck->source_addr));

    sck->ip = ip;
    sck->port = port;

    sck->state = UDPSOCKET_INITIALIZED;
    return sck;
}

bool udp_connect_socket(UDPSocket* sck, int timeout_us) {
    if (sck->state == UDPSOCKET_OFF) return 0;
    if (sck->state == UDPSOCKET_CONNECTED) return 1;
    while (1) {
        sck->dest_addr->sin_addr.s_addr = inet_addr(sck->ip);
        sck->dest_addr->sin_family = AF_INET;
        sck->dest_addr->sin_port = htons(sck->port);
        sck->addr_family = AF_INET;
        sck->ip_protocol = IPPROTO_IP;

        sck->socket = socket(sck->addr_family, SOCK_DGRAM, sck->ip_protocol);
        if (sck->socket < 0) {
            ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
            return 0;
        }

        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = timeout_us;
        setsockopt(sck->socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        sck->state = UDPSOCKET_CONNECTED;
        return 1;
    }
}

bool udp_send_socket(UDPSocket* sck, const void *dataptr, size_t size) {
    int err;
    if (sck->state != UDPSOCKET_CONNECTED) return 0;
    /* envoi de la requête */
    err = sendto(sck->socket, dataptr, size, 0, (struct sockaddr*) sck->dest_addr, sizeof(*(sck->dest_addr)));
    if (err < 0) {
        ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
        return 0;
    }
    return 1;

    // if (sock != -1) {
    //     ESP_LOGE(TAG, "Shutting down socket and restarting...");
    //     shutdown(sock, 0);
    //     close(sock);
    // }
}

int udp_receive_socket(UDPSocket* sck, const void* dataptr, size_t size) {
    if (sck->state != UDPSOCKET_CONNECTED) return -1;
    return recvfrom(sck->socket, dataptr, size, 0, (struct sockaddr*) sck->source_addr, &(sck->socklen));
}