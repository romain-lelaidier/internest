
#include "audio_packet.h"

// Connections to INMP441 I2S microphone
#define I2S_SD GPIO_NUM_2
#define I2S_WS GPIO_NUM_3
#define I2S_SCK GPIO_NUM_4

// Use I2S Processor 0
#define I2S_PORT I2S_NUM_0

// audio sampling
#define SAMPLE_RATE         48000
#define BYTES_PER_SAMPLE    2

// UDP packets
#define PACKET_SIZE         1024*8*3
#define UDP_IP              CONFIG_ESP_UDP_IP
#define AUDIO_UDP_PORT      atoi(CONFIG_ESP_UDP_AUDIO_PORT)
 
void i2s_install() {
    // Set up I2S Processor configuration
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = 8 * BYTES_PER_SAMPLE,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin() {
    // Set I2S pin configuration
    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };

    i2s_set_pin(I2S_PORT, &pin_config);
}

// Ring buffer for audio samples
char* audio_buffer;
volatile int write_index = 0;
volatile int read_index = 0;
volatile int samples_available = 0;

void audio_init() {
    audio_buffer = malloc(sizeof(char) * PACKET_SIZE * 2);
    i2s_install();
    i2s_setpin();
    i2s_start(I2S_PORT);
}

void i2s_read_task(void *pvParameters) {
    char* raw_samples = malloc(1024 * sizeof(char) * BYTES_PER_SAMPLE);
    
    while (1) {
        size_t bytes_read = 0;
        i2s_read(I2S_PORT, (void **)raw_samples, 1024 * sizeof(char) * BYTES_PER_SAMPLE, &bytes_read, 100);
        
        if (bytes_read > 0) {
            // Write to ring buffer
            for (int i = 0; i < bytes_read; i++) {
                audio_buffer[write_index] = raw_samples[i];
                write_index = (write_index + 1) % (PACKET_SIZE * 2);
                samples_available++;
            }
        }
    }
}

void send_audio_task(void *pvParameters) {
    while (wifi_state != WIFI_CONNECTED) {
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
    
    UDPSocket* sck = udp_create_socket(UDP_IP, AUDIO_UDP_PORT);
    udp_connect_socket(sck, 1 * 1000000);

    char* samples = malloc((6 + 8 + PACKET_SIZE) * sizeof(char));
    int64_t t;
    
    while (1) {
        // Wait until enough samples are available
        if (samples_available >= PACKET_SIZE) {
            int sample_index = 0;
            while (sample_index < PACKET_SIZE && samples_available > 0) {
                samples[6 + 8 + sample_index] = audio_buffer[read_index];
                read_index = (read_index + 1) % (PACKET_SIZE * 2);
                sample_index++;
                samples_available--;
            }
            
            if (sample_index == PACKET_SIZE) {
                if (is_synced()) {
                    t = synced_micros();
                    // printf("sending packet at t=%lld\n", t);
                    // samples[0] = ESP_ID;
                    memcpy(samples, mac, 6 * sizeof(char));
                    memcpy(samples + 6 * sizeof(char), &t, 8 * sizeof(char));
                    udp_send_socket(sck, samples, 6 + 8 + sample_index);
                }
            }
        }
        vTaskDelay(1);
    }
}