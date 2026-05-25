#pragma once
#include <stddef.h>
#include <stdint.h>
// constantes principales
#define UCSP_MAGIC 0x5543
#define UCSP_DATAGRAM_SIZE 500
#define UCSP_HEADER_SIZE 21
#define UCSP_MAX_PAYLOAD 479
#define UCSP_DEFAULT_PORT 9000
// tiempo maximo esperando ack
#define UCSP_TIMEOUT_MS 500
// cuantos intentos antes de rendirse
#define UCSP_MAX_RETRIES 5
// tipos de mensajes
#define UCSP_TYPE_SYN 0x01
#define UCSP_TYPE_SYN_ACK 0x02
#define UCSP_TYPE_DATA 0x03
#define UCSP_TYPE_ACK 0x04
#define UCSP_TYPE_NACK 0x05
#define UCSP_TYPE_FIN 0x06
// struct principal del protocolo
// packed para que no meta bytes raros
typedef struct __attribute__((packed)) {
  uint16_t magic;
  uint8_t type;
  uint32_t seq_num;
  uint16_t total_frags;
  uint16_t frag_index;
  uint16_t session_id;
  uint16_t payload_len;
  uint32_t crc32;
  uint8_t reserved[2];
  uint8_t payload[479];
} UCSPDatagram;

// si esto falla algo esta mal en el struct
static_assert(sizeof(UCSPDatagram) == 500,
              "UCSPDatagram debe medir exactamente 500 bytes");

// info basica del worker
typedef struct {

  const char *ip;

  int port;

} WorkerInfo;

// manda data y espera respuesta
int ucsp_send_data(float *data_in, size_t send_elements, const char *worker_ip,
                   int port, float *result_out, size_t recv_elements);

// reparte data a varios workers y promedia resultados
int ucsp_distribute(float **data_in_per_worker, size_t *elements_per_worker,
                    WorkerInfo *workers, int n_workers, float *gradient_avg,
                    size_t model_params);

// callback para procesar data recibida
typedef void (*DataCallback)(float *received_data, size_t total_elements,
                             float *result_out);

// worker escuchando conexiones
int ucsp_listen(int port, size_t expected_recv_elements,
                size_t elements_to_return, DataCallback callback);

// funciones internas
uint32_t ucsp_crc32(const uint8_t *data, size_t len);

void ucsp_make_datagram(UCSPDatagram *dg, uint8_t type, uint32_t seq,
                        uint16_t session_id, uint16_t total_frags,
                        uint16_t frag_index, const uint8_t *payload,
                        uint16_t payload_len);

// revisa magic + crc
int ucsp_validate(const UCSPDatagram *dg);
