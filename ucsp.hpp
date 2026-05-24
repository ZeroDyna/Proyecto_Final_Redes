#pragma once

#include <stdint.h>
#include <stddef.h>

// ─── Constantes ───────────────────────────────────────────────
#define UCSP_MAGIC          0x5543   // "UC" en ASCII
#define UCSP_DATAGRAM_SIZE  500      // tamaño fijo de todo datagrama
#define UCSP_HEADER_SIZE    21       // bytes de cabecera
#define UCSP_MAX_PAYLOAD    479      // bytes disponibles para datos

#define UCSP_DEFAULT_PORT   9000
#define UCSP_TIMEOUT_MS     500      // tiempo de espera por ACK (ajustar en pruebas)
#define UCSP_MAX_RETRIES    5        // reintentos antes de abortar

// ─── Tipos de mensaje ─────────────────────────────────────────
#define UCSP_TYPE_SYN       0x01   // Main → Worker: inicia sesión
#define UCSP_TYPE_SYN_ACK   0x02   // Worker → Main: sesión aceptada
#define UCSP_TYPE_DATA      0x03   // ambos: lleva datos en el payload
#define UCSP_TYPE_ACK       0x04   // ambos: confirma recepción ok
#define UCSP_TYPE_NACK      0x05   // ambos: error, retransmitir
#define UCSP_TYPE_FIN       0x06   // Main → Worker: cierra sesión

// ─── Struct del datagrama ─────────────────────────────────────
// __attribute__((packed)) evita que el compilador agregue padding
typedef struct __attribute__((packed)) {
    uint16_t magic;          // siempre 0x5543
    uint8_t  type;           // tipo de mensaje (ver defines arriba)
    uint32_t seq_num;        // número de secuencia
    uint16_t total_frags;    // cuántos datagramas tiene este mensaje
    uint16_t frag_index;     // índice de este datagrama (empieza en 0)
    uint16_t session_id;     // identifica la sesión actual
    uint16_t payload_len;    // cuántos bytes válidos hay en payload
    uint32_t crc32;          // checksum de cabecera + payload
    uint8_t  reserved[2];    // reservado, siempre 0x0000
    uint8_t  payload[479];   // datos reales, resto es zero-padding
} UCSPDatagram;

// Verifica en tiempo de compilación que el struct mide exactamente 500 bytes
static_assert(sizeof(UCSPDatagram) == 500, "UCSPDatagram debe medir exactamente 500 bytes");

// ─── Info de un worker ────────────────────────────────────────
typedef struct {
    const char* ip;
    int         port;
} WorkerInfo;

// ─── Declaraciones de funciones (implementadas en ucsp.cpp) ───

// Envía data_in (de tamaño send_elements) y espera recibir result_out
int ucsp_send_data(float* data_in, size_t send_elements,
                   const char* worker_ip, int port,
                   float* result_out, size_t recv_elements);

// Distribuye arreglos distintos a cada worker y promedia SOLO los gradientes del modelo
int ucsp_distribute(float** data_in_per_worker, size_t* elements_per_worker,
                    WorkerInfo* workers, int n_workers,
                    float* gradient_avg, size_t model_params);

// Pone a escuchar al worker en un puerto
// Llama a callback cada vez que llega un buffer completo
typedef void (*DataCallback)(float* received_data, size_t total_elements,
                             float* result_out);
                             
// El Worker ahora sabe cuánto va a recibir y cuánto debe devolver
int ucsp_listen(int port, size_t expected_recv_elements, size_t elements_to_return, DataCallback callback);

// Utilidades internas (también usadas por main_server y worker)
uint32_t ucsp_crc32(const uint8_t* data, size_t len);
void     ucsp_make_datagram(UCSPDatagram* dg, uint8_t type,
                            uint32_t seq, uint16_t session_id,
                            uint16_t total_frags, uint16_t frag_index,
                            const uint8_t* payload, uint16_t payload_len);
int      ucsp_validate(const UCSPDatagram* dg);  // verifica magic + CRC