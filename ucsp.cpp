#include "ucsp.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <zlib.h>          
#include <sys/socket.h>    
#include <netinet/in.h>    
#include <arpa/inet.h>     
#include <unistd.h>        

uint32_t ucsp_crc32(const uint8_t* data, size_t len) {
    return (uint32_t)crc32(0L, data, len);
}

// arma el paquete completo
void ucsp_make_datagram(UCSPDatagram* dg, uint8_t type,
                        uint32_t seq, uint16_t session_id,
                        uint16_t total_frags, uint16_t frag_index,
                        const uint8_t* payload, uint16_t payload_len) {

    memset(dg, 0, sizeof(UCSPDatagram));

    // pasar todo a network order
    dg->magic       = htons(UCSP_MAGIC);
    dg->type        = type; 
    dg->seq_num     = htonl(seq);
    dg->total_frags = htons(total_frags);
    dg->frag_index  = htons(frag_index);
    dg->session_id  = htons(session_id);
    dg->payload_len = htons(payload_len);

    if (payload != NULL && payload_len > 0) {
        memcpy(dg->payload, payload, payload_len);
    }

    // crc de header + payload
    uint8_t crc_input[15 + 479];
    memcpy(crc_input,      dg,          15);
    memcpy(crc_input + 15, dg->payload, 479);
    
    dg->crc32 = htonl(ucsp_crc32(crc_input, 15 + 479));
}

// revisar si el paquete esta bien
int ucsp_validate(const UCSPDatagram* dg) {
    if (ntohs(dg->magic) != UCSP_MAGIC) return -1;

    uint8_t crc_input[15 + 479];
    memcpy(crc_input,      dg,          15);
    memcpy(crc_input + 15, dg->payload, 479);
    uint32_t computed = ucsp_crc32(crc_input, 15 + 479);

    if (computed != ntohl(dg->crc32)) return -2;
    return 0;
}

static int create_udp_socket() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("[ucsp] Error creando socket");
        return -1;
    }

    struct timeval tv;
    tv.tv_sec  = UCSP_TIMEOUT_MS / 1000;
    tv.tv_usec = (UCSP_TIMEOUT_MS % 1000) * 1000;

    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    return sock;
}

static int send_fragments(int sock, const uint8_t* data, size_t total_bytes,
                          uint16_t session_id,
                          struct sockaddr_in* dest, socklen_t dest_len) {

    uint16_t total_frags = (uint16_t)((total_bytes + UCSP_MAX_PAYLOAD - 1) / UCSP_MAX_PAYLOAD);

    for (uint16_t i = 0; i < total_frags; i++) {

        size_t offset     = (size_t)i * UCSP_MAX_PAYLOAD;

        size_t chunk_size = (offset + UCSP_MAX_PAYLOAD <= total_bytes)
                            ? UCSP_MAX_PAYLOAD
                            : (total_bytes - offset);

        UCSPDatagram dg;
        int retries = 0;

        while (retries < UCSP_MAX_RETRIES) {

            ucsp_make_datagram(&dg, UCSP_TYPE_DATA, i, session_id,
                               total_frags, i,
                               data + offset, (uint16_t)chunk_size);

            sendto(sock, &dg, UCSP_DATAGRAM_SIZE, 0,
                   (struct sockaddr*)dest, dest_len);

            UCSPDatagram resp;

            ssize_t n = recvfrom(sock, &resp, UCSP_DATAGRAM_SIZE, 0, NULL, NULL);

            if (n == UCSP_DATAGRAM_SIZE && ucsp_validate(&resp) == 0
                && ntohs(resp.session_id) == session_id) {
                
                if (resp.type == UCSP_TYPE_ACK && ntohl(resp.seq_num) == i) {
                    break;  
                }

                if (resp.type == UCSP_TYPE_NACK) {
                    printf("[ucsp] NACK recibido para frag %d, retransmitiendo...\n", i);
                    retries++;
                    continue;
                }
            }

            printf("[ucsp] Timeout en frag %d (intento %d/%d)\n",
                   i, retries + 1, UCSP_MAX_RETRIES);

            retries++;
        }

        if (retries == UCSP_MAX_RETRIES) {
            printf("[ucsp] Frag %d fallo despues de %d intentos. Abortando.\n",
                   i, UCSP_MAX_RETRIES);

            return -1;
        }
    }

    return 0;
}

static int recv_fragments(int sock, uint8_t* out_buf, size_t total_bytes,
                          uint16_t session_id,
                          struct sockaddr_in* src, socklen_t src_len) {

    uint16_t total_frags = (uint16_t)((total_bytes + UCSP_MAX_PAYLOAD - 1) / UCSP_MAX_PAYLOAD);

    uint16_t expected = 0;

    while (expected < total_frags) {

        UCSPDatagram dg;

        ssize_t n = recvfrom(sock, &dg, UCSP_DATAGRAM_SIZE, 0,
                             (struct sockaddr*)src, &src_len);

        if (n != UCSP_DATAGRAM_SIZE) {
            printf("[ucsp] Timeout esperando frag %d\n", expected);
            return -1;
        }

        if (ucsp_validate(&dg) != 0 || ntohs(dg.session_id) != session_id) {

            printf("[ucsp] Datagrama corrupto, enviando NACK para frag %d\n", expected);

            UCSPDatagram nack;

            ucsp_make_datagram(&nack, UCSP_TYPE_NACK, expected,
                               session_id, 0, 0, NULL, 0);

            sendto(sock, &nack, UCSP_DATAGRAM_SIZE, 0,
                   (struct sockaddr*)src, src_len);

            continue;
        }

        if (dg.type != UCSP_TYPE_DATA) continue;

        if (ntohs(dg.frag_index) != expected) continue;

        size_t offset = (size_t)expected * UCSP_MAX_PAYLOAD;

        memcpy(out_buf + offset, dg.payload, ntohs(dg.payload_len));

        UCSPDatagram ack;

        ucsp_make_datagram(&ack, UCSP_TYPE_ACK,
                           ntohl(dg.seq_num),
                           session_id, 0, 0, NULL, 0);

        sendto(sock, &ack, UCSP_DATAGRAM_SIZE, 0,
               (struct sockaddr*)src, src_len);

        expected++;
    }

    return 0;
}

int ucsp_send_data(float* data_in, size_t send_elements,
                   const char* worker_ip, int port,
                   float* data_out, size_t recv_elements) {

    int sock = create_udp_socket();

    if (sock < 0) return -1;

    struct sockaddr_in worker_addr;

    memset(&worker_addr, 0, sizeof(worker_addr));

    worker_addr.sin_family = AF_INET;
    worker_addr.sin_port   = htons(port);

    inet_pton(AF_INET, worker_ip, &worker_addr.sin_addr);

    socklen_t addr_len = sizeof(worker_addr);

    static uint16_t session_counter = 0;

    uint16_t session_id = ++session_counter;

    // handshake
    int retries = 0;

    while (retries < UCSP_MAX_RETRIES) {

        UCSPDatagram syn;

        ucsp_make_datagram(&syn, UCSP_TYPE_SYN,
                           0, session_id, 0, 0, NULL, 0);

        sendto(sock, &syn, UCSP_DATAGRAM_SIZE, 0,
               (struct sockaddr*)&worker_addr, addr_len);

        UCSPDatagram resp;

        ssize_t n = recvfrom(sock, &resp,
                             UCSP_DATAGRAM_SIZE, 0, NULL, NULL);

        if (n == UCSP_DATAGRAM_SIZE && ucsp_validate(&resp) == 0
            && resp.type == UCSP_TYPE_SYN_ACK
            && ntohs(resp.session_id) == session_id) {

            printf("[ucsp] SYN_ACK recibido de %s:%d\n", worker_ip, port);
            break;
        }

        retries++;
    }

    if (retries == UCSP_MAX_RETRIES) {
        close(sock);
        return -2;
    }

    // mandar data
    size_t send_bytes = send_elements * sizeof(float);

    int ret = send_fragments(sock,
                             (uint8_t*)data_in,
                             send_bytes,
                             session_id,
                             &worker_addr,
                             addr_len);

    if (ret != 0) {
        close(sock);
        return -3;
    }

    // recibir respuesta
    size_t recv_bytes = recv_elements * sizeof(float);

    ret = recv_fragments(sock,
                         (uint8_t*)data_out,
                         recv_bytes,
                         session_id,
                         &worker_addr,
                         addr_len);

    if (ret != 0) {
        close(sock);
        return -4;
    }

    // cerrar conexion
    UCSPDatagram fin;

    ucsp_make_datagram(&fin, UCSP_TYPE_FIN,
                       0, session_id, 0, 0, NULL, 0);

    sendto(sock, &fin, UCSP_DATAGRAM_SIZE, 0,
           (struct sockaddr*)&worker_addr, addr_len);

    close(sock);

    return 0;
}

int ucsp_distribute(float** data_in_per_worker, size_t* elements_per_worker,
                    WorkerInfo* workers, int n_workers,
                    float* gradient_avg, size_t model_params) {

    float** results = new float*[n_workers];

    for (int i = 0; i < n_workers; i++) {
        results[i] = new float[model_params](); 
    }

    int failed = 0;

    for (int i = 0; i < n_workers; i++) {

        printf("[ucsp] Enviando %zu elementos al worker %d...\n",
               elements_per_worker[i], i);
        
        int ret = ucsp_send_data(data_in_per_worker[i],
                                 elements_per_worker[i],
                                 workers[i].ip,
                                 workers[i].port,
                                 results[i],
                                 model_params);

        if (ret != 0) {
            printf("[ucsp] Worker %d fallo.\n", i);
            failed++;
        }
    }

    int success = n_workers - failed;

    if (success == 0) {

        for (int i = 0; i < n_workers; i++) {
            delete[] results[i];
        }

        delete[] results;

        return -1;
    }

    memset(gradient_avg, 0, model_params * sizeof(float));

    for (int i = 0; i < n_workers; i++) {

        for (size_t j = 0; j < model_params; j++) {
            gradient_avg[j] += results[i][j];
        }
    }

    for (size_t j = 0; j < model_params; j++) {
        gradient_avg[j] /= (float)success;
    }

    for (int i = 0; i < n_workers; i++) {
        delete[] results[i];
    }

    delete[] results;

    return 0;
}

int ucsp_listen(int port,
                size_t expected_recv_elements,
                size_t elements_to_return,
                DataCallback callback) {

    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    if (sock < 0) return -1;

    struct timeval tv;

    tv.tv_sec  = UCSP_TIMEOUT_MS / 1000;
    tv.tv_usec = (UCSP_TIMEOUT_MS % 1000) * 1000;

    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in my_addr;

    memset(&my_addr, 0, sizeof(my_addr));

    my_addr.sin_family      = AF_INET;
    my_addr.sin_addr.s_addr = INADDR_ANY;
    my_addr.sin_port        = htons(port);

    if (bind(sock, (struct sockaddr*)&my_addr, sizeof(my_addr)) < 0) {
        close(sock);
        return -2;
    }

    printf("[Worker] Escuchando en puerto %d...\n", port);

    size_t recv_bytes   = expected_recv_elements * sizeof(float);
    size_t return_bytes = elements_to_return * sizeof(float);

    uint8_t* recv_buf = new uint8_t[recv_bytes];

    float* result_buf = new float[elements_to_return];

    while (true) {

        struct sockaddr_in main_addr;
        socklen_t main_len = sizeof(main_addr);

        UCSPDatagram dg;

        ssize_t n = recvfrom(sock, &dg,
                             UCSP_DATAGRAM_SIZE, 0,
                             (struct sockaddr*)&main_addr,
                             &main_len);

        if (n != UCSP_DATAGRAM_SIZE) continue;

        if (ucsp_validate(&dg) != 0 || dg.type != UCSP_TYPE_SYN) continue;

        uint16_t session_id = ntohs(dg.session_id);

        UCSPDatagram syn_ack;

        ucsp_make_datagram(&syn_ack, UCSP_TYPE_SYN_ACK,
                           0, session_id, 0, 0, NULL, 0);

        sendto(sock, &syn_ack, UCSP_DATAGRAM_SIZE, 0,
               (struct sockaddr*)&main_addr, main_len);

        memset(recv_buf, 0, recv_bytes);

        if (recv_fragments(sock,
                           recv_buf,
                           recv_bytes,
                           session_id,
                           &main_addr,
                           main_len) != 0) {
            continue;
        }

        memset(result_buf, 0, return_bytes);

        callback((float*)recv_buf,
                 expected_recv_elements,
                 result_buf);

        if (send_fragments(sock,
                           (uint8_t*)result_buf,
                           return_bytes,
                           session_id,
                           &main_addr,
                           main_len) != 0) {
            continue;
        }

        UCSPDatagram fin;

        ssize_t n_fin = recvfrom(sock, &fin,
                                 UCSP_DATAGRAM_SIZE, 0,
                                 NULL, NULL);

        if (n_fin == UCSP_DATAGRAM_SIZE) {

            if (ucsp_validate(&fin) == 0
                && fin.type == UCSP_TYPE_FIN) {

                printf("[Worker] FIN recibido. Sesion %d cerrada.\n",
                       session_id);
            }
        }
    }

    delete[] recv_buf;
    delete[] result_buf;

    close(sock);

    return 0;
}
