#include "ucsp.hpp"
#include "nn_config.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstring>
#include <thread>
#include <mutex>

using namespace std;


// =====================================================
// ESTRUCTURA QUE REPRESENTA EL ESTADO DE UN WORKER
// =====================================================
typedef struct {
    WorkerInfo info;
    int disponible;
    int fallos;
} EstadoWorker;

mutex mtx_print;



/*===========================================
ARMA EL BUFFER QUE SE ENVIARÁ AL WORKER

FORMATO:
[PESOS DEL MODELO][DATOS DEL BATCH]
Retorna la cantidad total de floats enviados.
=============================================*/
size_t armar_buffer_worker(
    float* buf,
    const float* params_modelo,
    size_t cant_params,
    const float* datos_batch,
    size_t tam_batch
) {

    //copia pesos del modelo
    memcpy(buf, params_modelo, cant_params * sizeof(float));

    //copia datos del batch
    if (datos_batch != NULL && tam_batch > 0) {
        memcpy(buf + cant_params, datos_batch, tam_batch * sizeof(float));
    }

    return cant_params + tam_batch;
}


/*==========================
Se encarga de:
1) Empaquetar pesos + batch
2) Enviarlos al worker
3) Esperar gradientes
4) Guardar el resultado
===========================*/
void enviar_a_worker(
    int id_batch,
    int id_worker,
    const float* params_modelo,
    size_t cant_params,
    const vector<float>& batch,
    EstadoWorker* workers,
    vector<vector<float>>& resultados,
    vector<int>& fallidos
) {
    size_t tam_total = cant_params + batch.size();

    float* buf_envio = new float[tam_total];
    float* buf_respuesta = new float[cant_params]();

    armar_buffer_worker(
        buf_envio,
        params_modelo,
        cant_params,
        batch.data(),
        batch.size()
    );

    {
        lock_guard<mutex> lock(mtx_print);
        printf("[SERVER] Batch %d -> Worker %d (%s:%d)\n",
               id_batch,
               id_worker,
               workers[id_worker].info.ip,
               workers[id_worker].info.port);
    }

    int ret = ucsp_send_data(
        buf_envio,
        tam_total,
        workers[id_worker].info.ip,
        workers[id_worker].info.port,
        buf_respuesta,
        cant_params
    );

    if (ret == 0) {
        resultados[id_batch].assign(buf_respuesta, buf_respuesta + cant_params);

        lock_guard<mutex> lock(mtx_print);
        printf("[SERVER] Batch %d OK\n", id_batch);
    } else {
        fallidos[id_batch] = 1;
        workers[id_worker].fallos++;

        lock_guard<mutex> lock(mtx_print);
        printf("[SERVER] Batch %d FALLÓ en Worker %d, código %d\n",
               id_batch, id_worker, ret);
    }

    delete[] buf_envio;
    delete[] buf_respuesta;
}


/*============================================
DISTRIBUYE LOS BATCHES ENTRE LOS WORKERS
Batch 0 -> Worker 0
Batch 1 -> Worker 1
Batch 2 -> Worker 2
...

Cada envío se realiza en un thread diferente.
=============================================*/
int distribuir_batches(
    const float* params_modelo,
    size_t cant_params,
    const vector<vector<float>>& partes_batch,
    EstadoWorker* workers,
    int num_workers,
    vector<vector<float>>& resultados
) {
    if (partes_batch.empty()) {
        fprintf(stderr, "[SERVER] Sin batches\n");
        return -1;
    }

    resultados.clear();
    resultados.resize(partes_batch.size());

    vector<int> fallidos(partes_batch.size(), 0);
    vector<thread> hilos;

    for (size_t i = 0; i < partes_batch.size(); i++) {
        int id_worker = i % num_workers;

        hilos.push_back(thread(
            enviar_a_worker,
            (int)i,
            id_worker,
            params_modelo,
            cant_params,
            cref(partes_batch[i]),
            workers,
            ref(resultados),
            ref(fallidos)
        ));
    }

    for (auto& h : hilos) {
        h.join();
    }

    int cant_fallidos = 0;

    for (int f : fallidos) {
        if (f) cant_fallidos++;
    }

    if (cant_fallidos == (int)partes_batch.size()) {
        return -1;
    }

    if (cant_fallidos > 0) {
        return -2;
    }

    return 0;
}


/*=================================================
grad_final =
(grad_worker1 + grad_worker2 + grad_worker3)/ N
Esto implementa Data Parallel Training.
===================================================*/
void promediar_gradientes(
    const vector<vector<float>>& resultados,
    float* grad_prom,
    size_t cant_params
) {
    memset(grad_prom, 0, cant_params * sizeof(float));

    int validos = 0;

    for (const auto& res : resultados) {
        if (res.size() == cant_params) {
            for (size_t i = 0; i < cant_params; i++) {
                grad_prom[i] += res[i];
            }
            validos++;
        }
    }

    if (validos > 0) {
        for (size_t i = 0; i < cant_params; i++) {
            grad_prom[i] /= (float)validos;
        }

        printf("[SERVER] Gradientes promediados: %d workers válidos\n", validos);
    } else {
        printf("[SERVER] No hubo gradientes válidos\n");
    }
}


/*=============================================
FUNCIÓN PRINCIPAL DEL SERVIDOR
Es llamada desde Python.
Flujo:                
                        Python
                          ↓
                server_process_batches()
                          ↓
                        workers
                           ↓
                       gradientes
                           ↓
                        promedio
                           ↓
                         Python
==============================================*/
extern "C" {

int server_process_batches(
    float* params_modelo_ptr,
    float* datos_batch_ptr,
    int* tam_batches_ptr,
    int num_batches,
    const char** workers_ip,
    int* workers_port,
    int num_workers,
    float* grad_prom_ptr
) {
    if (!params_modelo_ptr || !datos_batch_ptr || !tam_batches_ptr ||
        !workers_ip || !workers_port || !grad_prom_ptr) {
        fprintf(stderr, "[SERVER] Error: puntero NULL\n");
        return -1;
    }

    size_t cant_params = TOTAL_MODEL_PARAMS;

    EstadoWorker* workers = new EstadoWorker[num_workers];

    for (int i = 0; i < num_workers; i++) {
        workers[i].info.ip = workers_ip[i];
        workers[i].info.port = workers_port[i];
        workers[i].disponible = 1;
        workers[i].fallos = 0;

        printf("[SERVER] Worker %d registrado: %s:%d\n",
               i, workers[i].info.ip, workers[i].info.port);
    }

    vector<vector<float>> partes_batch;
    float* batch_ptr = datos_batch_ptr;

    for (int i = 0; i < num_batches; i++) {
        int tam = tam_batches_ptr[i];

        vector<float> batch(batch_ptr, batch_ptr + tam);
        partes_batch.push_back(batch);

        batch_ptr += tam;

        printf("[SERVER] Batch %d tiene %d elementos\n", i, tam);
    }

    vector<vector<float>> resultados;

    int ret = distribuir_batches(
        params_modelo_ptr,
        cant_params,
        partes_batch,
        workers,
        num_workers,
        resultados
    );

    if (ret == -1) {
        fprintf(stderr, "[SERVER] Fallo total\n");
        delete[] workers;
        return -3;
    }

    promediar_gradientes(
        resultados,
        grad_prom_ptr,
        cant_params
    );

    delete[] workers;

    printf("[SERVER] Proceso terminado\n");

    return ret;
}

}

// =====================================================
// MAIN DE PRUEBA
//
// NO se usa en producción.
//
// Solo sirve para probar:
//
// Server <-> Workers
//
// sin necesidad de ejecutar Python.
// =====================================================
int main() {
    printf("=== PRUEBA LOCAL SERVER CON N WORKERS ===\n");

    const int num_workers = 3;
    const int num_batches = 3;
    const int batch_samples = 50;

    const char* workers_ip[num_workers] = {
        "127.0.0.1",
        "127.0.0.1",
        "127.0.0.1"
    };

    int workers_port[num_workers] = {
        9001,
        9002,
        9003
    };

    int sample_size = INPUT_DIM + NUM_CLASSES;
    int tam_batch = batch_samples * sample_size;

    vector<float> params_modelo(TOTAL_MODEL_PARAMS, 0.01f);
    vector<float> datos_batch(num_batches * tam_batch, 0.0f);

    for (int b = 0; b < num_batches; b++) {
        for (int s = 0; s < batch_samples; s++) {
            int base = b * tam_batch + s * sample_size;

            for (int i = 0; i < INPUT_DIM; i++) {
                datos_batch[base + i] = 0.5f;
            }

            int clase = s % NUM_CLASSES;

            for (int c = 0; c < NUM_CLASSES; c++) {
                datos_batch[base + INPUT_DIM + c] = 0.0f;
            }

            datos_batch[base + INPUT_DIM + clase] = 1.0f;
        }
    }

    int tam_batches[num_batches];

    for (int i = 0; i < num_batches; i++) {
        tam_batches[i] = tam_batch;
    }

    vector<float> grad_prom(TOTAL_MODEL_PARAMS, 0.0f);

    int ret = server_process_batches(
        params_modelo.data(),
        datos_batch.data(),
        tam_batches,
        num_batches,
        workers_ip,
        workers_port,
        num_workers,
        grad_prom.data()
    );

    printf("[TEST] Retorno: %d\n", ret);

    printf("[TEST] Primeros 10 gradientes:\n");

    for (int i = 0; i < 10; i++) {
        printf("%f ", grad_prom[i]);
    }

    printf("\n");

    return 0;
}