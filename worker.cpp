#include "ucsp.hpp"
#include "nn_config.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace std;


//funcion de activacion
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

//funcion para realizar el backpropagation
float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}


//Convierte los resultados finales de la red en probabilidades
void softmax(const vector<float>& logits, vector<float>& probs) {
    float max_logit = logits[0];

    for (int i = 1; i < NUM_CLASSES; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum = 0.0f;

    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = exp(logits[i] - max_logit);
        sum += probs[i];
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] /= sum;
    }
}


//Funcion que se ejecuta cuando el servidor manda datos al worker
void mi_callback(float* buffer, size_t n, float* result_out) {
    cout << "[WORKER] Buffer recibido con " << n << " floats" << endl;

    //Desempaqueta los datos
    float* w1 = buffer + OFFSET_W1;
    float* b1 = buffer + OFFSET_B1;

    float* w2 = buffer + OFFSET_W2;
    float* b2 = buffer + OFFSET_B2;

    float* w3 = buffer + OFFSET_W3;
    float* b3 = buffer + OFFSET_B3;

    float* w_logits = buffer + OFFSET_W_LOGITS;
    float* b_logits = buffer + OFFSET_B_LOGITS;

    float* w_logvars = buffer + OFFSET_W_LOGVARS;
    float* b_logvars = buffer + OFFSET_B_LOGVARS;

    //el batch empieza despues de todos los pesos del modelo
    float* batch = buffer + TOTAL_MODEL_PARAMS;

    //
    size_t batch_elements = n - TOTAL_MODEL_PARAMS;
    size_t sample_size = INPUT_DIM + NUM_CLASSES;
    size_t batch_size = batch_elements / sample_size;

    cout << "[WORKER] Batch size: " << batch_size << endl;

    memset(result_out, 0, TOTAL_MODEL_PARAMS * sizeof(float));

    float* gw1 = result_out + OFFSET_W1;
    float* gb1 = result_out + OFFSET_B1;

    float* gw2 = result_out + OFFSET_W2;
    float* gb2 = result_out + OFFSET_B2;

    float* gw3 = result_out + OFFSET_W3;
    float* gb3 = result_out + OFFSET_B3;

    float* gw_logits = result_out + OFFSET_W_LOGITS;
    float* gb_logits = result_out + OFFSET_B_LOGITS;

    float* gw_logvars = result_out + OFFSET_W_LOGVARS;
    float* gb_logvars = result_out + OFFSET_B_LOGVARS;

    //worker procesa cada muestra (pasa datos por la red)
    for (size_t sample = 0; sample < batch_size; sample++) {
        float* x = batch + sample * sample_size;
        float* y = x + INPUT_DIM;

        vector<float> z1(HIDDEN1);
        vector<float> a1(HIDDEN1);

        vector<float> z2(HIDDEN2);
        vector<float> a2(HIDDEN2);

        vector<float> z3(HIDDEN3);
        vector<float> a3(HIDDEN3);

        vector<float> logits(NUM_CLASSES);
        vector<float> probs(NUM_CLASSES);

        for (int h = 0; h < HIDDEN1; h++) {
            float sum = b1[h];

            for (int i = 0; i < INPUT_DIM; i++) {
                //calculo de la capa 1
                sum += w1[h * INPUT_DIM + i] * x[i];
            }

            z1[h] = sum;
            a1[h] = relu(sum);
        }

        for (int h = 0; h < HIDDEN2; h++) {
            float sum = b2[h];

            for (int i = 0; i < HIDDEN1; i++) {
                sum += w2[h * HIDDEN1 + i] * a1[i];
            }

            z2[h] = sum;
            a2[h] = relu(sum);
        }

        for (int h = 0; h < HIDDEN3; h++) {
            float sum = b3[h];

            for (int i = 0; i < HIDDEN2; i++) {
                sum += w3[h * HIDDEN2 + i] * a2[i];
            }

            z3[h] = sum;
            a3[h] = relu(sum);
        }

        for (int c = 0; c < NUM_CLASSES; c++) {
            float sum = b_logits[c];

            for (int i = 0; i < HIDDEN3; i++) {
                sum += w_logits[c * HIDDEN3 + i] * a3[i];
            }

            logits[c] = sum;
        }

        softmax(logits, probs);

        vector<float> d_logits(NUM_CLASSES);

        for (int c = 0; c < NUM_CLASSES; c++) {
            //este calculo compara lo que predijo la red vs lo que debio predecir
            d_logits[c] = (probs[c] - y[c]) / (float)batch_size;
        }

        vector<float> d_a3(HIDDEN3, 0.0f);

        for (int c = 0; c < NUM_CLASSES; c++) {
            gb_logits[c] += d_logits[c];

            for (int i = 0; i < HIDDEN3; i++) {
                gw_logits[c * HIDDEN3 + i] += d_logits[c] * a3[i];
                d_a3[i] += d_logits[c] * w_logits[c * HIDDEN3 + i];
            }
        }

        vector<float> d_z3(HIDDEN3);

        for (int i = 0; i < HIDDEN3; i++) {
            d_z3[i] = d_a3[i] * relu_derivative(z3[i]);
        }

        vector<float> d_a2(HIDDEN2, 0.0f);

        for (int h = 0; h < HIDDEN3; h++) {
            gb3[h] += d_z3[h];

            for (int i = 0; i < HIDDEN2; i++) {
                gw3[h * HIDDEN2 + i] += d_z3[h] * a2[i];
                d_a2[i] += d_z3[h] * w3[h * HIDDEN2 + i];
            }
        }

        vector<float> d_z2(HIDDEN2);

        for (int i = 0; i < HIDDEN2; i++) {
            d_z2[i] = d_a2[i] * relu_derivative(z2[i]);
        }

        vector<float> d_a1(HIDDEN1, 0.0f);

        for (int h = 0; h < HIDDEN2; h++) {
            gb2[h] += d_z2[h];

            for (int i = 0; i < HIDDEN1; i++) {
                gw2[h * HIDDEN1 + i] += d_z2[h] * a1[i];
                d_a1[i] += d_z2[h] * w2[h * HIDDEN1 + i];
            }
        }

        vector<float> d_z1(HIDDEN1);

        for (int i = 0; i < HIDDEN1; i++) {
            d_z1[i] = d_a1[i] * relu_derivative(z1[i]);
        }

        for (int h = 0; h < HIDDEN1; h++) {
            gb1[h] += d_z1[h];

            for (int i = 0; i < INPUT_DIM; i++) {
                gw1[h * INPUT_DIM + i] += d_z1[h] * x[i];
            }
        }
    }

    for (int i = 0; i < W_LOGVARS_SIZE; i++) {
        gw_logvars[i] = 0.0f;
    }

    for (int i = 0; i < B_LOGVARS_SIZE; i++) {
        gb_logvars[i] = 0.0f;
    }

    cout << "[WORKER] Gradientes calculados correctamente" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Uso:" << endl;
        cout << "./worker <puerto> <cantidad_muestras_batch>" << endl;
        cout << endl;
        cout << "Ejemplo:" << endl;
        cout << "./worker 9001 50" << endl;
        return 1;
    }

    int port = atoi(argv[1]);
    int batch_samples = atoi(argv[2]);

    size_t elementos_batch = batch_samples * (INPUT_DIM + NUM_CLASSES);
    size_t elementos_a_recibir = TOTAL_MODEL_PARAMS + elementos_batch;
    size_t elementos_a_devolver = TOTAL_MODEL_PARAMS;

    cout << "=== WORKER UCSP ===" << endl;
    cout << "Puerto: " << port << endl;
    cout << "Muestras esperadas: " << batch_samples << endl;
    cout << "Elementos a recibir: " << elementos_a_recibir << endl;
    cout << "Elementos a devolver: " << elementos_a_devolver << endl;

    int ret = ucsp_listen(
        port,
        elementos_a_recibir,
        elementos_a_devolver,
        mi_callback
    );

    if (ret != 0) {
        cerr << "[WORKER] Error en ucsp_listen: " << ret << endl;
        return 1;
    }

    return 0;
}