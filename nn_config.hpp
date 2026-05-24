// nn_config.hpp
#pragma once

// ─── Dimensiones de la arquitectura ───────────────────────────
#define INPUT_DIM 14
#define HIDDEN1 128
#define HIDDEN2 64
#define HIDDEN3 32
#define NUM_CLASSES 3

// ─── Tamaños de las matrices de Pesos (W) ─────────────────────
// En PyTorch, una capa Linear(in, out) tiene pesos de tamaño (out x in)
// pero al aplanarse (flatten) en 1D, el tamaño es in * out.
#define W1_SIZE         (INPUT_DIM * HIDDEN1)     // 1792
#define W2_SIZE         (HIDDEN1 * HIDDEN2)       // 8192
#define W3_SIZE         (HIDDEN2 * HIDDEN3)       // 2048
#define W_LOGITS_SIZE   (HIDDEN3 * NUM_CLASSES)   // 96
#define W_LOGVARS_SIZE  (HIDDEN3 * NUM_CLASSES)   // 96

// ─── Tamaños de los vectores de Sesgos (b) ────────────────────
#define B1_SIZE         (HIDDEN1)                 // 128
#define B2_SIZE         (HIDDEN2)                 // 64
#define B3_SIZE         (HIDDEN3)                 // 32
#define B_LOGITS_SIZE   (NUM_CLASSES)             // 3
#define B_LOGVARS_SIZE  (NUM_CLASSES)             // 3

// ─── Parámetros Totales ───────────────────────────────────────
#define TOTAL_MODEL_PARAMS (W1_SIZE + B1_SIZE + \
                            W2_SIZE + B2_SIZE + \
                            W3_SIZE + B3_SIZE + \
                            W_LOGITS_SIZE + B_LOGITS_SIZE + \
                            W_LOGVARS_SIZE + B_LOGVARS_SIZE) // Total: 12454 floats

 // ─── Offsets en el buffer serializado ────────────────────────
// Define dónde empieza cada bloque dentro del float* buffer
// Orden: W1, B1, W2, B2, W3, B3, W_logits, B_logits, W_logvars, B_logvars
#define OFFSET_W1       0
#define OFFSET_B1       (OFFSET_W1      + W1_SIZE)        // 1792
#define OFFSET_W2       (OFFSET_B1      + B1_SIZE)        // 1920
#define OFFSET_B2       (OFFSET_W2      + W2_SIZE)        // 10112
#define OFFSET_W3       (OFFSET_B2      + B2_SIZE)        // 10176
#define OFFSET_B3       (OFFSET_W3      + W3_SIZE)        // 12224
#define OFFSET_W_LOGITS (OFFSET_B3      + B3_SIZE)        // 12256
#define OFFSET_B_LOGITS (OFFSET_W_LOGITS + W_LOGITS_SIZE) // 12352
#define OFFSET_W_LOGVARS (OFFSET_B_LOGITS + B_LOGITS_SIZE)// 12355
#define OFFSET_B_LOGVARS (OFFSET_W_LOGVARS + W_LOGVARS_SIZE) // 12451