// nn_config.hpp
#pragma once

// dimensiones de la red
#define INPUT_DIM 14
#define HIDDEN1 128
#define HIDDEN2 64
#define HIDDEN3 32
#define NUM_CLASSES 3

// tamaños de pesos
// linear(in, out) => in * out
#define W1_SIZE (INPUT_DIM * HIDDEN1)
#define W2_SIZE (HIDDEN1 * HIDDEN2)
#define W3_SIZE (HIDDEN2 * HIDDEN3)
#define W_LOGITS_SIZE (HIDDEN3 * NUM_CLASSES)
#define W_LOGVARS_SIZE (HIDDEN3 * NUM_CLASSES)

// tamaños de biases
#define B1_SIZE (HIDDEN1)
#define B2_SIZE (HIDDEN2)
#define B3_SIZE (HIDDEN3)
#define B_LOGITS_SIZE (NUM_CLASSES)
#define B_LOGVARS_SIZE (NUM_CLASSES)

// total de parametros del modelo
#define TOTAL_MODEL_PARAMS                                                     \
  (W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE + W3_SIZE + B3_SIZE + W_LOGITS_SIZE + \
   B_LOGITS_SIZE + W_LOGVARS_SIZE + B_LOGVARS_SIZE)

// offsets dentro del buffer serializado
// orden:
// W1, B1, W2, B2, W3, B3,
// W_logits, B_logits,
// W_logvars, B_logvars

#define OFFSET_W1 0

#define OFFSET_B1 (OFFSET_W1 + W1_SIZE)

#define OFFSET_W2 (OFFSET_B1 + B1_SIZE)

#define OFFSET_B2 (OFFSET_W2 + W2_SIZE)

#define OFFSET_W3 (OFFSET_B2 + B2_SIZE)

#define OFFSET_B3 (OFFSET_W3 + W3_SIZE)

#define OFFSET_W_LOGITS (OFFSET_B3 + B3_SIZE)

#define OFFSET_B_LOGITS (OFFSET_W_LOGITS + W_LOGITS_SIZE)

#define OFFSET_W_LOGVARS (OFFSET_B_LOGITS + B_LOGITS_SIZE)

#define OFFSET_B_LOGVARS (OFFSET_W_LOGVARS + W_LOGVARS_SIZE)
