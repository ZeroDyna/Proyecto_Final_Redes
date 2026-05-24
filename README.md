# Proyecto Final

## Red de flujo

```
Python (red neuronal)
     │
     │  extrae matrices de pesos
     ▼
Main (C++)  ──────────────────────────────────────────────────
     │                                                        │
     ├──► Worker 1: batch [a,b,c] → calcula ∇L₁             │
     ├──► Worker 2: batch [d,e,f] → calcula ∇L₂             │ via UDP + RDT
     └──► Worker 3: batch [g,h,i] → calcula ∇L₃             │
                         │                                    │
                         ▼                                    │
              ∇L_avg = (∇L₁ + ∇L₂ + ∇L₃) / 3 ◄─────────────
                         │
                         ▼
              Python actualiza pesos: W ← W − η · ∇L_avg
```

---

## ¿Qué hace cada integrante?

### Fabian — `ucsp.cpp` (LISTO)

Implementó la librería de transporte. Esta librería no sabe nada de redes neuronales,
solo sabe mover `float*` de un lado al otro con confiabilidad.

Funciones principales:
- `ucsp_send_data(...)` — envía un buffer a un worker y espera el resultado
- `ucsp_distribute(...)` — envía a todos los workers y promedia los gradientes
- `ucsp_listen(...)` — loop del worker esperando datos del Main

Características del protocolo:
- Datagramas de exactamente **500 bytes**
- **CRC32** para detectar corrupción
- **ACK / NACK / Timeout** para garantizar entrega (RDT)
- **Network Byte Order** (htons/htonl) para compatibilidad entre máquinas

---

### Integrante 2 — `main_server.cpp`

Es el coordinador. Recibe las matrices de Python, arma los buffers para cada worker y llama a `ucsp_distribute`.

Lo que debe hacer:
1. Recibir del lado Python: las matrices de pesos y el batch actual
2. Para cada worker, construir su buffer de entrada:
   ```
   buffer_worker_i = [model_params (12454 floats)] + [batch_i (tamaño variable)]
   ```
3. Llamar a `ucsp_distribute(...)` con esos buffers
4. Devolver `gradient_avg` a Python para que actualice los pesos

Archivos que necesita incluir:
```cpp
#include "ucsp.hpp"
#include "nn_config.hpp"
```

---

### Integrante 3 — `worker.cpp`

Es el esclavo. Recibe un buffer con los pesos del modelo + su batch, hace el forward pass
completo con backpropagation y devuelve los gradientes.

Lo que debe hacer:
1. Llamar a `ucsp_listen(port, elementos_a_recibir, elementos_a_devolver, mi_callback)`
2. Implementar `mi_callback(float* buffer, size_t n, float* result_out)` que:
   - Desempaqueta el buffer usando los offsets de `nn_config.hpp`
   - Corre forward pass por las 4 capas
   - Corre backpropagation
   - Escribe los gradientes en `result_out`

Cómo desempacar el buffer (usar `nn_config.hpp`):
```cpp
float* w1    = buffer + OFFSET_W1;
float* b1    = buffer + OFFSET_B1;
float* w2    = buffer + OFFSET_W2;
// ... etc
// el batch viene después de todos los parámetros del modelo
float* batch = buffer + TOTAL_MODEL_PARAMS;
```

Archivos que necesita incluir:
```cpp
#include "ucsp.hpp"
#include "nn_config.hpp"
```

---

### Integrante 4 — `bindings.cpp` + `distributed_trainer.py`

Es el integrador. Conecta Python con C++ usando pybind11 para que Python pueda
llamar a `ucsp_distribute` como si fuera una función Python normal.

`bindings.cpp` debe exponer:
```cpp
// Python llamará esto como: ucsp.distribute(model_buffer, batch_list, workers)
m.def("distribute", &ucsp_distribute_wrapper, "Distribuye y promedia gradientes");
```

`distributed_trainer.py` modifica `basicClasificacion.py` para que en el loop
de entrenamiento, en lugar de hacer `optimizer.step()` localmente, llame a ucsp:
```python
import ucsp  # la librería compilada

# Extraer pesos como numpy float32
model_buffer = extraer_pesos(model)

# Distribuir y recibir gradientes promediados
grad_avg = ucsp.distribute(model_buffer, batch_chunks, workers)

# Actualizar pesos en Python
actualizar_pesos(model, grad_avg)
```

### Guía para compilar los bindings:
```bash
pip install pybind11
python setup.py build_ext --inplace
```

---

## Constantes importantes (`nn_config.hpp`)

| Constante           | Valor  | Descripción              |
|---------------------|--------|--------------------------|
| `TOTAL_MODEL_PARAMS`| 12454  | floats totales del modelo |
| `INPUT_DIM`         | 14     | entradas de la red        |
| `HIDDEN1`           | 128    | neuronas capa 1           |
| `HIDDEN2`           | 64     | neuronas capa 2           |
| `HIDDEN3`           | 32     | neuronas capa 3           |
| `NUM_CLASSES`       | 3      | salidas de la red         |

---

## Cómo correr el proyecto (4 terminales en una PC)

```bash
# Terminal 1 — Worker en puerto 9001
./worker 9001

# Terminal 2 — Worker en puerto 9002
./worker 9002

# Terminal 3 — Worker en puerto 9003
./worker 9003

# Terminal 4 — Main + Python
python distributed_trainer.py
```

## Cómo compilar

```bash
# Compilar worker
g++ -o worker worker.cpp ucsp.cpp -lz

# Compilar main server
g++ -o main_server main_server.cpp ucsp.cpp -lz

# Compilar bindings para Python
python setup.py build_ext --inplace
```
