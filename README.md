# Proyecto Final Redes y Comunicacion

Entrenamiento de una red neuronal en PyTorch donde el cálculo del gradiente se
distribuye entre varios workers en C++ usando un protocolo propio sobre UDP
(stop-and-wait, con ACK/NACK/timeout y CRC32).

```
Python (red neuronal)
     │
     │  extrae matrices de pesos
     ▼
distributed_trainer.py ────────────────────────────────────────
     │                                                          │
     ├──► Worker 1 (puerto 9001): batch 1 → ∇L₁                │
     ├──► Worker 2 (puerto 9002): batch 2 → ∇L₂                │ vía UDP + RDT (ucsp)
     ├──► Worker 3 (puerto 9003): batch 3 → ∇L₃                │
     └──► Worker 4 (puerto 9004): batch 4 → ∇L₄                │
                         │                                      │
                         ▼                                      │
              ∇L_avg = (∇L₁+∇L₂+∇L₃+∇L₄) / 4 ◄──────────────────
                         │
                         ▼
              Python actualiza pesos: W ← W − η · ∇L_avg
```

---

## Estructura del repo

| Archivo                  | Qué es |
|---------------------------|--------|
| `ucsp.cpp` / `ucsp.hpp`   | Librería de transporte (UDP + RDT 3.0). No sabe nada de redes neuronales, solo mueve `float*` de forma confiable. Acá vive `ucsp_distribute`, el core real del proyecto. |
| `worker.cpp`              | Proceso esclavo: recibe pesos + batch, hace forward + backprop, devuelve gradientes. Se compila como binario independiente (`./worker`). |
| `bindings.cpp`            | Expone `ucsp_distribute` a Python vía pybind11 (`import ucsp`). |
| `nn_config.hpp`           | Dimensiones de la red y offsets del buffer serializado. |
| `distributed_trainer.py`  | Script de entrenamiento. Define la lista de workers, llama a `ucsp.distribute(...)`. |
| `basicClasificacion.py`   | Versión sin distribuir (entrenamiento local), de referencia. |
| `setup.py`                | Build de los bindings de Python (`ucsp.so`). |

---

## Requisitos previos

- **g++** con soporte C++17
- **zlib** (`-lz`, usado para el CRC32)
- **Python 3.9+** (probado también en 3.12 y 3.14)
- Paquetes de Python:
  ```bash
  pip install pybind11 torch numpy pandas matplotlib scikit-learn --break-system-packages
  ```

> En distros como Arch, `pip install` fuera de un venv requiere `--break-system-packages`
> (o usar un entorno virtual: `python -m venv venv && source venv/bin/activate`).

---

## Compilación

### 1. El binario del worker

Es el mismo binario para todos los workers — solo cambia el puerto al ejecutarlo:

```bash
g++ -std=c++17 -O2 -o worker worker.cpp ucsp.cpp -lz
```

### 2. El módulo de Python (`ucsp`)

```bash
python3 setup.py build_ext --inplace
```

Esto genera un archivo `ucsp.cpython-<version>-<arch>.so` en la carpeta del proyecto.
Es lo que importa `distributed_trainer.py` con `import ucsp`. El warning de
`sign-compare` al compilar es inofensivo, se puede ignorar.

Verificá que cargó bien:

```bash
python3 -c "import ucsp; print(ucsp.TOTAL_MODEL_PARAMS)"
# debe imprimir: 12454
```

---

## Cómo correr el proyecto (5 terminales)

```bash
# Terminales 1-4 — un worker por puerto
./worker 9001 16
./worker 9002 16
./worker 9003 16
./worker 9004 16

# Terminal 5 — entrenamiento
python3 distributed_trainer.py
```

El segundo argumento de `./worker` (`16`) es `SAMPLES_PER_WORKER` y debe coincidir
con el valor de `S` definido en `distributed_trainer.py`.

---
## Constantes de la red (`nn_config.hpp`)

| Constante            | Valor | Descripción                |
|-----------------------|-------|-----------------------------|
| `TOTAL_MODEL_PARAMS`  | 12454 | floats totales del modelo   |
| `INPUT_DIM`           | 14    | entradas de la red          |
| `HIDDEN1`             | 128   | neuronas capa 1             |
| `HIDDEN2`             | 64    | neuronas capa 2             |
| `HIDDEN3`             | 32    | neuronas capa 3             |
| `NUM_CLASSES`         | 3     | salidas de la red           |

## Protocolo UCSP (resumen)

- Datagramas de **exactamente 500 bytes** (`UCSP_DATAGRAM_SIZE`)
- **CRC32** para detectar corrupción
- **ACK / NACK / Timeout** (stop-and-wait, RDT 3.0)
- **Network Byte Order** (htons/htonl) para compatibilidad entre máquinas
- Timeout: 5000 ms, máx. 5 reintentos antes de marcar el worker como fallido

---
