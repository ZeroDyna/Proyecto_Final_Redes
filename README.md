## Descripción del proyecto

Sistema de entrenamiento de una red neuronal donde el **forward pass** se ejecuta en un modelo central, los resultados (activaciones/outputs) se distribuyen a **3–4 dispositivos** para recalcular gradientes localmente, y finalmente los gradientes se **promedian** antes de actualizar los pesos del modelo. Este patrón se conoce como entrenamiento distribuido sincrónico o *gradient averaging*.

---

## Flujo de entrenamiento

```
Modelo central
    │
    ▼  Forward pass → [outputs, activaciones]
    │
    ├──→ Dispositivo 1 → ∇L₁
    ├──→ Dispositivo 2 → ∇L₂
    ├──→ Dispositivo 3 → ∇L₃
    └──→ Dispositivo 4 → ∇L₄
                │
                ▼
    ∇L_avg = (∇L₁ + ∇L₂ + ∇L₃ + ∇L₄) / N
                │
                ▼
    W ← W − η · ∇L_avg
```

## Mapa completo

```
basicClasificacion.py
        │
        │  extrae matriz de pesos (numpy)
        ▼
   ucsp (C++ + pybind11)          ← ustedes construyen esto
        │
        │  UDP + RDT
        ├──→ Worker 1 → calcula su parte
        ├──→ Worker 2 → calcula su parte
        └──→ Worker 3 → calcula su parte
                │
                ▼
        promedia resultados
                │
        ▼
   Python recibe numpy matrix
        │
        ▼
   model actualiza pesos (W ← W - η·∇L)
```
