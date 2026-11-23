# Compresión de un agente DRL+GNN para ingeniería de tráfico: poda, cuantización y destilación de ENERO

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)

🇬🇧 [Read in English](README.md)

Trabajo Fin de Grado del Grado en Ingeniería de Tecnologías de Telecomunicación, **Universidad de Granada** (ETSIIT, 2026).
**Autor:** Marino Ramírez Pulido · **Tutor:** José Camacho Páez · **Calificación:** 10

📄 **Memoria completa:** [`thesis/proyecto.pdf`](thesis/proyecto.pdf)

> Esta página es un resumen en español. La instalación, la estructura del repositorio y los comandos para reproducir cada experimento están en el [README principal](README.md). Los comandos son los mismos.

---

## Resumen

[ENERO](https://github.com/BNN-UPC/ENERO) es un optimizador de ingeniería de tráfico para redes WAN. Un agente de aprendizaje por refuerzo profundo con una red neuronal de grafos (DRL+GNN) elige un nodo intermedio para cada demanda crítica, y después una búsqueda local (LS) refina la solución. Este trabajo estudia si las **técnicas de compresión de redes neuronales** permiten reducir el coste de ejecutar el agente sin empeorar el enrutamiento.

- **Migré ENERO a Gymnasium**, adapté el entrenamiento a CPU (unas 87 h) y **detecté y corregí un sesgo sistemático** en la función de recompensa.
- Apliqué **poda por magnitud** (50 y 90 %, con reentrenamiento PPO), **cuantización posterior al entrenamiento** (TFLite FP16 e INT8 de rango dinámico) y **destilación de conocimiento** hacia un estudiante con 4,6 veces menos parámetros.
- Evalué todos los modelos en **10 topologías no vistas** (de 20 a 43 nodos) frente a OSPF, la LS, una **cota de programación lineal (LP)** y la **formulación exacta MILP**, con contrastes estadísticos emparejados.

## Conclusiones principales

1. **La compresión conserva la calidad del enrutamiento.** Con la LS, todos los modelos comprimidos mejoran la utilización máxima de enlace (MLU) de OSPF entre un 32,6 y un 35,0 % de media, frente al 34,3 % del original.
2. **La destilación es la técnica más adecuada para una CPU convencional.** El estudiante acelera la inferencia **3,2 veces** y reduce **hasta un 25 %** la energía por optimización. La poda y la cuantización solo reducen el tamaño.
3. **La red neuronal no es el cuello de botella.** La LS ocupa en torno al 70 % del tiempo de optimización.
4. **El resultado negativo que importa:** en redes de este tamaño, **la formulación exacta MILP es preferible a ENERO**, original o comprimido. Es mejor en las 10 topologías, unas **70 veces más rápida** y **no necesita entrenamiento**. La compresión no cambia esta conclusión, porque la desventaja de ENERO está en el enfoque y no en el tamaño de su red neuronal.

| Modelo | Parámetros no nulos | Tamaño (gzip) | Latencia mediana por decisión | Mejora sobre OSPF (DRL + LS) |
|---|---:|---:|---:|---:|
| Original | 12 737 | 46,0 KB | 16,1 ms | 34,3 % |
| Podado 50 % | 6 577 | 28,3 KB | 16,1 ms | 35,0 % |
| Podado 90 % | 1 647 | 10,0 KB | 16,1 ms | 32,6 % |
| **Estudiante (destilación)** | **2 753** | **9,9 KB** | **5,1 ms** | **34,5 %** (media de 4 semillas) |
| TFLite FP16 | – | 50,5 KB | 72,2 ms | 34,5 % |
| TFLite INT8 dinámico | – | 27,7 KB | 121,6 ms | 34,0 % |

| | ENERO (original) | MILP, demandas críticas | MILP, todas las demandas | Cota LP |
|---|---:|---:|---:|---:|
| 3 topologías principales: mejora / tiempo | 34,3 % / 20,0 s | 40,3 % / **0,28 s** | **49,2 %** / 9,8 s | 49,3 % |
| 7 redes mayores: mejora / tiempo | 16,2 % / 68,5 s | 22,3 % / **0,99 s** | **28,7 %** / 6,5 s | 28,7 % |

## Licencia y créditos

Este repositorio es una **obra derivada de [ENERO](https://github.com/BNN-UPC/ENERO)** (Almasan *et al.*, Barcelona Neural Networking Center, UPC), publicada con licencia Apache 2.0, y se distribuye con la misma [licencia](LICENSE). Los cambios respecto al original se resumen en [`NOTICE`](NOTICE). Si utilizas este código, cita el artículo original de ENERO; la referencia está en el [README principal](README.md#acknowledgements-and-license).

## Contacto

**Marino Ramírez Pulido** · [maarinoraamirez02@gmail.com](mailto:maarinoraamirez02@gmail.com) · [LinkedIn](https://www.linkedin.com/in/marino-ramirez)
