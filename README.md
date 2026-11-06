# Ejecución del proyecto paso a paso

Esta guía reproduce el trabajo completo: entrena el agente base, lo comprime de tres formas, evalúa todos los modelos, resuelve la formulación exacta y genera las tablas y figuras.

Los modelos entrenados y los resultados ya están incluidos, así que no hace falta ejecutarlo todo. Se puede empezar por cualquier paso a partir del 3 o ir directamente al paso 13 para regenerar las tablas y figuras. Cada paso sustituye los resultados que ya existen.

Duración orientativa en un portátil de 8 núcleos sin GPU: unas 87 horas el entrenamiento, 9 horas cada reentrenamiento de la poda, varias horas cada destilación y unas 20 horas la evaluación de las redes mayores.

## 1. Instalar

Se necesita Python 3.10 y [Conda](https://docs.conda.io). Todo se ejecuta en CPU.

Descarga el proyecto, abre una terminal en él y crea el entorno:

```bash
conda create -n compresion-enero python=3.10 -y
conda activate compresion-enero
pip install -r requirements.txt
```

Después, entra en el código. Todos los comandos siguientes se lanzan desde ahí:

```bash
cd codigo
```

## 2. Entrenar el agente base

Borra el registro y el estado del entrenamiento anterior:

```bash
python -c "import os; os.remove('../logs/entrenamiento.txt'); os.remove('../modelos/estado_entrenamiento/estado.pckl')"
```

Lanza el entrenamiento:

```bash
python entrenar_agente_base.py -max_iters 300
```

Si se interrumpe, se reanuda desde el último bloque de 20 episodios completado. Por ejemplo, tras el episodio 100:

```bash
python entrenar_agente_base.py -start_iter 100 -start_c 101 -max_iters 300
```

En el trabajo, el modelo base es el *checkpoint* 282 y el entrenamiento se detuvo en el episodio 292 por un error numérico, de modo que el 292 no debe usarse. Si entrenas de nuevo, los resultados no serán idénticos: elige tu mejor episodio y sustituye el número 282 en los pasos siguientes.

## 3. Comprobar la recompensa y obtener la referencia de OSPF

```bash
python medir_sesgo_recompensa.py -g BtAsiaPac -f ../datasets/principales/NEW_BtAsiaPac/EVALUATE
python medir_sesgo_recompensa.py -g Garr199905 -f ../datasets/principales/NEW_Garr199905/EVALUATE
python medir_sesgo_recompensa.py -g Goodnet -f ../datasets/principales/NEW_Goodnet/EVALUATE
python calcular_referencia_ospf.py
```

## 4. Elegir el modelo base

Compara los tres mejores candidatos en matrices que no se usaron durante el entrenamiento:

```bash
python evaluar_modelos.py -g BtAsiaPac -f ../datasets/principales/NEW_BtAsiaPac/EVALUATE -orig ../modelos/base/ckpt_ACT-193 -model C282 pruned ../modelos/base/ckpt_ACT-282 -model C252 pruned ../modelos/base/ckpt_ACT-252 -t0 20 -n 30 -no_hc -out ../resultados/seleccion_modelo_base
python evaluar_modelos.py -g Garr199905 -f ../datasets/principales/NEW_Garr199905/EVALUATE -orig ../modelos/base/ckpt_ACT-193 -model C282 pruned ../modelos/base/ckpt_ACT-282 -model C252 pruned ../modelos/base/ckpt_ACT-252 -t0 20 -n 30 -no_hc -out ../resultados/seleccion_modelo_base
python evaluar_modelos.py -g Goodnet -f ../datasets/principales/NEW_Goodnet/EVALUATE -orig ../modelos/base/ckpt_ACT-193 -model C282 pruned ../modelos/base/ckpt_ACT-282 -model C252 pruned ../modelos/base/ckpt_ACT-252 -t0 20 -n 30 -no_hc -out ../resultados/seleccion_modelo_base
```

## 5. Podar el modelo

Mide la calidad a distintos niveles de poda sin reentrenar:

```bash
python barrido_poda_sin_reentrenar.py
```

Poda al 50 % y al 90 % y reentrena cada modelo:

```bash
python podar_y_reentrenar.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -top BtAsiaPac ../datasets/principales/NEW_BtAsiaPac -top Garr199905 ../datasets/principales/NEW_Garr199905 -top Goodnet ../datasets/principales/NEW_Goodnet -sparsity 0.5 -episodes 80 -eval_tms 20 -out ../modelos/podado50
python podar_y_reentrenar.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -top BtAsiaPac ../datasets/principales/NEW_BtAsiaPac -top Garr199905 ../datasets/principales/NEW_Garr199905 -top Goodnet ../datasets/principales/NEW_Goodnet -sparsity 0.9 -episodes 80 -eval_tms 20 -out ../modelos/podado90
```

## 6. Cuantizar el modelo

```bash
python cuantizar_a_tflite.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -cal BtAsiaPac ../datasets/principales/NEW_BtAsiaPac/TRAIN -cal Garr199905 ../datasets/principales/NEW_Garr199905/TRAIN -cal Goodnet ../datasets/principales/NEW_Goodnet/TRAIN -out ../modelos/cuantizados
```

La variante `int8` termina con un error de TensorFlow Lite, que queda anotado en el informe. Es lo esperado: en el trabajo se usan las variantes FP32, FP16 y de rango dinámico.

## 7. Destilar el estudiante

Prueba las tres configuraciones:

```bash
python destilar_estudiante.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -top BtAsiaPac ../datasets/principales/NEW_BtAsiaPac -top Garr199905 ../datasets/principales/NEW_Garr199905 -top Goodnet ../datasets/principales/NEW_Goodnet -alpha 0.7 -temperature 2 -epochs 40 -tms_epoch 4 -eval_tms 20 -out ../modelos/estudiante_alpha07_tau2
python destilar_estudiante.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -top BtAsiaPac ../datasets/principales/NEW_BtAsiaPac -top Garr199905 ../datasets/principales/NEW_Garr199905 -top Goodnet ../datasets/principales/NEW_Goodnet -alpha 1 -temperature 2 -epochs 40 -tms_epoch 4 -eval_tms 20 -out ../modelos/estudiante_alpha1_tau2
python destilar_estudiante.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -top BtAsiaPac ../datasets/principales/NEW_BtAsiaPac -top Garr199905 ../datasets/principales/NEW_Garr199905 -top Goodnet ../datasets/principales/NEW_Goodnet -alpha 1 -temperature 4 -epochs 40 -tms_epoch 4 -eval_tms 20 -out ../modelos/estudiante_semilla9
```

Repite la configuración elegida con otras tres semillas. Estos comandos pueden ir en paralelo, cada uno en su terminal:

```bash
python destilar_estudiante.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -top BtAsiaPac ../datasets/principales/NEW_BtAsiaPac -top Garr199905 ../datasets/principales/NEW_Garr199905 -top Goodnet ../datasets/principales/NEW_Goodnet -alpha 1 -temperature 4 -epochs 40 -tms_epoch 4 -eval_tms 20 -seed 1 -tf_threads 4 -out ../modelos/estudiante_semilla1
python destilar_estudiante.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -top BtAsiaPac ../datasets/principales/NEW_BtAsiaPac -top Garr199905 ../datasets/principales/NEW_Garr199905 -top Goodnet ../datasets/principales/NEW_Goodnet -alpha 1 -temperature 4 -epochs 40 -tms_epoch 4 -eval_tms 20 -seed 2 -tf_threads 4 -out ../modelos/estudiante_semilla2
python destilar_estudiante.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 -top BtAsiaPac ../datasets/principales/NEW_BtAsiaPac -top Garr199905 ../datasets/principales/NEW_Garr199905 -top Goodnet ../datasets/principales/NEW_Goodnet -alpha 1 -temperature 4 -epochs 40 -tms_epoch 4 -eval_tms 20 -seed 3 -tf_threads 4 -out ../modelos/estudiante_semilla3
```

## 8. Evaluar todos los modelos

Ejecuta este paso sin otras cargas en el equipo, porque mide tiempos:

```bash
python evaluar_modelos.py -g EliBackbone -f ../datasets/principales/NEW_EliBackbone/EVALUATE -model Podado50 pruned ../modelos/podado50/ckpt_PRUNED_s50_BEST-1 -model Podado90 pruned ../modelos/podado90/ckpt_PRUNED_s90_BEST-1 -model Estudiante student ../modelos/estudiante_semilla9/ckpt_STUDENT_V2_BEST-1 -model TFLite_FP32 tflite ../modelos/cuantizados/actor_fp32.tflite -model FP16 tflite ../modelos/cuantizados/actor_fp16.tflite -model INT8_dinamico tflite ../modelos/cuantizados/actor_dinamico.tflite -n 50 -lp -ls
python evaluar_modelos.py -g HurricaneElectric -f ../datasets/principales/NEW_HurricaneElectric/EVALUATE -model Podado50 pruned ../modelos/podado50/ckpt_PRUNED_s50_BEST-1 -model Podado90 pruned ../modelos/podado90/ckpt_PRUNED_s90_BEST-1 -model Estudiante student ../modelos/estudiante_semilla9/ckpt_STUDENT_V2_BEST-1 -model TFLite_FP32 tflite ../modelos/cuantizados/actor_fp32.tflite -model FP16 tflite ../modelos/cuantizados/actor_fp16.tflite -model INT8_dinamico tflite ../modelos/cuantizados/actor_dinamico.tflite -n 50 -lp -ls
python evaluar_modelos.py -g Janetbackbone -f ../datasets/principales/NEW_Janetbackbone/EVALUATE -model Podado50 pruned ../modelos/podado50/ckpt_PRUNED_s50_BEST-1 -model Podado90 pruned ../modelos/podado90/ckpt_PRUNED_s90_BEST-1 -model Estudiante student ../modelos/estudiante_semilla9/ckpt_STUDENT_V2_BEST-1 -model TFLite_FP32 tflite ../modelos/cuantizados/actor_fp32.tflite -model FP16 tflite ../modelos/cuantizados/actor_fp16.tflite -model INT8_dinamico tflite ../modelos/cuantizados/actor_dinamico.tflite -n 50 -lp -ls
```

## 9. Resolver la formulación exacta

```bash
python resolver_milp.py -g EliBackbone -f ../datasets/principales/NEW_EliBackbone/EVALUATE -n 50 -scope criticas todas -tl 10
python resolver_milp.py -g HurricaneElectric -f ../datasets/principales/NEW_HurricaneElectric/EVALUATE -n 50 -scope criticas todas -tl 10
python resolver_milp.py -g Janetbackbone -f ../datasets/principales/NEW_Janetbackbone/EVALUATE -n 50 -scope criticas todas -tl 10
```

## 10. Evaluar los estudiantes de cada semilla

Pueden ir en paralelo, cada uno en su terminal:

```bash
python evaluar_estudiante_por_semilla.py -tag s1 -ckpt ../modelos/estudiante_semilla1/ckpt_STUDENT_V2_BEST-1 -tf_threads 4
python evaluar_estudiante_por_semilla.py -tag s2 -ckpt ../modelos/estudiante_semilla2/ckpt_STUDENT_V2_BEST-1 -tf_threads 4
python evaluar_estudiante_por_semilla.py -tag s3 -ckpt ../modelos/estudiante_semilla3/ckpt_STUDENT_V2_BEST-1 -tf_threads 4
```

Al terminar, resume los resultados:

```bash
python resumir_estudiantes_por_semilla.py
```

## 11. Medir la latencia

Ejecuta este paso sin otras cargas en el equipo:

```bash
python medir_latencia_por_hilos.py
python resumir_latencia_por_hilos.py
python diagnosticar_latencia_tflite.py
```

## 12. Evaluar las redes mayores

La primera orden mide tiempos y debe ir sin otras cargas. La segunda solo mide calidad:

```bash
python evaluar_redes_mayores.py -fase milp enero
python evaluar_redes_mayores.py -fase compresion -tf_threads 4
python analizar_redes_mayores.py
```

## 13. Generar las tablas y figuras

```bash
python tablas_y_figuras_compresion.py -potencia 23.9 -optim_anio 105120
python figuras_semillas_y_redes_mayores.py
python figuras_capitulo_resultados.py -escal ../resultados/redes_mayores/figuras/tabla_memoria_escalabilidad.csv
python dibujar_topologias.py -g EliBackbone
python dibujar_topologias.py -g HurricaneElectric
python dibujar_topologias.py -g Janetbackbone
```

## Créditos y licencias

Este proyecto es un trabajo académico independiente, sin relación con los autores de ENERO ni de Topology Zoo.

- **Código:** parte del código deriva de [ENERO](https://github.com/BNN-UPC/ENERO), publicado bajo licencia Apache 2.0, y se ha modificado en este trabajo. La licencia está en `LICENSE`, y `NOTICE` indica qué ficheros derivan de ENERO y qué se ha cambiado.
- **Datos:** las matrices de tráfico y las topologías son un subconjunto del [conjunto de datos público de ENERO](https://bnn.upc.edu/download/enero-dataset/). Las topologías proceden de [Internet Topology Zoo](https://doi.org/10.25909/30153949), publicado bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), y ENERO las adaptó a su formato.

Si usas este proyecto, cita también estos trabajos:

> P. Almasan, S. Xiao, X. Cheng, X. Shi, P. Barlet-Ros y A. Cabellos-Aparicio, «ENERO: Efficient real-time WAN routing optimization with Deep Reinforcement Learning», *Computer Networks*, vol. 214, 109166, 2022. <https://doi.org/10.1016/j.comnet.2022.109166>

> S. Knight, H. X. Nguyen, N. Falkner, R. Bowden y M. Roughan, «The Internet Topology Zoo», *IEEE Journal on Selected Areas in Communications*, vol. 29, n.º 9, pp. 1765-1775, 2011. <https://doi.org/10.1109/JSAC.2011.111002>
