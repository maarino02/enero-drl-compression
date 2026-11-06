# -*- coding: utf-8 -*-
"""
medir_latencia_por_hilos.py - Paso 5: latencia TF vs TFLite con el mismo numero de hilos
==================================================================================
Lanza evaluar_modelos.py (sin Hill Climbing) para varias configuraciones de hilos,
con el MISMO numero de hilos en TensorFlow y en el interprete de TFLite:

  def : configuracion por defecto de ambos (la de la memoria)
  1   : 1 hilo en TF y en TFLite
  4   : 4 hilos
  8   : 8 hilos (nucleos fisicos del Ryzen 7 4800H)

Salida: ../resultados/latencia/h<cfg>/<Topologia>_{detalle,resumen,modelos,config}.csv/json
        (el CSV de detalle incluye 'lat_inv_mediana_ms' = latencia sin el redimensionado de TFLite)
Ejecutar SOLO, sin otros procesos pesados, para que las latencias sean comparables.
Si se interrumpe, al relanzarlo se salta lo que ya este hecho.

Uso:  python medir_latencia_por_hilos.py            (o con -n 5 -cfg def 1 4 8)
"""
import os
import sys
import argparse
import subprocess

D = '../datasets/principales'
TOPOS = ['EliBackbone', 'HurricaneElectric', 'Janetbackbone']
MODELOS = [
    ['-model', 'Estudiante', 'student', '../modelos/estudiante_semilla9/ckpt_STUDENT_V2_BEST-1'],
    ['-model', 'TFLite_FP32', 'tflite', '../modelos/cuantizados/actor_fp32.tflite'],
    ['-model', 'FP16', 'tflite', '../modelos/cuantizados/actor_fp16.tflite'],
    ['-model', 'INT8_dinamico', 'tflite', '../modelos/cuantizados/actor_dinamico.tflite'],
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', type=int, default=5, help='TMs por topologia')
    p.add_argument('-cfg', nargs='+', default=['def', '1', '4', '8'])
    p.add_argument('-out', default='../resultados/latencia')
    a = p.parse_args()
    for cfg in a.cfg:
        out = os.path.join(a.out, f"h{cfg}")
        for topo in TOPOS:
            if os.path.exists(os.path.join(out, f"{topo}_resumen.csv")):
                print(f"[ya hecho] h{cfg} {topo}")
                continue
            cmd = [sys.executable, 'evaluar_modelos.py', '-g', topo, '-f', f"{D}/NEW_{topo}/EVALUATE",
                   '-orig', '../modelos/base/ckpt_ACT-282', '-n', str(a.n), '-no_hc', '-out', out]
            for m in MODELOS:
                cmd += m
            if cfg != 'def':
                cmd += ['-threads', cfg, '-tf_threads', cfg]
            print(f"\n>>> hilos={cfg} | {topo}", flush=True)
            subprocess.run(cmd, check=True)
    print("\nPaso 5 terminado. Resumen con: python resumir_latencia_por_hilos.py")


if __name__ == '__main__':
    main()
