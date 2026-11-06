# -*- coding: utf-8 -*-
"""
evaluar_redes_mayores.py - Paso 8: MILP frente a ENERO segun el tamano de la red
============================================================================
Topologias del conjunto zoo de ENERO no usadas en el entrenamiento, ordenadas por tamano
(nodos / enlaces dirigidos). GtsPoland y Bics tienen el mismo numero de nodos y densidad muy distinta.

  WideJpn 30/66 | GtsPoland 33/74 (dispersa) | Bics 33/96 (densa) | Evolink 37/90
  Cesnet200511 39/88 | LambdaNet 42/92 | Sanet 43/90

Se descartaron Arn, Cynet, Reuna, Carnet y BtLatinAmerica: en ellas la MLU de OSPF coincide con
la cota LP (el enlace mas cargado es un puente sin alternativa), asi que ningun metodo puede mejorar
nada y no sirven para comparar (comprobado con 3 TMs el 6/10/2026).

Las tres topologias de evaluacion de la memoria (20-29 nodos) ya estan evaluadas con 50 TMs
en evaluacion_principal y milp, y se anaden a las figuras automaticamente.

Fases (-fase, se pueden indicar varias):
  milp       : resolver_milp.py, ambitos criticas y todas, limites 10 y 60 s
  enero      : evaluar_modelos.py con Original + Estudiante, BL desde OSPF (-ls) y cota LP (-lp).
               MIDE TIEMPOS: ejecutar con el ordenador libre (sin otros procesos pesados).
  compresion : evaluar_modelos.py con Podado50, Podado90 e INT8 dinamico (sin el original).
               Solo interesa la calidad: puede ir en paralelo con otras tareas.

Salida: ../resultados/redes_mayores/{milp,enero,compresion}/...
Si se interrumpe, al relanzarlo se salta lo que ya este completo.

Uso:
  python evaluar_redes_mayores.py -fase milp enero          (de noche, en solitario)
  python evaluar_redes_mayores.py -fase compresion          (otro dia, puede ir en paralelo)
Despues: python analizar_redes_mayores.py
"""
import os
import sys
import argparse
import subprocess

import pandas as pd

ZOO = '../datasets/zoo'
TOPOS = ['WideJpn', 'GtsPoland', 'Bics', 'Evolink', 'Cesnet200511', 'LambdaNet', 'Sanet']
ORIG = '../modelos/base/ckpt_ACT-282'
EST = '../modelos/estudiante_semilla9/ckpt_STUDENT_V2_BEST-1'
P50 = '../modelos/podado50/ckpt_PRUNED_s50_BEST-1'
P90 = '../modelos/podado90/ckpt_PRUNED_s90_BEST-1'
INT8 = '../modelos/cuantizados/actor_dinamico.tflite'


def hecho_milp(out, topo, n):
    f = os.path.join(out, f"{topo}_milp.csv")
    if not os.path.exists(f):
        return False
    return len(pd.read_csv(f, sep=';', decimal=',')) >= 4 * n   # 2 ambitos x 2 limites por TM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-fase', nargs='+', default=['milp', 'enero'], choices=['milp', 'enero', 'compresion'])
    p.add_argument('-n', type=int, default=10, help='TMs por topologia')
    p.add_argument('-topos', nargs='+', default=TOPOS)
    p.add_argument('-tf_threads', type=int, default=None,
                   help='Hilos de TF/TFLite en la fase compresion si va en paralelo con otras tareas')
    p.add_argument('-out', default='../resultados/redes_mayores')
    a = p.parse_args()
    py = sys.executable

    for fase in a.fase:
        out = os.path.join(a.out, fase)
        os.makedirs(out, exist_ok=True)
        for topo in a.topos:
            f = f"{ZOO}/{topo}"
            if fase == 'milp':
                if hecho_milp(out, topo, a.n):
                    print(f"[ya hecho] milp {topo}")
                    continue
                cmd = [py, 'resolver_milp.py', '-g', topo, '-f', f, '-n', str(a.n),
                       '-scope', 'criticas', 'todas', '-tl', '10', '60', '-out', out]
            else:
                if os.path.exists(os.path.join(out, f"{topo}_resumen.csv")):
                    print(f"[ya hecho] {fase} {topo}")
                    continue
                cmd = [py, 'evaluar_modelos.py', '-g', topo, '-f', f, '-n', str(a.n), '-out', out]
                if fase == 'enero':
                    cmd += ['-orig', ORIG, '-model', 'Estudiante', 'student', EST, '-ls', '-lp']
                else:
                    cmd += ['-sin_original', '-model', 'Podado50', 'pruned', P50,
                            '-model', 'Podado90', 'pruned', P90, '-model', 'INT8_dinamico', 'tflite', INT8]
                    if a.tf_threads:
                        cmd += ['-tf_threads', str(a.tf_threads), '-threads', str(a.tf_threads)]
            print(f"\n>>> {fase} | {topo}", flush=True)
            subprocess.run(cmd, check=True)
    print("\nTerminado. Figuras con: python analizar_redes_mayores.py")


if __name__ == '__main__':
    main()
