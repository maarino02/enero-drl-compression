# -*- coding: utf-8 -*-
"""
evaluar_estudiante_por_semilla.py - Paso 7: evaluacion de los estudiantes destilados con otras semillas
==========================================================================================
Evalua UN estudiante en las tres topologias de evaluacion (50 TMs, con Hill Climbing),
igual que en la evaluacion final de la memoria, pero sin volver a ejecutar el modelo
original (sus resultados ya estan en evaluacion_principal). Lanzar una vez por semilla,
cada una en su pestana; pueden ir en paralelo (aqui solo interesa la calidad, no el tiempo).

Salida: ../resultados/semillas/<tag>/<Topologia>_{detalle,resumen,modelos,config}...
Si se interrumpe, al relanzarlo se salta lo que ya este hecho.

Uso:
  python evaluar_estudiante_por_semilla.py -tag s1 -ckpt ../modelos/estudiante_semilla1/ckpt_STUDENT_V2_BEST-1
"""
import os
import sys
import argparse
import subprocess

D = '../datasets/principales'
TOPOS = ['EliBackbone', 'HurricaneElectric', 'Janetbackbone']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-tag', required=True, help='Nombre de la semilla, p. ej. s1')
    p.add_argument('-ckpt', required=True, help='Checkpoint del estudiante (..._BEST-1)')
    p.add_argument('-n', type=int, default=50)
    p.add_argument('-tf_threads', type=int, default=None, help='Hilos de TF (p. ej. 4 si hay varias pestanas)')
    p.add_argument('-out', default='../resultados/semillas')
    a = p.parse_args()
    if not os.path.exists(a.ckpt + '.index'):
        raise SystemExit(f"No existe {a.ckpt}.index: revisa la ruta del checkpoint")
    out = os.path.join(a.out, a.tag)
    for topo in TOPOS:
        if os.path.exists(os.path.join(out, f"{topo}_resumen.csv")):
            print(f"[ya hecho] {a.tag} {topo}")
            continue
        cmd = [sys.executable, 'evaluar_modelos.py', '-g', topo, '-f', f"{D}/NEW_{topo}/EVALUATE",
               '-sin_original', '-model', f"Estudiante_{a.tag}", 'student', a.ckpt,
               '-n', str(a.n), '-out', out]
        if a.tf_threads:
            cmd += ['-tf_threads', str(a.tf_threads)]
        print(f"\n>>> {a.tag} | {topo}", flush=True)
        subprocess.run(cmd, check=True)
    print(f"\nEvaluacion de {a.tag} terminada. Resumen conjunto con: python resumir_estudiantes_por_semilla.py")


if __name__ == '__main__':
    main()
