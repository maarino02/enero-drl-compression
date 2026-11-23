# -*- coding: utf-8 -*-
"""
evaluation/run_scalability.py - Step 8: MILP vs ENERO depending on the size of the network
==========================================================================================
Topologies of the ENERO zoo set not used in training, sorted by size
(nodes / directed links). GtsPoland and Bics have the same number of nodes and very different density.

  WideJpn 30/66 | GtsPoland 33/74 (sparse) | Bics 33/96 (dense) | Evolink 37/90
  Cesnet200511 39/88 | LambdaNet 42/92 | Sanet 43/90

Arn, Cynet, Reuna, Carnet and BtLatinAmerica were discarded: in them the MLU of OSPF equals
the LP bound (the most loaded link is a bridge without an alternative), so no method can improve
anything and they are useless for the comparison (checked with 3 TMs on 6/10/2026).

The three evaluation topologies of the thesis (20-29 nodes) are already evaluated with 50 TMs
in results/main and results/milp, and they are added to the figures automatically.

Phases (-fase, several can be given):
  milp       : evaluation.milp, scopes criticas and todas, limits 10 and 60 s
  enero      : evaluation.evaluate with Original + Estudiante, LS from OSPF (-ls) and LP bound (-lp).
               MEASURES TIMES: run it with the computer idle (no other heavy processes).
  compresion : evaluation.evaluate with Podado50, Podado90 and INT8 dinamico (without the original).
               Only the quality matters: it can run in parallel with other tasks.

Output: results/scalability/{milp,enero,compresion}/...
If it is interrupted, when relaunched it skips what is already complete.

Usage (from code/):
  python -m evaluation.run_scalability -fase milp enero          (at night, alone)
  python -m evaluation.run_scalability -fase compresion          (another day, can run in parallel)
Then: python -m figures.scalability_figures
"""
import os
import sys
import argparse
import subprocess

import pandas as pd

import config

ZOO = config.ZOO_TOPOLOGIES_DIR
TOPOS = ['WideJpn', 'GtsPoland', 'Bics', 'Evolink', 'Cesnet200511', 'LambdaNet', 'Sanet']
ORIG = config.MODELS_DIR + '/base/ckpt_ACT-282'
EST = config.MODELS_DIR + '/student/ckpt_STUDENT_V2_BEST-1'
P50 = config.MODELS_DIR + '/pruned_50/ckpt_PRUNED_s50_BEST-1'
P90 = config.MODELS_DIR + '/pruned_90/ckpt_PRUNED_s90_BEST-1'
INT8 = config.MODELS_DIR + '/tflite/actor_dinamico.tflite'


def hecho_milp(out, topo, n):
    f = os.path.join(out, f"{topo}_milp.csv")
    if not os.path.exists(f):
        return False
    return len(pd.read_csv(f, sep=';', decimal=',')) >= 4 * n   # 2 scopes x 2 limits per TM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-fase', nargs='+', default=['milp', 'enero'], choices=['milp', 'enero', 'compresion'])
    p.add_argument('-n', type=int, default=10, help='TMs per topology')
    p.add_argument('-topos', nargs='+', default=TOPOS)
    p.add_argument('-tf_threads', type=int, default=None,
                   help='TF/TFLite threads in the compresion phase if it runs in parallel with other tasks')
    p.add_argument('-out', default=config.RESULTS_DIR + '/scalability')
    a = p.parse_args()
    py = sys.executable

    for fase in a.fase:
        out = os.path.join(os.path.abspath(a.out), fase)
        os.makedirs(out, exist_ok=True)
        for topo in a.topos:
            f = f"{ZOO}/{topo}"
            if fase == 'milp':
                if hecho_milp(out, topo, a.n):
                    print(f"[already done] milp {topo}")
                    continue
                cmd = [py, '-m', 'evaluation.milp', '-g', topo, '-f', f, '-n', str(a.n),
                       '-scope', 'criticas', 'todas', '-tl', '10', '60', '-out', out]
            else:
                if os.path.exists(os.path.join(out, f"{topo}_resumen.csv")):
                    print(f"[already done] {fase} {topo}")
                    continue
                cmd = [py, '-m', 'evaluation.evaluate', '-g', topo, '-f', f, '-n', str(a.n), '-out', out]
                if fase == 'enero':
                    cmd += ['-orig', ORIG, '-model', 'Estudiante', 'student', EST, '-ls', '-lp']
                else:
                    cmd += ['-sin_original', '-model', 'Podado50', 'pruned', P50,
                            '-model', 'Podado90', 'pruned', P90, '-model', 'INT8_dinamico', 'tflite', INT8]
                    if a.tf_threads:
                        cmd += ['-tf_threads', str(a.tf_threads), '-threads', str(a.tf_threads)]
            print(f"\n>>> {fase} | {topo}", flush=True)
            subprocess.run(cmd, check=True, cwd=config.CODE_DIR)
    print("\nFinished. Figures with: python -m figures.scalability_figures")


if __name__ == '__main__':
    main()
