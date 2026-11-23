# -*- coding: utf-8 -*-
"""
evaluation/run_seeds.py - Step 7: evaluation of the students distilled with other seeds
=======================================================================================
Evaluates ONE student on the three evaluation topologies (50 TMs, with Hill Climbing),
as in the final evaluation of the thesis, but without running the original model again
(its results are already in results/main). Launch it once per seed, each one in its own
terminal tab; they can run in parallel (here only the quality matters, not the time).

Output: results/seeds/<tag>/<Topology>_{detalle,resumen,modelos,config}...
If it is interrupted, when relaunched it skips what is already done.

Usage (from code/):
  python -m evaluation.run_seeds -tag s1 -ckpt models/student_seeds/s1/ckpt_STUDENT_V2_BEST-1
"""
import os
import sys
import argparse
import subprocess

import config

D = config.MAIN_TOPOLOGIES_DIR
TOPOS = ['EliBackbone', 'HurricaneElectric', 'Janetbackbone']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-tag', required=True, help='Name of the seed, e.g. s1')
    p.add_argument('-ckpt', required=True, help='Checkpoint of the student (..._BEST-1)')
    p.add_argument('-n', type=int, default=50)
    p.add_argument('-tf_threads', type=int, default=None, help='TF threads (e.g. 4 if there are several tabs)')
    p.add_argument('-out', default=config.RESULTS_DIR + '/seeds')
    a = p.parse_args()
    if not os.path.exists(a.ckpt + '.index'):
        raise SystemExit(f"{a.ckpt}.index does not exist: check the path of the checkpoint")
    out = os.path.join(os.path.abspath(a.out), a.tag)
    for topo in TOPOS:
        if os.path.exists(os.path.join(out, f"{topo}_resumen.csv")):
            print(f"[already done] {a.tag} {topo}")
            continue
        cmd = [sys.executable, '-m', 'evaluation.evaluate', '-g', topo, '-f', f"{D}/NEW_{topo}/EVALUATE",
               '-sin_original', '-model', f"Estudiante_{a.tag}", 'student', os.path.abspath(a.ckpt),
               '-n', str(a.n), '-out', out]
        if a.tf_threads:
            cmd += ['-tf_threads', str(a.tf_threads)]
        print(f"\n>>> {a.tag} | {topo}", flush=True)
        subprocess.run(cmd, check=True, cwd=config.CODE_DIR)
    print(f"\nEvaluation of {a.tag} finished. Joint summary with: python -m analysis.summarize_seeds")


if __name__ == '__main__':
    main()
