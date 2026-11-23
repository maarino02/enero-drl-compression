# -*- coding: utf-8 -*-
"""
evaluation/run_thread_latency.py - Step 5: TF vs TFLite latency with the same number of threads
===============================================================================================
Launches evaluation.evaluate (without Hill Climbing) for several thread configurations,
with the SAME number of threads in TensorFlow and in the TFLite interpreter:

  def : default configuration of both (the one of the thesis)
  1   : 1 thread in TF and in TFLite
  4   : 4 threads
  8   : 8 threads (physical cores of the Ryzen 7 4800H)

Output: results/thread_latency/h<cfg>/<Topology>_{detalle,resumen,modelos,config}.csv/json
        (the detail CSV includes 'lat_inv_mediana_ms' = latency without the TFLite resizing)
Run it ALONE, without other heavy processes, so that the latencies are comparable.
If it is interrupted, when relaunched it skips what is already done.

Usage (from code/):  python -m evaluation.run_thread_latency            (or with -n 5 -cfg def 1 4 8)
"""
import os
import sys
import argparse
import subprocess

import config

D = config.MAIN_TOPOLOGIES_DIR
TOPOS = ['EliBackbone', 'HurricaneElectric', 'Janetbackbone']
MODELOS = [
    ['-model', 'Estudiante', 'student', config.MODELS_DIR + '/student/ckpt_STUDENT_V2_BEST-1'],
    ['-model', 'TFLite_FP32', 'tflite', config.MODELS_DIR + '/tflite/actor_fp32.tflite'],
    ['-model', 'FP16', 'tflite', config.MODELS_DIR + '/tflite/actor_fp16.tflite'],
    ['-model', 'INT8_dinamico', 'tflite', config.MODELS_DIR + '/tflite/actor_dinamico.tflite'],
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', type=int, default=5, help='TMs per topology')
    p.add_argument('-cfg', nargs='+', default=['def', '1', '4', '8'])
    p.add_argument('-out', default=config.RESULTS_DIR + '/thread_latency')
    a = p.parse_args()
    for cfg in a.cfg:
        out = os.path.join(os.path.abspath(a.out), f"h{cfg}")
        for topo in TOPOS:
            if os.path.exists(os.path.join(out, f"{topo}_resumen.csv")):
                print(f"[already done] h{cfg} {topo}")
                continue
            cmd = [sys.executable, '-m', 'evaluation.evaluate', '-g', topo, '-f', f"{D}/NEW_{topo}/EVALUATE",
                   '-orig', config.MODELS_DIR + '/base/ckpt_ACT-282', '-n', str(a.n), '-no_hc', '-out', out]
            for m in MODELOS:
                cmd += m
            if cfg != 'def':
                cmd += ['-threads', cfg, '-tf_threads', cfg]
            print(f"\n>>> threads={cfg} | {topo}", flush=True)
            subprocess.run(cmd, check=True, cwd=config.CODE_DIR)
    print("\nStep 5 finished. Summary with: python -m analysis.summarize_thread_latency")


if __name__ == '__main__':
    main()
