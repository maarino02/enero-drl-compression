# -*- coding: utf-8 -*-
"""
analysis/summarize_thread_latency.py - Summary of step 5 (latency with the same number of threads)
==================================================================================================
Reads results/thread_latency/h<cfg>/*_latencias.pckl and *_detalle.csv and generates, in figures/output/:
  resumen_latencia.csv  median per decision (ms) per configuration, topology and model,
                        with and without the resizing of the TFLite interpreter, and time of the DRL stage
  fig_latencia_hilos.(png|pdf)
Usage (from code/):  python -m analysis.summarize_thread_latency
"""
import os
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

BASE = config.RESULTS_DIR + '/thread_latency'
OUT = config.FIGURES_OUTPUT_DIR
ORDEN = ['def', '1', '4', '8']


def main():
    filas = []
    for d in sorted(glob.glob(os.path.join(BASE, 'h*'))):
        cfg = os.path.basename(d)[1:]
        for f in glob.glob(os.path.join(d, '*_latencias.pckl')):
            topo = os.path.basename(f).replace('_latencias.pckl', '')
            lats = pickle.load(open(f, 'rb'))
            det = pd.read_csv(os.path.join(d, f"{topo}_detalle.csv"), sep=';', decimal=',')
            for mod, v in lats.items():
                sub = det[det.modelo == mod]
                filas.append({'hilos': cfg, 'topologia': topo, 'modelo': mod,
                              'lat_mediana_ms': float(np.median(v) * 1000),
                              'lat_p95_ms': float(np.percentile(v, 95) * 1000),
                              'lat_sin_redim_ms': float(sub['lat_inv_mediana_ms'].median())
                              if 'lat_inv_mediana_ms' in sub else np.nan,
                              't_drl_s': float(sub['t_drl_s'].mean()), 'n_decisiones': len(v)})
    res = pd.DataFrame(filas)
    if res.empty:
        raise SystemExit("There are no results in " + BASE)
    res['hilos'] = pd.Categorical(res['hilos'], [c for c in ORDEN if c in set(res['hilos'])], ordered=True)
    res = res.sort_values(['hilos', 'topologia', 'modelo'])
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, 'resumen_latencia.csv'), index=False, sep=';', decimal=',')
    piv = res.pivot_table(index='modelo', columns='hilos', values='lat_mediana_ms', aggfunc='median', observed=True)
    piv2 = res.pivot_table(index='modelo', columns='hilos', values='lat_sin_redim_ms', aggfunc='median', observed=True)
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(res.round(2).to_string(index=False))
        print("\nMedian latency per decision (ms), median of the three topologies:\n", piv.round(2))
        print("\nSame, without the TFLite resizing (ms):\n", piv2.round(2))

    fig, ax = plt.subplots(figsize=(7, 4))
    piv.T.plot(ax=ax, marker='o')
    ax.set_yscale('log')
    ax.set_xlabel('Hilos (TensorFlow y TFLite)')
    ax.set_ylabel('Latencia mediana por decision (ms, log)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_latencia_hilos.{ext}'), dpi=200, bbox_inches='tight')
    print(f"\nFiles in {OUT}/")


if __name__ == '__main__':
    main()
