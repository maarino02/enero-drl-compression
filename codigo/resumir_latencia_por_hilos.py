# -*- coding: utf-8 -*-
"""
resumir_latencia_por_hilos.py - Resumen del paso 5 (latencia con el mismo numero de hilos)
=====================================================================================
Lee latencia/h<cfg>/*_latencias.pckl y *_detalle.csv y genera:
  resumen_latencia.csv  mediana por decision (ms) por configuracion, topologia y modelo,
                        con y sin el redimensionado del interprete TFLite, y tiempo de la etapa DRL
  fig_latencia_hilos.(png|pdf)
Uso: python resumir_latencia_por_hilos.py
"""
import os
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '../resultados/latencia'
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
        raise SystemExit("No hay resultados en " + BASE)
    res['hilos'] = pd.Categorical(res['hilos'], [c for c in ORDEN if c in set(res['hilos'])], ordered=True)
    res = res.sort_values(['hilos', 'topologia', 'modelo'])
    res.to_csv(os.path.join(BASE, 'resumen_latencia.csv'), index=False, sep=';', decimal=',')
    piv = res.pivot_table(index='modelo', columns='hilos', values='lat_mediana_ms', aggfunc='median', observed=True)
    piv2 = res.pivot_table(index='modelo', columns='hilos', values='lat_sin_redim_ms', aggfunc='median', observed=True)
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(res.round(2).to_string(index=False))
        print("\nLatencia mediana por decision (ms), mediana de las tres topologias:\n", piv.round(2))
        print("\nIdem, sin el redimensionado de TFLite (ms):\n", piv2.round(2))

    fig, ax = plt.subplots(figsize=(7, 4))
    piv.T.plot(ax=ax, marker='o')
    ax.set_yscale('log')
    ax.set_xlabel('Hilos (TensorFlow y TFLite)')
    ax.set_ylabel('Latencia mediana por decision (ms, log)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(BASE, f'fig_latencia_hilos.{ext}'), dpi=200, bbox_inches='tight')
    print(f"\nFicheros en {BASE}/")


if __name__ == '__main__':
    main()
