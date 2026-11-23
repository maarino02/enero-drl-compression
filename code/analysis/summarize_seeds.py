# -*- coding: utf-8 -*-
"""
analysis/summarize_seeds.py - Summary of step 7 (variability between distillation seeds)
========================================================================================
Joins the student of the thesis (seed 9, results/main) with the ones evaluated
in results/seeds/<tag> and computes, per topology and seed:
  - improvement over OSPF of the DRL stage and of DRL+HC (mean per TM)
  - MLU difference (DRL+HC) with the original and paired Wilcoxon p-value
and then the mean and standard deviation between seeds.

Output: figures/output/resumen_semillas.csv and resumen_semillas_agregado.csv
        figures/output/fig_semillas.png/.pdf
Usage (from code/):  python -m analysis.summarize_seeds
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

BASE = config.RESULTS_DIR + '/main'
SEM = config.RESULTS_DIR + '/seeds'
OUT = config.FIGURES_OUTPUT_DIR


def leer(carpeta):
    fs = sorted(glob.glob(os.path.join(carpeta, '*_detalle.csv')))
    return pd.concat([pd.read_csv(f, sep=';', decimal=',') for f in fs], ignore_index=True) if fs else None


def main():
    base = leer(BASE)
    orig = base[base.modelo == 'Original'].set_index(['topologia', 'tm'])
    estudiantes = {'s9 (memoria)': base[base.modelo == 'Estudiante'].set_index(['topologia', 'tm'])}
    for d in sorted(glob.glob(os.path.join(SEM, '*'))):
        if os.path.isdir(d):
            df = leer(d)
            if df is not None:
                estudiantes[os.path.basename(d)] = df.set_index(['topologia', 'tm'])

    filas = []
    for tag, est in estudiantes.items():
        for top in sorted(est.index.get_level_values(0).unique()):
            e, o = est.loc[top], orig.loc[top]
            o = o.loc[e.index]
            dif = e['drl_hc'] - o['drl_hc']
            try:
                p = wilcoxon(e['drl_hc'], o['drl_hc']).pvalue
            except ValueError:
                p = np.nan
            filas.append({'semilla': tag, 'topologia': top, 'n_tms': len(e),
                          'mejora_drl_%': (100 * (e['ospf'] - e['drl_mejor']) / e['ospf']).mean(),
                          'mejora_drlhc_%': (100 * (e['ospf'] - e['drl_hc']) / e['ospf']).mean(),
                          'mejora_drlhc_original_%': (100 * (o['ospf'] - o['drl_hc']) / o['ospf']).mean(),
                          'dif_mlu_vs_original': dif.mean(), 'p_wilcoxon': p,
                          'mejor_que_original': bool(p < 0.05 and dif.mean() < 0),
                          'peor_que_original': bool(p < 0.05 and dif.mean() > 0)})
    res = pd.DataFrame(filas)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, 'resumen_semillas.csv'), index=False, sep=';', decimal=',')
    agg = res.groupby('topologia')[['mejora_drl_%', 'mejora_drlhc_%', 'mejora_drlhc_original_%']].agg(['mean', 'std', 'min', 'max'])
    agg.to_csv(os.path.join(OUT, 'resumen_semillas_agregado.csv'), sep=';', decimal=',')
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(res.round(4).to_string(index=False))
        print('\n', agg.round(2))

    # Figure: DRL+HC improvement per topology; one point per seed, line = original
    tops = sorted(res.topologia.unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, top in enumerate(tops):
        sub = res[res.topologia == top]
        ax.scatter(np.full(len(sub), i) + np.linspace(-0.12, 0.12, len(sub)), sub['mejora_drlhc_%'],
                   color='#00798c', zorder=3, label='Estudiante (una semilla)' if i == 0 else None)
        ax.hlines(sub['mejora_drlhc_original_%'].iloc[0], i - 0.3, i + 0.3, color='#d1495b', lw=2,
                  label='Original' if i == 0 else None)
    ax.set_xticks(range(len(tops)))
    ax.set_xticklabels(tops)
    ax.set_ylabel('Mejora de la MLU sobre OSPF, DRL + BL (%)')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_semillas.{ext}'), dpi=200, bbox_inches='tight')
    print(f"\nFiles in {OUT}/")


if __name__ == '__main__':
    main()
