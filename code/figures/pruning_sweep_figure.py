# -*- coding: utf-8 -*-
"""
figures/pruning_sweep_figure.py - Figure of the pruning sweep (fig_poda_barrido, Chapter 7)
===========================================================================================
Reads results/pruning_sweep/barrido_poda.csv (written by compression.pruning_sweep) and the
fine-tuning histories of the pruned models (models/pruned_50, models/pruned_90), and draws the
MLU improvement over OSPF vs. the percentage of actor parameters at zero, per topology and on
average, without retraining. The stars are the best validation after retraining at 50 and 90 %.

Output: figures/output/fig_poda_barrido.png
Usage (from code/):  python -m figures.pruning_sweep_figure
"""
import os
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

TOPOS = ['BtAsiaPac', 'Goodnet', 'Garr199905']
COL = {'BtAsiaPac': '#00798c', 'Goodnet': '#edae49', 'Garr199905': '#66a182', 'media': '#1b3a5c', 'ft': '#d1495b'}
HIST = {50.0: config.MODELS_DIR + '/pruned_50/historial_poda_s50.csv',
        90.0: config.MODELS_DIR + '/pruned_90/historial_poda_s90.csv'}


def main():
    p = argparse.ArgumentParser(description="Figure of the pruning sweep")
    p.add_argument('-csv', default=config.RESULTS_DIR + '/pruning_sweep/barrido_poda.csv')
    p.add_argument('-out', default=config.FIGURES_OUTPUT_DIR)
    a = p.parse_args()

    d = pd.read_csv(a.csv, sep=';', decimal=',')
    x = d['parametros_a_cero_%']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.3})
    fig, ax = plt.subplots(figsize=(7.9, 4.6))
    for t in TOPOS:
        ax.plot(x, d[f'mejora_{t}_%'], '-o', color=COL[t], ms=4, lw=1.5, alpha=0.85, label=t)
    ax.plot(x, d['mejora_media_%'], '-o', color=COL['media'], ms=7, lw=3, label='Media, sin reentrenamiento')
    base = float(d.loc[d['poda_por_matriz_%'] == 0, 'mejora_media_%'].iloc[0])
    ax.axhline(base, color='gray', ls=':', lw=1.2)
    ax.text(1, base + 0.5, f'Modelo base ({base:.2f} %)'.replace('.', ','), color='gray', fontsize=9, va='bottom')
    label = 'Media, tras el reentrenamiento'
    for level, hist in HIST.items():
        best = pd.read_csv(hist, sep=';', decimal=',')['mejora_val_%'].max()
        row = d[d['poda_por_matriz_%'] == level].iloc[0]
        px, p0 = row['parametros_a_cero_%'], row['mejora_media_%']
        ax.annotate('', xy=(px, best - 0.7), xytext=(px, p0 + 0.3),
                    arrowprops=dict(arrowstyle='->', color=COL['ft'], lw=1.8))
        ax.plot([px], [best], marker='*', color=COL['ft'], ms=18, mec='black', mew=0.6, ls='none', label=label)
        label = None
    ax.set_yticks(range(0, 36, 5))
    ax.set_xlabel('Parámetros del actor a cero (%)')
    ax.set_ylabel('Mejora de la MLU sobre OSPF (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'fig_poda_barrido.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure in {path}")


if __name__ == '__main__':
    main()
