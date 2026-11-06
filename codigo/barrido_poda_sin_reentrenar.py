# -*- coding: utf-8 -*-
"""
barrido_poda_sin_reentrenar.py - Barrido de poda sin reentrenamiento (Figura fig_poda_barrido, Capitulo 7)
===========================================================================================
Reconstruido para el repositorio: el barrido original se lanzo con podar_y_reentrenar.py
-episodes 0 y sus valores solo quedaron en la consola. Este script repite esa medida.

Para cada nivel de poda construye el actor podado a partir del modelo base (mismas mascaras
que podar_y_reentrenar.py) y mide la mejora de la MLU sobre OSPF de la etapa DRL en las
matrices de validacion. Las estrellas de la figura son la mejor validacion tras reentrenar,
tomada de historial_poda_s50.csv y historial_poda_s90.csv.

Salida: ../resultados/barrido_poda/barrido_poda.csv
        ../resultados/barrido_poda/fig_poda_barrido.png

Uso (desde codigo/):
  python barrido_poda_sin_reentrenar.py                (mide y dibuja)
  python barrido_poda_sin_reentrenar.py -solo_figura   (dibuja a partir del CSV ya guardado)
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '../datasets/principales'
TOPOS = ['BtAsiaPac', 'Goodnet', 'Garr199905']
NIVELES = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]
COL = {'BtAsiaPac': '#00798c', 'Goodnet': '#edae49', 'Garr199905': '#66a182', 'media': '#1b3a5c', 'ft': '#d1495b'}
HIST = {'Podado50': '../modelos/podado50/historial_poda_s50.csv',
        'Podado90': '../modelos/podado90/historial_poda_s90.csv'}


def medir(a):
    import gymnasium as gym
    import podar_y_reentrenar as ft

    envs = {}
    for t in TOPOS:
        env = gym.make(ft.ENV_NAME).unwrapped
        env.generate_environment(f"{BASE}/NEW_{t}/EVALUATE", t, ft.EPISODE_LENGTH, ft.NUM_ACTIONS, ft.PCT_DEMANDS)
        env.top_K_critical_demands = True
        envs[t] = env
    filas = []
    for s in NIVELES:
        agente = ft.PrunedAgent(a.teacher_ckpt, a.teacher_ckpt.replace('ckpt_ACT', 'ckpt_CRT'), s)
        r = ft.evaluate(envs, agente, list(range(a.eval_tms)))
        ceros, total = ft.sparsity_report(agente.actor)
        fila = {'poda_por_matriz_%': 100 * s, 'parametros_a_cero_%': 100 * ceros / total,
                'mejora_media_%': float(np.mean([v[1] for v in r.values()]))}
        for t in TOPOS:
            fila[f'mejora_{t}_%'] = r[t][1]
            fila[f'mlu_{t}'] = r[t][0]
        filas.append(fila)
        print(fila, flush=True)
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    pd.DataFrame(filas).to_csv(a.csv, index=False, sep=';', decimal=',')


def dibujar(a):
    d = pd.read_csv(a.csv, sep=';', decimal=',')
    x = d['parametros_a_cero_%']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.3})
    fig, ax = plt.subplots(figsize=(7.9, 4.6))
    for t in TOPOS:
        ax.plot(x, d[f'mejora_{t}_%'], '-o', color=COL[t], ms=4, lw=1.5, alpha=0.85, label=t)
    ax.plot(x, d['mejora_media_%'], '-o', color=COL['media'], ms=7, lw=3, label='Media, sin reentrenamiento')
    base = float(d['mejora_media_%'].iloc[0])
    ax.axhline(base, color='gray', ls=':', lw=1.2)
    ax.text(1, base + 0.5, f'Modelo base ({base:.2f} %)'.replace('.', ','), color='gray', fontsize=9, va='bottom')
    etiqueta = 'Media, tras el reentrenamiento'
    for nivel, hist in zip((50.0, 90.0), HIST.values()):
        h = pd.read_csv(hist, sep=';', decimal=',')['mejora_val_%'].max()
        fila = d[d['poda_por_matriz_%'] == nivel].iloc[0]
        px, p0 = fila['parametros_a_cero_%'], fila['mejora_media_%']
        ax.annotate('', xy=(px, h - 0.7), xytext=(px, p0 + 0.3), arrowprops=dict(arrowstyle='->', color=COL['ft'], lw=1.8))
        ax.plot([px], [h], marker='*', color=COL['ft'], ms=18, mec='black', mew=0.6, ls='none', label=etiqueta)
        etiqueta = None
    ax.set_yticks(range(0, 36, 5))
    ax.set_xlabel('Parámetros del actor a cero (%)')
    ax.set_ylabel('Mejora de la MLU sobre OSPF (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, 'fig_poda_barrido.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figura en {a.out}/fig_poda_barrido.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-teacher_ckpt', default='../modelos/base/ckpt_ACT-282')
    p.add_argument('-eval_tms', type=int, default=20, help='TMs de validacion por topologia')
    p.add_argument('-csv', default='../resultados/barrido_poda/barrido_poda.csv')
    p.add_argument('-out', default='../resultados/barrido_poda')
    p.add_argument('-solo_figura', action='store_true')
    a = p.parse_args()
    if not a.solo_figura:
        medir(a)
    dibujar(a)


if __name__ == '__main__':
    main()
