# -*- coding: utf-8 -*-
"""
tablas_y_figuras_compresion.py - Tablas y figuras de compresion a partir de evaluar_modelos.py
===========================================================================================
Lee todos los *_resumen.csv, *_detalle.csv y *_modelos.json de la carpeta de
resultados y genera, en <carpeta>/figuras:

  tabla_compresion.tex   Tabla LaTeX (booktabs): tamano, latencia, MLU, acuerdo
  tabla_energia.tex      Estimacion energetica basada en tiempos MEDIDOS
  fig_mejora_ospf.pdf    Mejora de MLU sobre OSPF por topologia y modelo (DRL y DRL+HC)
  fig_pareto.pdf         Tamano comprimido vs. brecha con la cota LP (tamano del punto = latencia)
  fig_latencia.pdf       Distribucion de la latencia de inferencia por modelo
  fig_tiempos.pdf        Tiempo de optimizacion por TM desglosado en fase DRL y fase HC
  fig_acuerdo.pdf        Coincidencia de decisiones con el modelo original

Uso:
  python tablas_y_figuras_compresion.py -d ../resultados/evaluacion_principal -potencia 45 -optim_anio 1000000
"""
import os
import glob
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3})
PALETA = ['#1b3a5c', '#d1495b', '#edae49', '#00798c', '#66a182', '#8d6a9f', '#7f7f7f', '#e07b39', '#2e86c1', '#a0522d']


def leer(d):
    rs = [pd.read_csv(f, sep=';', decimal=',') for f in sorted(glob.glob(os.path.join(d, '*_resumen.csv')))
          if '_eager' not in f]
    ds = [pd.read_csv(f, sep=';', decimal=',') for f in sorted(glob.glob(os.path.join(d, '*_detalle.csv')))
          if '_eager' not in f]
    if not rs:
        raise SystemExit(f"No hay *_resumen.csv en {d}")
    ms = {}
    for f in glob.glob(os.path.join(d, '*_modelos.json')):
        for m in json.load(open(f, encoding='utf-8')):
            ms[m['modelo']] = m
    lats = {}
    for f in glob.glob(os.path.join(d, '*_latencias.pckl')):
        if '_eager' in f:
            continue
        for k, v in pickle.load(open(f, 'rb')).items():
            lats.setdefault(k, []).extend(v)
    return pd.concat(rs, ignore_index=True), pd.concat(ds, ignore_index=True), ms, lats


def colores(modelos):
    return {m: PALETA[i % len(PALETA)] for i, m in enumerate(modelos)}


def guardar(fig, out, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(out, nombre + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out, nombre + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def kb(x):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x / 1024:.1f}"


def tabla_compresion(res, det, ms, lats, out):
    modelos = list(dict.fromkeys(res['modelo']))
    g = det.groupby('modelo')
    lineas = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
              r"Modelo & Par\'ametros & No nulos & FP32 (KB) & gzip (KB) & Lat. mediana (ms) & "
              r"Mejora DRL (\%) & Acuerdo (\%) \\", r"\midrule"]
    for m in modelos:
        s = ms.get(m, {})
        sub = g.get_group(m)
        mejora = ((sub['ospf'] - sub['drl_mejor']) / sub['ospf'] * 100).mean()
        acuerdo = 100 * sub['acuerdo'].mean() if sub['acuerdo'].notna().any() else float('nan')
        lat = np.median(np.array(lats.get(m, [np.nan])) * 1000)
        par = s.get('parametros')
        nz = s.get('no_nulos')
        fp32 = s.get('bytes_fp32') if s.get('tipo') == 'tf' else s.get('bytes_checkpoint_disco')
        lineas.append(f"{m} & {par if par else '--'} & {nz if nz else '--'} & {kb(fp32)} & "
                      f"{kb(s.get('bytes_gzip'))} & {lat:.2f} & {mejora:.1f} & "
                      f"{'--' if np.isnan(acuerdo) else f'{acuerdo:.1f}'} \\\\")
    lineas += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, 'tabla_compresion.tex'), 'w', encoding='utf-8').write('\n'.join(lineas))
    print('\n'.join(lineas))


def tabla_energia(det, out, potencia, n_anio):
    g = det.groupby('modelo', sort=False)
    filas = []
    for m, sub in g:
        filas.append((m, sub['t_drl_s'].mean(), sub['t_total_s'].mean()))
    if 'lp' in det and det['t_lp_s'].notna().any():
        filas.append(('Cota LP', det.groupby('tm')['t_lp_s'].first().mean(), np.nan))
    lin = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"M\'etodo & $t$ DRL (s) & $t$ DRL+HC (s) & $E$/optimizaci\'on (J) & $E$/a\~no (kWh) \\",
           r"\midrule"]
    for m, t_drl, t_tot in filas:
        t_ref = t_tot if not np.isnan(t_tot) else t_drl
        e_j = potencia * t_ref
        e_kwh = e_j * n_anio / 3.6e6
        lin.append(f"{m} & {t_drl:.3f} & {'--' if np.isnan(t_tot) else f'{t_tot:.3f}'} & {e_j:.1f} & {e_kwh:.2f} \\\\")
    lin += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, 'tabla_energia.tex'), 'w', encoding='utf-8').write('\n'.join(lin))
    print('\n'.join(lin))


def fig_mejora(det, out, col):
    d = det.copy()
    d['mej_drl'] = (d['ospf'] - d['drl_mejor']) / d['ospf'] * 100
    d['mej_hc'] = (d['ospf'] - d['drl_hc']) / d['ospf'] * 100
    tops = list(dict.fromkeys(d['topologia']))
    mods = list(dict.fromkeys(d['modelo']))
    w = 0.8 / len(mods)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(mods):
        sub = d[d['modelo'] == m].groupby('topologia')[['mej_drl', 'mej_hc']].mean().reindex(tops)
        x = np.arange(len(tops)) + (i - (len(mods) - 1) / 2) * w
        ax.bar(x, sub['mej_drl'], w * 0.95, color=col[m], label=m)
        if sub['mej_hc'].notna().any():
            ax.bar(x, sub['mej_hc'] - sub['mej_drl'], w * 0.95, bottom=sub['mej_drl'], color=col[m],
                   alpha=0.35, hatch='//', edgecolor=col[m])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(len(tops)))
    ax.set_xticklabels(tops)
    ax.set_ylabel('Reduccion de MLU sobre OSPF (%)')
    ax.legend(ncol=min(4, len(mods)), fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title('Parte solida: solo DRL | parte rayada: ganancia adicional del Hill Climbing', fontsize=9)
    guardar(fig, out, 'fig_mejora_ospf')


def fig_pareto(det, ms, lats, out, col):
    from matplotlib.ticker import ScalarFormatter
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k, (m, sub) in enumerate(det.groupby('modelo', sort=False)):
        if sub['lp'].isna().all():
            continue
        brecha = ((sub['drl_mejor'] - sub['lp']) / sub['lp'] * 100).mean()
        tam = ms.get(m, {}).get('bytes_gzip', np.nan) / 1024
        lat = np.median(np.array(lats.get(m, [np.nan])) * 1000)
        mk = 'o' if ms.get(m, {}).get('tipo') == 'tf' else 's'
        ax.scatter(tam, brecha, s=90, marker=mk, color=col[m], alpha=0.9, edgecolor='k', lw=0.6, zorder=3,
                   label=f"{m} ({lat:.1f} ms)")
    ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5),
              title='Modelo (latencia mediana)', title_fontsize=8)
    ax.margins(x=0.15, y=0.1)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xlabel('Tamano comprimido con gzip (KB, escala log)')
    ax.set_ylabel('Brecha media del DRL con la cota LP (%)')
    ax.set_title('Abajo a la izquierda = mejor (pequeno y cerca del optimo)', fontsize=9)
    guardar(fig, out, 'fig_pareto')


def fig_latencia(lats, out, col):
    mods = [m for m in col if m in lats]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    datos = [np.array(lats[m]) * 1000 for m in mods]
    bp = ax.boxplot(datos, showfliers=False, patch_artist=True, widths=0.55)
    for patch, m in zip(bp['boxes'], mods):
        patch.set_facecolor(col[m])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(mods) + 1))
    ax.set_xticklabels(mods, rotation=15)
    ax.set_yscale('log')
    ax.set_ylabel('Latencia por decision (ms, log)')
    guardar(fig, out, 'fig_latencia')


def fig_tiempos(det, out, col):
    tops = list(dict.fromkeys(det['topologia']))
    mods = list(dict.fromkeys(det['modelo']))
    fig, axs = plt.subplots(1, len(tops), figsize=(3.3 * len(tops), 3.6), sharey=False, squeeze=False)
    for ax, t in zip(axs[0], tops):
        sub = det[det['topologia'] == t].groupby('modelo')[['t_drl_s', 't_hc_s']].mean().reindex(mods)
        x = np.arange(len(mods))
        ax.bar(x, sub['t_drl_s'], color=[col[m] for m in mods], label='Fase DRL')
        ax.bar(x, sub['t_hc_s'], bottom=sub['t_drl_s'], color='lightgray', edgecolor='gray', label='Fase HC')
        ax.set_xticks(x)
        ax.set_xticklabels(mods, rotation=35, ha='right', fontsize=8)
        ax.set_title(t, fontsize=10)
    axs[0][0].set_ylabel('Tiempo medio por TM (s)')
    axs[0][0].legend(fontsize=8)
    guardar(fig, out, 'fig_tiempos')


def fig_acuerdo(det, out, col):
    d = det[det['acuerdo'].notna()]
    if d.empty:
        return
    tab = d.groupby(['topologia', 'modelo'])['acuerdo'].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    tab.plot(kind='bar', ax=ax, color=[col[m] for m in tab.columns], width=0.75)
    ax.set_ylabel('Decisiones iguales al original (%)')
    ax.set_xlabel('')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(fontsize=8, frameon=False)
    guardar(fig, out, 'fig_acuerdo')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', default='../resultados/evaluacion_principal')
    p.add_argument('-potencia', type=float, default=45.0, help='Potencia media de CPU supuesta (W)')
    p.add_argument('-optim_anio', type=float, default=1e6, help='Reoptimizaciones al ano')
    a = p.parse_args()
    out = os.path.join(a.d, 'figuras')
    os.makedirs(out, exist_ok=True)
    res, det, ms, lats = leer(a.d)
    col = colores(list(dict.fromkeys(det['modelo'])))
    tabla_compresion(res, det, ms, lats, out)
    tabla_energia(det, out, a.potencia, a.optim_anio)
    fig_mejora(det, out, col)
    fig_pareto(det, ms, lats, out, col)
    fig_latencia(lats, out, col)
    fig_tiempos(det, out, col)
    fig_acuerdo(det, out, col)
    print(f"\nFiguras y tablas en {out}")


if __name__ == '__main__':
    main()
