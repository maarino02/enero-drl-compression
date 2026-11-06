# -*- coding: utf-8 -*-
"""
figuras_capitulo_resultados.py - Figuras definitivas de la memoria (Capitulo 7)
=========================================================================
Regenera, con coma decimal y la terminologia de la memoria (etapa DRL, LS):
  fig_ent_recompensa.png  (Figura 7.1)
  fig_ent_mlu.png         (Figura 7.2)
  fig_mejora_ospf.png     (Figura 7.4)
  fig_tiempos.png         (Figura 7.6)
  fig_tipo_red.pdf/.png   (Figura 7.5, solo si se indica -escal)
  fig_calidad_tiempo.pdf/.png (comparacion con el MILP, solo si se indica -escal)

Uso (una linea, desde codigo/):
  python figuras_capitulo_resultados.py -log ../logs/entrenamiento.txt -res ../resultados/evaluacion_principal -escal ../resultados/redes_mayores/figuras/tabla_memoria_escalabilidad.csv -out ../resultados/figuras_memoria
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3})
C_NEW, C_REF = '#1b3a5c', '#d1495b'
PALETA = ['#1b3a5c', '#d1495b', '#edae49', '#00798c', '#66a182', '#8d6a9f', '#7f7f7f']
NOMBRES = {'TFLite_FP32': 'TFLite FP32', 'INT8_dinamico': 'INT8 dinámico'}


def coma(x, _pos=None):
    """Numero con coma decimal, sin ceros sobrantes y con signo menos tipografico."""
    s = f'{x:.10g}'
    if 'e' in s:
        s = f'{x:.2f}'
    return s.replace('-', '\u2212').replace('.', ',')


class FormatoComa(ScalarFormatter):
    """Igual que el formato por defecto (mismos decimales en todas las marcas), pero con coma."""
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', ',')


def ejes_coma(ax, x=True):
    for eje in ((ax.xaxis, ax.yaxis) if x else (ax.yaxis,)):
        f = FormatoComa(useOffset=False)
        f.set_scientific(False)
        eje.set_major_formatter(f)


def nombre(m):
    return NOMBRES.get(m, m)


def guardar(fig, out, nombre_fig):
    fig.tight_layout()
    fig.savefig(os.path.join(out, nombre_fig + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- entrenamiento
def leer_log(path):
    d = {}
    with open(path, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            p = line.strip().split(',')
            if len(p) >= 2 and p[1] != '' and not line.startswith('MAX'):
                try:
                    d.setdefault(p[0], []).append(float(p[1]))
                except ValueError:
                    pass
    n = len(d.get('REW', []))
    d = {k: np.array(v[:n]) for k, v in d.items()}
    if 'a' in d and np.isnan(d['a']).any():  # se descarta el episodio con NaN y los posteriores
        n = int(np.argmax(np.isnan(d['a'])))
        d = {k: v[:n] for k, v in d.items()}
    return d, n


def mm(x, w):
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode='valid')


def figuras_entrenamiento(a):
    d, n = leer_log(a.log)
    ep = np.arange(1, n + 1)
    w = a.w

    # Figura 7.1: retorno de validacion
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(ep, d['REW'], color=C_NEW, alpha=0.3, lw=0.8, label='Retorno por episodio')
    ax.plot(ep[w - 1:], mm(d['REW'], w), color=C_NEW, lw=2, label=f'Media móvil ({w} episodios)')
    ax.axhline(0, color=C_REF, ls='--', lw=1.2, label='Igual que OSPF (retorno 0)')
    ax.axvline(a.entropy_ep, color='gray', ls=':', lw=1.2)
    ax.text(a.entropy_ep - 3, 0.06, 'entropía\n0,1 → 0,01', fontsize=8, color='gray', ha='right',
            transform=ax.get_xaxis_transform())
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Retorno de validación')
    ejes_coma(ax)
    ax.legend(fontsize=8, loc='lower center', frameon=True)
    guardar(fig, a.out, 'fig_ent_recompensa')

    # Figura 7.2: MLU de validacion
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, d['MLUmean'], color=C_NEW, alpha=0.3, lw=0.8)
    ax.plot(ep[w - 1:], mm(d['MLUmean'], w), color=C_NEW, lw=2, label='MLU media (media móvil)')
    ax.plot(ep[w - 1:], mm(d['<'], w), color=C_NEW, lw=1.2, ls='-.', label='Peor matriz (media móvil)')
    ax.axhline(a.ospf_mlu, color=C_REF, ls='--', lw=1.2, label=f'OSPF, MLU media ({coma(round(a.ospf_mlu, 3))})')
    ax.axhline(a.ospf_max, color=C_REF, ls=':', lw=1.2, label=f'OSPF, peor matriz ({coma(round(a.ospf_max, 3))})')
    ax.axvline(a.entropy_ep, color='gray', ls=':', lw=1.2)
    ax.set_xlabel('Episodio')
    ax.set_ylabel('MLU (60 matrices de validación)')
    ejes_coma(ax)
    ax.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False)
    guardar(fig, a.out, 'fig_ent_mlu')


# ---------------------------------------------------------------- compresion
def leer_resultados(d):
    ds = [pd.read_csv(f, sep=';', decimal=',') for f in sorted(glob.glob(os.path.join(d, '*_detalle.csv')))
          if '_eager' not in f]
    if not ds:
        raise SystemExit(f'No hay *_detalle.csv en {d}')
    det = pd.concat(ds, ignore_index=True)
    return det[det['modelo'] != 'Original_v1']


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
        ax.bar(x, sub['mej_drl'], w * 0.95, color=col[m], label=nombre(m))
        ax.bar(x, sub['mej_hc'] - sub['mej_drl'], w * 0.95, bottom=sub['mej_drl'], color=col[m],
               alpha=0.35, hatch='//', edgecolor=col[m])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(len(tops)))
    ax.set_xticklabels(tops)
    ax.set_ylabel('Mejora de la MLU sobre OSPF (%)')
    ejes_coma(ax, x=False)
    ax.legend(ncol=4, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.set_title('Parte sólida: etapa DRL  |  parte rayada: mejora adicional de la LS', fontsize=9)
    guardar(fig, out, 'fig_mejora_ospf')


def fig_tiempos(det, out, col):
    tops = list(dict.fromkeys(det['topologia']))
    mods = list(dict.fromkeys(det['modelo']))
    fig, axs = plt.subplots(1, len(tops), figsize=(3.3 * len(tops), 3.6), squeeze=False)
    for ax, t in zip(axs[0], tops):
        sub = det[det['topologia'] == t].groupby('modelo')[['t_drl_s', 't_hc_s']].mean().reindex(mods)
        x = np.arange(len(mods))
        ax.bar(x, sub['t_drl_s'], color=[col[m] for m in mods], label='Etapa DRL')
        ax.bar(x, sub['t_hc_s'], bottom=sub['t_drl_s'], color='lightgray', edgecolor='gray', label='Etapa LS')
        ax.set_xticks(x)
        ax.set_xticklabels([nombre(m) for m in mods], rotation=35, ha='right', fontsize=8)
        ax.set_title(t, fontsize=10)
        ejes_coma(ax, x=False)
    axs[0][0].set_ylabel('Tiempo medio por matriz (s)')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor='#555555', label='Etapa DRL (color de cada modelo)'),
                        Patch(facecolor='lightgray', edgecolor='gray', label='Etapa LS')],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False)
    guardar(fig, out, 'fig_tiempos')


# ---------------------------------------------------------------- tipo de red (Figura 7.5)
# Desplazamiento de cada etiqueta (en puntos) para que no se solapen
OFFSET = {'WideJpn': (9, -9), 'GtsPoland': (9, 4), 'LambdaNet': (-9, 6), 'Sanet': (0, -15),
          'Cesnet200511': (10, 2), 'Evolink': (10, -4), 'Bics': (10, 2), 'EliBackbone': (8, 2),
          'Janetbackbone': (-10, 0), 'HurricaneElectric': (-10, 0)}
ALINEA = {'LambdaNet': 'right', 'Janetbackbone': 'right', 'HurricaneElectric': 'right', 'Sanet': 'center'}


def fig_tipo_red(path, out):
    t = pd.read_csv(path, sep=';', decimal=',')
    t['ganancia'] = t['orig'] - t['bl']          # puntos porcentuales: ENERO completo - LS sin agente
    fig, ax = plt.subplots(figsize=(7, 4.2))
    n = t['nodos']
    sc = ax.scatter(t['grado_medio'], t['ganancia'], c=n, cmap='viridis', s=(n - 14) * 11,
                    edgecolor='k', lw=0.6, zorder=3, vmin=n.min(), vmax=n.max())
    for _, r in t.iterrows():
        dx, dy = OFFSET.get(r['topologia'], (8, 2))
        ax.annotate(r['topologia'], (r['grado_medio'], r['ganancia']), xytext=(dx, dy),
                    textcoords='offset points', fontsize=8, ha=ALINEA.get(r['topologia'], 'left'),
                    va='center')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Grado medio de la red (enlaces por nodo)')
    ax.set_ylabel('Ganancia de ENERO sobre la LS\nsin agente (puntos porcentuales)')
    ax.margins(x=0.12, y=0.12)
    ejes_coma(ax)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Número de nodos')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_tipo_red.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'fig_tipo_red.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- comparacion con el MILP
METODOS = [  # (columna mejora, columna tiempo, etiqueta, color, marcador)
    ('orig', 't_orig', 'ENERO (original)', '#1b3a5c', 'o'),
    ('est', 't_est', 'ENERO (estudiante)', '#00798c', 's'),
    ('bl', 't_bl', 'LS sin agente', '#7f7f7f', 'D'),
    ('mc', 't_mc', 'MILP demandas críticas', '#e07b39', '^'),
    ('mt', 't_mt', 'MILP todas las demandas', '#a0522d', 'v'),
]


def fig_calidad_tiempo(path, out):
    from matplotlib.ticker import FuncFormatter
    from matplotlib.lines import Line2D
    t = pd.read_csv(path, sep=';', decimal=',')
    grupos = [('Topologías de evaluación principales', t.iloc[:3]), ('Redes mayores (30 a 43 nodos)', t.iloc[3:])]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=False)
    for ax, (titulo, g) in zip(axs, grupos):
        for c_m, c_t, lab, col, mk in METODOS:
            ax.scatter(g[c_t], g[c_m], color=col, marker=mk, s=28, alpha=0.45, edgecolor='none', zorder=2)
            ax.scatter(g[c_t].mean(), g[c_m].mean(), color=col, marker=mk, s=130, edgecolor='k', lw=0.8, zorder=3)
        ax.axhline(g['lp'].mean(), color='k', ls='--', lw=1)
        ax.text(0.03, g['lp'].mean(), f"Cota LP (media, {coma(round(g['lp'].mean(), 1))} %)", fontsize=8,
                va='bottom', transform=ax.get_yaxis_transform())
        ax.set_xscale('log')
        ax.set_xlim(0.05, 250)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: coma(x)))
        ax.yaxis.set_major_formatter(FormatoComa(useOffset=False))
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel('Tiempo por matriz (s, escala logarítmica)')
    axs[0].set_ylabel('Mejora de la MLU sobre OSPF (%)')
    h = [Line2D([], [], color=col, marker=mk, ls='', markersize=8, markeredgecolor='k', label=lab)
         for _, _, lab, col, mk in METODOS]
    h.append(Line2D([], [], color='k', ls='--', lw=1, label='Cota LP'))
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_calidad_tiempo.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out, 'fig_calidad_tiempo.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-log', default='../logs/entrenamiento.txt')
    p.add_argument('-res', default='../resultados/evaluacion_principal')
    p.add_argument('-out', default='../resultados/figuras_memoria')
    p.add_argument('-ospf_mlu', type=float, default=1.1239)
    p.add_argument('-ospf_max', type=float, default=1.1952)
    p.add_argument('-entropy_ep', type=int, default=242)
    p.add_argument('-w', type=int, default=10)
    p.add_argument('-escal', default=None, help='tabla_memoria_escalabilidad.csv (Figura 7.5)')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    figuras_entrenamiento(a)
    det = leer_resultados(a.res)
    col = {m: PALETA[i % len(PALETA)] for i, m in enumerate(dict.fromkeys(det['modelo']))}
    fig_mejora(det, a.out, col)
    fig_tiempos(det, a.out, col)
    if a.escal:
        fig_tipo_red(a.escal, a.out)
        fig_calidad_tiempo(a.escal, a.out)
    print(f'Figuras en {a.out}')


if __name__ == '__main__':
    main()
