# -*- coding: utf-8 -*-
"""
analizar_redes_mayores.py - Figuras y tablas del paso 8 (y base para el analisis por tipo de red)
================================================================================================
Junta, por topologia:
  - las 3 topologias de evaluacion de la memoria (evaluacion_principal + milp, 50 TMs)
  - las topologias zoo de redes_mayores/{enero,milp,compresion} (10 TMs)
y genera en redes_mayores/figuras:

  caracteristicas_topologias.csv  nodos, enlaces, grado medio, densidad, diametro, demandas criticas...
  escalabilidad_resumen.csv       por topologia y metodo: mejora sobre OSPF, distancia a la cota LP,
                                  distancia al MILP de criticas y tiempo medio por TM
  fig_escal_tiempo.(png|pdf)      tiempo medio por TM frente al numero de nodos (escala log)
  fig_escal_distancia.(png|pdf)   distancia media a la cota LP frente al numero de nodos
  fig_escal_compresion.(png|pdf)  mejora DRL+BL de cada modelo comprimido frente al tamano

Uso: python analizar_redes_mayores.py
"""
import os
import glob
import math

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ZOO = '../datasets/zoo'
EVAL = '../datasets/principales'
OUT = '../resultados/redes_mayores'
FIG = os.path.join(OUT, 'figuras')
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})


def leer(patron):
    fs = sorted(glob.glob(patron))
    return pd.concat([pd.read_csv(f, sep=';', decimal=',') for f in fs], ignore_index=True) if fs else pd.DataFrame()


def ruta_grafo(topo):
    for f in (f"{EVAL}/NEW_{topo}/EVALUATE/{topo}.graph", f"{ZOO}/{topo}/{topo}.graph"):
        if os.path.exists(f):
            return f
    return None


def caracteristicas(topo):
    f = ruta_grafo(topo)
    lineas = open(f).read().split('\n')
    i = next(k for k, l in enumerate(lineas) if l.startswith('EDGES'))
    n = int(next(l for l in lineas if l.startswith('NODES')).split()[1])
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    caps = []
    for l in lineas[i + 2:]:
        p = l.split()
        if len(p) >= 5:
            g.add_edge(int(p[1]), int(p[2]))
            caps.append(float(p[4]))
    e = g.number_of_edges()
    u = g.to_undirected()
    return {'topologia': topo, 'nodos': n, 'enlaces_dirigidos': e, 'grado_medio': e / n,
            'densidad': e / (n * (n - 1)), 'diametro': nx.diameter(u),
            'camino_medio': nx.average_shortest_path_length(u),
            'cv_capacidad': float(np.std(caps) / np.mean(caps)),
            'demandas_criticas': math.ceil(n * (n - 1) * 0.15)}


def main():
    os.makedirs(FIG, exist_ok=True)
    # --- Resultados de ENERO / BL / LP (por TM) ---
    enero = pd.concat([leer('../resultados/evaluacion_principal/*_detalle.csv'),
                       leer(os.path.join(OUT, 'enero', '*_detalle.csv')),
                       leer(os.path.join(OUT, 'compresion', '*_detalle.csv'))], ignore_index=True)
    milp = pd.concat([leer('../resultados/milp/*_milp.csv'), leer(os.path.join(OUT, 'milp', '*_milp.csv'))],
                     ignore_index=True)
    if enero.empty:
        raise SystemExit("No hay resultados")
    topos = sorted(enero.topologia.unique())
    car = pd.DataFrame([caracteristicas(t) for t in topos]).sort_values('nodos')
    car.to_csv(os.path.join(FIG, 'caracteristicas_topologias.csv'), index=False, sep=';', decimal=',')
    print(car.round(3).to_string(index=False))

    # Referencia por TM: OSPF y cota LP (del modelo original o de cualquier fila de esa TM)
    ref = enero.dropna(subset=['lp']).groupby(['topologia', 'tm'])[['ospf', 'lp', 'ls_solo', 't_ls_s', 't_lp_s']].first()
    mc = milp[(milp.ambito == 'criticas')].groupby(['topologia', 'tm']).milp.min().rename('milp_crit')

    filas = []

    def anadir(topo, metodo, mlu, t, n):
        r = ref.loc[topo].loc[mlu.index]
        c = mc.loc[topo].reindex(mlu.index) if topo in mc.index.get_level_values(0) else pd.Series(np.nan, mlu.index)
        filas.append({'topologia': topo, 'metodo': metodo, 'n_tms': n,
                      'mejora_ospf_%': (100 * (r.ospf - mlu) / r.ospf).mean(),
                      'dist_lp_%': (100 * (mlu - r.lp) / r.lp).mean(),
                      'dist_milp_crit_%': (100 * (mlu - c) / c).mean(),
                      't_medio_s': float(np.mean(t))})

    for topo in topos:
        if topo not in ref.index.get_level_values(0):
            continue
        r = ref.loc[topo]
        anadir(topo, 'BL desde OSPF', r.ls_solo, r.t_ls_s, len(r))
        anadir(topo, 'Cota LP', r.lp, r.t_lp_s, len(r))
        for mod, sub in enero[enero.topologia == topo].groupby('modelo'):
            sub = sub.set_index('tm')
            anadir(topo, f"ENERO {mod}", sub.drl_hc, sub.t_total_s, len(sub))
        for (amb, tl), sub in milp[milp.topologia == topo].groupby(['ambito', 'limite_s']):
            sub = sub.set_index('tm')
            anadir(topo, f"MILP {amb} ({tl:.0f} s)", sub.milp, sub.t_s, len(sub))
    res = pd.DataFrame(filas).merge(car[['topologia', 'nodos', 'enlaces_dirigidos', 'grado_medio', 'densidad']],
                                    on='topologia')
    res.to_csv(os.path.join(FIG, 'escalabilidad_resumen.csv'), index=False, sep=';', decimal=',')

    # --- Figuras ---
    estilos = {'ENERO Original': ('#1b3a5c', 'o', '-'), 'ENERO Estudiante': ('#00798c', 's', '-'),
               'BL desde OSPF': ('#7f7f7f', '^', '--'), 'MILP criticas (10 s)': ('#d1495b', 'D', '-'),
               'MILP todas (10 s)': ('#edae49', 'v', '-'), 'MILP todas (60 s)': ('#e07b39', 'v', ':'),
               'Cota LP': ('#66a182', 'x', ':')}

    def curva(ax, met, col_y):
        s = res[res.metodo == met].sort_values('nodos')
        if s.empty:
            return
        c, m, ls = estilos[met]
        ax.plot(s.nodos, s[col_y], marker=m, ls=ls, color=c, label=met.replace('ENERO ', 'ENERO + BL, '))

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for met in estilos:
        curva(ax, met, 't_medio_s')
    ax.set_yscale('log')
    ax.set_xlabel('Numero de nodos')
    ax.set_ylabel('Tiempo medio por matriz de trafico (s, log)')
    ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'fig_escal_tiempo.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for met in estilos:
        if met != 'Cota LP':
            curva(ax, met, 'dist_lp_%')
    ax.set_xlabel('Numero de nodos')
    ax.set_ylabel('Distancia media a la cota LP (%)')
    ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'fig_escal_distancia.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    comp = res[res.metodo.str.startswith('ENERO') | (res.metodo == 'BL desde OSPF')]
    if not comp.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        for met, s in comp.groupby('metodo'):
            s = s.sort_values('nodos')
            ax.plot(s.nodos, s['mejora_ospf_%'], marker='o', label=met.replace('ENERO ', ''))
        ax.set_xlabel('Numero de nodos')
        ax.set_ylabel('Mejora media de la MLU sobre OSPF (%)')
        ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIG, f'fig_escal_compresion.{ext}'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    with pd.option_context('display.width', 220, 'display.max_columns', None):
        print(res.round(2).to_string(index=False))
    print(f"\nFiguras y tablas en {FIG}/")


if __name__ == '__main__':
    main()
