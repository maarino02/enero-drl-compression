# -*- coding: utf-8 -*-
"""
figuras_semillas_y_redes_mayores.py - Figuras de la memoria con los resultados de los pasos 7 y 8 (octubre 2026)
=================================================================================================
Lee:
  evaluacion_principal/   evaluacion principal (3 topologias, 50 TMs)
  milp/            MILP en esas 3 topologias
  semillas/        estudiantes destilados con las semillas 1, 2 y 3
  redes_mayores/   7 topologias zoo (10 TMs): enero, milp, compresion
y escribe en ../resultados/figuras_memoria/:
  fig_semillas.(png|pdf)          mejora DRL+BL por topologia y semilla frente al original
  fig_escal_tiempo.(png|pdf)      tiempo por matriz de cada metodo, topologias ordenadas por tamano
  fig_escal_calidad.(png|pdf)     mejora sobre OSPF de cada metodo, mismas topologias
  fig_tipo_red.(png|pdf)          ganancia de ENERO sobre la BL frente al grado medio de la red
Uso: python figuras_semillas_y_redes_mayores.py
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '../resultados/figuras_memoria'
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
COL = {'orig': '#1b3a5c', 'est': '#00798c', 'bl': '#7f7f7f', 'mc': '#d1495b', 'mt': '#edae49', 'lp': '#66a182'}


def leer(p):
    fs = sorted(glob.glob(p))
    return pd.concat([pd.read_csv(f, sep=';', decimal=',') for f in fs], ignore_index=True) if fs else pd.DataFrame()


def guardar(fig, nombre):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{nombre}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def imp(ospf, x):
    return 100 * (ospf - x) / ospf


def fig_semillas(base):
    o = base[base.modelo == 'Original'].set_index(['topologia', 'tm'])
    est = {'9': base[base.modelo == 'Estudiante']}
    for s in ('1', '2', '3'):
        est[s] = leer(f'../resultados/semillas/s{s}/*_detalle.csv')
    tops = ['EliBackbone', 'HurricaneElectric', 'Janetbackbone']
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, t in enumerate(tops):
        oo = o.loc[t]
        ax.hlines(imp(oo.ospf, oo.drl_hc).mean(), i - 0.32, i + 0.32, color=COL['orig'], lw=2.5,
                  label='Original' if i == 0 else None)
        vals = []
        for k, (s, d) in enumerate(est.items()):
            dd = d[d.topologia == t]
            v = imp(dd.ospf, dd.drl_hc).mean()
            vals.append(v)
            ax.scatter(i - 0.18 + 0.12 * k, v, color=COL['est'], zorder=3, s=36,
                       label='Estudiante (una semilla)' if i == 0 and k == 0 else None)
            ax.annotate(s, (i - 0.18 + 0.12 * k, v), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=7, color=COL['est'])
        ax.scatter(i + 0.36, np.mean(vals), marker='D', color='k', s=30, zorder=3,
                   label='Media de las semillas' if i == 0 else None)
    ax.set_xticks(range(len(tops)))
    ax.set_xticklabels(tops)
    ax.set_ylabel('Mejora de la MLU sobre OSPF,\nENERO completo (%)')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    guardar(fig, 'fig_semillas')


def datos_escalabilidad(base):
    enero = pd.concat([base, leer('../resultados/redes_mayores/enero/*_detalle.csv')], ignore_index=True)
    milp = pd.concat([leer('../resultados/milp/*_milp.csv'), leer('../resultados/redes_mayores/milp/*_milp.csv')])
    car = pd.read_csv('../resultados/redes_mayores/figuras/caracteristicas_topologias.csv', sep=';', decimal=',')
    car = car.sort_values(['nodos', 'enlaces_dirigidos'])
    filas = []
    for t in car.topologia:
        e = enero[enero.topologia == t]
        o = e[e.modelo == 'Original']
        s = e[e.modelo == 'Estudiante']
        mc = milp[(milp.topologia == t) & (milp.ambito == 'criticas') & (milp.limite_s == 10)]
        mt = milp[(milp.topologia == t) & (milp.ambito == 'todas') & (milp.limite_s == 10)]
        filas.append({'topologia': t,
                      'orig': imp(o.ospf, o.drl_hc).mean(), 't_orig': o.t_total_s.mean(),
                      'est': imp(s.ospf, s.drl_hc).mean(), 't_est': s.t_total_s.mean(),
                      'bl': imp(o.ospf, o.ls_solo).mean(), 't_bl': o.t_ls_s.mean(),
                      'mc': imp(mc.ospf, mc.milp).mean(), 't_mc': mc.t_s.mean(),
                      'mt': imp(mt.ospf, mt.milp).mean(), 't_mt': mt.t_s.mean(),
                      'lp': imp(o.ospf, o.lp).mean()})
    return car.merge(pd.DataFrame(filas), on='topologia')


def fig_escalabilidad(d):
    x = np.arange(len(d))
    etiquetas = [f"{t}\n({n}, {g:.1f})" for t, n, g in zip(d.topologia, d.nodos, d.grado_medio)]
    series_t = [('t_orig', 'ENERO completo (original)', COL['orig'], 'o', '-'),
                ('t_est', 'ENERO completo (estudiante)', COL['est'], 's', '-'),
                ('t_bl', 'Búsqueda local desde OSPF', COL['bl'], '^', '--'),
                ('t_mc', 'MILP demandas críticas', COL['mc'], 'D', '-'),
                ('t_mt', 'MILP todas las demandas (10 s)', COL['mt'], 'v', '-')]
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, lab, col, m, ls in series_t:
        ax.plot(x, d[c], marker=m, ls=ls, color=col, label=lab)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylabel('Tiempo medio por matriz (s, log)')
    ax.legend(fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=3)
    guardar(fig, 'fig_escal_tiempo')

    series_q = [('orig', 'ENERO completo (original)', COL['orig'], 'o', '-'),
                ('est', 'ENERO completo (estudiante)', COL['est'], 's', '-'),
                ('bl', 'Búsqueda local desde OSPF', COL['bl'], '^', '--'),
                ('mc', 'MILP demandas críticas', COL['mc'], 'D', '-'),
                ('lp', 'Cota LP (≈ MILP todas)', COL['lp'], 'x', ':')]
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, lab, col, m, ls in series_q:
        ax.plot(x, d[c], marker=m, ls=ls, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylabel('Mejora media de la MLU sobre OSPF (%)')
    ax.legend(fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=3)
    guardar(fig, 'fig_escal_calidad')

    fig, ax = plt.subplots(figsize=(6.5, 4))
    gan = d.orig - d.bl
    sc = ax.scatter(d.grado_medio, gan, s=d.nodos * 6, c=d.nodos, cmap='viridis', edgecolor='k', zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r.topologia, (r.grado_medio, r.orig - r.bl), textcoords='offset points',
                    xytext=(6, 4), fontsize=7.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlim(2.0, 3.45)
    ax.set_xlabel('Grado medio de la red (enlaces por nodo)')
    ax.set_ylabel('Ganancia de ENERO sobre la búsqueda\nlocal desde OSPF (puntos)')
    fig.colorbar(sc, ax=ax, label='Número de nodos')
    guardar(fig, 'fig_tipo_red')


def main():
    os.makedirs(OUT, exist_ok=True)
    base = leer('../resultados/evaluacion_principal/*_detalle.csv')
    fig_semillas(base)
    d = datos_escalabilidad(base)
    d.to_csv('../resultados/redes_mayores/figuras/tabla_memoria_escalabilidad.csv', index=False, sep=';', decimal=',')
    with pd.option_context('display.width', 250, 'display.max_columns', None):
        print(d.round(2).to_string(index=False))
    fig_escalabilidad(d)
    print(f"Figuras en {OUT}")


if __name__ == '__main__':
    main()
