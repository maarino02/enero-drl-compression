# -*- coding: utf-8 -*-
"""
resolver_milp.py - Optimo exacto del enrutamiento por punto intermedio (MILP)
====================================================================================
Complementa la cota LP de evaluar_modelos.py. La cota LP (flujo multicomodidad
divisible) no es alcanzable con un unico punto intermedio por demanda, por lo que la
"distancia a la cota LP" mezcla el error del optimizador con la holgura de la propia
cota. Este script resuelve el problema REAL que aborda ENERO, sobre el mismo modelo
de red del entorno (caminos mas cortos en saltos, candidatos sin bucles ni repetidos):

  min  theta
  s.a. sum_m y[k,m] = 1                                   para cada demanda k variable
       base_e + sum_{k,m} b_k * a[k,m,e] * y[k,m] <= theta * c_e   para cada enlace e
       y[k,m] en {0,1},  theta >= 0

  a[k,m,e] = veces que el camino SP(s_k,m)+SP(m,d_k) atraviesa el enlace e
  base_e   = carga de las demandas fijas (enrutadas por OSPF)

Ambitos (-scope):
  criticas : solo son variables las demandas criticas que ve el agente DRL (las de
             los 5 enlaces mas cargados, hasta el 15 %); el resto va por OSPF. Es el
             optimo del espacio de busqueda de la etapa DRL.
  todas    : todas las demandas son variables. Es el optimo del enrutamiento por punto
             intermedio, y su cota dual es una cota inferior mucho mas ajustada que la LP.

Se resuelve con scipy.optimize.milp (HiGHS), con limite de tiempo. Se guardan el valor
de la mejor solucion entera, la cota dual (cota inferior valida aunque no se llegue al
optimo), el gap y el tiempo. La solucion se verifica reaplicandola en GraphEnv-v15.

Ejemplo (una linea):
  python resolver_milp.py -g EliBackbone
     -f ../datasets/principales/NEW_EliBackbone/EVALUATE
     -n 50 -scope criticas todas -tl 10 60 -out ../resultados/milp
"""
import os
import time
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import gymnasium as gym
import gym_graph  # noqa: F401  (registra los entornos)
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

PCT_DEMANDS = 0.15
EPISODE_LENGTH = 100
NUM_ACTIONS = 100


def candidate_paths(env, s, d):
    """Lista de (middlepoint, aristas del camino) con el mismo orden que usa el agente."""
    out = []
    for mp in env.src_dst_k_middlepoints[f"{s}:{d}"]:
        p1 = list(env.shortest_paths[s, mp])
        nodes = p1 if mp == d else p1[:-1] + list(env.shortest_paths[mp, d])
        out.append((mp, [env.edgesDict[f"{u}:{v}"] for u, v in zip(nodes[:-1], nodes[1:])]))
    return out


def build_and_solve(env, variable_demands, time_limit):
    """Construye y resuelve el MILP. Devuelve dict con resultados y el enrutamiento."""
    E = env.numEdges
    cap = np.zeros(E)
    for i in env.graph:
        for j in env.graph[i]:
            cap[env.edgesDict[f"{i}:{j}"]] = env.links_bw[i][j]

    var_set = {(s, d) for s, d in variable_demands}
    base = np.zeros(E)
    N = env.numNodes
    for s in range(N):
        for d in range(N):
            if s != d and (s, d) not in var_set:
                sp = list(env.shortest_paths[s, d])
                for u, v in zip(sp[:-1], sp[1:]):
                    base[env.edgesDict[f"{u}:{v}"]] += env.TM[s][d]

    cols, rows_eq, rows_cap, vals_cap, meta = 0, [], [], [], []
    for k, (s, d) in enumerate(variable_demands):
        bw = float(env.TM[s][d])
        for mp, edges in candidate_paths(env, s, d):
            rows_eq.append((k, cols))
            for e in edges:
                rows_cap.append((e, cols))
                vals_cap.append(bw)
            meta.append((s, d, mp))
            cols += 1
    nv = cols + 1  # + theta
    K = len(variable_demands)

    A_eq = coo_matrix((np.ones(len(rows_eq)), ([r for r, _ in rows_eq], [c for _, c in rows_eq])), shape=(K, nv))
    r_cap = [r for r, _ in rows_cap] + list(range(E))
    c_cap = [c for _, c in rows_cap] + [cols] * E
    v_cap = vals_cap + list(-cap)
    A_cap = coo_matrix((v_cap, (r_cap, c_cap)), shape=(E, nv))

    c = np.zeros(nv)
    c[cols] = 1.0
    integrality = np.ones(nv)
    integrality[cols] = 0
    lb = np.zeros(nv)
    ub = np.ones(nv)
    ub[cols] = np.inf
    cons = [LinearConstraint(A_eq.tocsr(), 1, 1), LinearConstraint(A_cap.tocsr(), -np.inf, -base)]

    t0 = time.perf_counter()
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub),
               options={'time_limit': time_limit, 'disp': False})
    t = time.perf_counter() - t0

    routing, value = {}, np.nan
    if res.x is not None:
        value = float(res.x[cols])
        for j in range(cols):
            if res.x[j] > 0.5:
                s, d, mp = meta[j]
                if mp != d:
                    routing[f"{s}:{d}"] = mp
    bound = getattr(res, 'mip_dual_bound', np.nan)
    gap = getattr(res, 'mip_gap', np.nan)
    return {'milp': value, 'cota_dual': float(bound) if bound is not None else np.nan,
            'gap': float(gap) if gap is not None else np.nan, 't_s': t, 'estado': res.status,
            'mensaje': res.message, 'n_binarias': cols, 'n_demandas': K}, routing


def mlu_from_env(env):
    m = 0.0
    for i in env.graph:
        for j in env.graph[i]:
            m = max(m, env.graph[i][j][0]['utilization'] / env.links_bw[i][j])
    return float(m)


def main():
    p = argparse.ArgumentParser(description="Optimo MILP del enrutamiento por punto intermedio")
    p.add_argument('-g', required=True, help='Topologia (ej. EliBackbone)')
    p.add_argument('-f', required=True, help='Carpeta EVALUATE de la topologia')
    p.add_argument('-n', type=int, default=50, help='Numero de TMs')
    p.add_argument('-t0', type=int, default=0, help='Primera TM')
    p.add_argument('-scope', nargs='+', default=['criticas', 'todas'], choices=['criticas', 'todas'])
    p.add_argument('-tl', nargs='+', type=float, default=[60.0], help='Limites de tiempo (s); uno o varios')
    p.add_argument('-out', default='../resultados/milp')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    env16 = gym.make('GraphEnv-v16').unwrapped
    env16.generate_environment(a.f, a.g, EPISODE_LENGTH, NUM_ACTIONS, PCT_DEMANDS)
    env16.top_K_critical_demands = True
    env15 = gym.make('GraphEnv-v15').unwrapped
    env15.generate_environment(a.f, a.g, EPISODE_LENGTH, NUM_ACTIONS, PCT_DEMANDS)
    print(f"=== MILP punto intermedio | {a.g} | {env16.numNodes} nodos, {env16.numEdges} enlaces ===")

    rows = []
    out_csv = os.path.join(a.out, f"{a.g}_milp.csv")
    for tm in range(a.t0, a.t0 + a.n):
        env16.reset(options={'tm_id': tm})
        ospf = float(env16.edgeMaxUti[2])
        criticas = [(s, d) for s, d, _ in env16.list_eligible_demands]
        todas = [(s, d) for s in range(env16.numNodes) for d in range(env16.numNodes) if s != d]
        for scope in a.scope:
            dems = criticas if scope == 'criticas' else todas
            for tl in a.tl:
                r, routing = build_and_solve(env16, dems, tl)
                env15.reset_DRL_hill_sp(tm, dict(routing), [])
                check = mlu_from_env(env15)
                rows.append({'topologia': a.g, 'tm': tm, 'ambito': scope, 'limite_s': tl, 'ospf': ospf,
                             **r, 'mlu_verificada': check})
                print(f"  TM {tm:>2} | {scope:<8} tl={tl:>5.0f}s | OSPF={ospf:.4f} MILP={r['milp']:.4f} "
                      f"(verif. {check:.4f}) cota={r['cota_dual']:.4f} gap={r['gap']:.3%} "
                      f"t={r['t_s']:.1f}s bin={r['n_binarias']}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False, sep=';', decimal=',')
    print(f"\nResultados en {out_csv}")


if __name__ == '__main__':
    main()
