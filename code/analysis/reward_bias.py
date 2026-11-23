# -*- coding: utf-8 -*-
"""
analysis/reward_bias.py - Quantifies the bias of the reward of GraphEnv-v16
===========================================================================
Runs GraphEnv-v16 and GraphEnv-v17 in parallel on the same traffic matrices and
with EXACTLY the same actions. As the dynamics of both environments are identical
(only the computation of the reward changes), at each step:

    bias_t = r_t(v16) - r_t(v17) = 10 * (sigma_partial_{t-1} - sigma_complete_{t-1})

Available policies:
  ospf      : for each demand, the candidate that reproduces the shortest path is
              chosen (nothing is changed). The correct reward (v17) is 0 at
              every step, so any reward of v16 is pure bias.
  aleatoria : uniform random action among the candidates (fixed seed).

It also checks the telescopic property of v17:
    sum_t r_t(v17) = 10*(MLU_OSPF - MLU_final) + 10*(sigma_OSPF - sigma_final)

Example (from code/, one topology; repeat for the other two):
  python -m analysis.reward_bias -g BtAsiaPac ^
     -f <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_BtAsiaPac/EVALUATE ^
     -n 20 -policy ospf aleatoria

It does not need any trained model. It does not interfere with a running training
(it only uses the CPU lightly), although it will slow it down a little while it runs.
"""
import os
import argparse
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import gymnasium as gym
import gym_graph  # noqa: F401  (registers the environments)

import config

PCT_DEMANDS = 0.15
EPISODE_LENGTH = 100
NUM_ACTIONS = 100
SEED = 9


def make_env(name, folder, topo):
    env = gym.make(name).unwrapped
    env.generate_environment(folder, topo, EPISODE_LENGTH, NUM_ACTIONS, PCT_DEMANDS)
    env.top_K_critical_demands = True
    return env


def action_keep_sp(env, src, dst):
    """Index of the candidate whose path matches the shortest path src->dst."""
    cands = env.src_dst_k_middlepoints[str(src) + ':' + str(dst)]
    sp = list(env.shortest_paths[src, dst])
    for i, mp in enumerate(cands):
        if mp == dst:
            return i
        path = list(env.shortest_paths[src, mp])[:-1] + list(env.shortest_paths[mp, dst])
        if path == sp:
            return i
    return None


def norm_std(env):
    u = np.divide(env.edge_state[:, 0], env.edge_state[:, 1],
                  out=np.zeros_like(env.edge_state[:, 0]), where=env.edge_state[:, 1] != 0)
    return float(np.std(u))


def run_episode(e16, e17, tm_id, policy, rng):
    _, i16 = e16.reset(options={'tm_id': tm_id})
    _, i17 = e17.reset(options={'tm_id': tm_id})
    src, dst = i17['src'], i17['dst']
    demand = i17.get('demand', i17.get('bw'))
    assert (i16['src'], i16['dst']) == (src, dst), "The environments are not synchronized"

    mlu_ospf = float(e17.edgeMaxUti[2])
    std_ospf = float(e17.prev_std)

    steps = []
    sin_accion_sp = 0
    while True:
        n_cand = len(e17.src_dst_k_middlepoints[str(src) + ':' + str(dst)])
        if policy == 'ospf':
            a = action_keep_sp(e17, src, dst)
            if a is None:  # should not happen; just in case
                a = 0
                sin_accion_sp += 1
        else:
            a = rng.randrange(n_cand)

        _, r16, t16, _, info16 = e16.step(a, demand, src, dst)
        _, r17, t17, _, info17 = e17.step(a, demand, src, dst)
        assert t16 == t17, "The environments finished at different steps"

        steps.append({'r16': float(r16), 'r17': float(r17), 'sesgo': float(r16) - float(r17)})
        demand, src, dst = info17['bw'], info17['src'], info17['dst']
        if t17:
            mlu_final = float(info17['edgeMaxUti'][2])
            std_final = float(info17['std_norm'])
            break

    ret16 = sum(s['r16'] for s in steps)
    ret17 = sum(s['r17'] for s in steps)
    teorico = 10 * (mlu_ospf - mlu_final) + 10 * (std_ospf - std_final)
    return steps, {
        'tm_id': tm_id, 'pasos': len(steps),
        'retorno_v16': ret16, 'retorno_v17': ret17,
        'sesgo_episodio': ret16 - ret17,
        'retorno_v17_teorico': teorico,
        'error_telescopico': ret17 - teorico,
        'mlu_ospf': mlu_ospf, 'mlu_final': mlu_final,
        'sin_accion_sp': sin_accion_sp,
    }


def main():
    p = argparse.ArgumentParser(description="Quantification of the bias of the reward (v16 vs v17)")
    p.add_argument('-g', required=True, help='Topology (e.g. BtAsiaPac)')
    p.add_argument('-f', required=True, help='Dataset folder (TRAIN or EVALUATE)')
    p.add_argument('-n', type=int, default=20, help='Number of TMs')
    p.add_argument('-t0', type=int, default=0, help='First TM')
    p.add_argument('-policy', nargs='+', default=['ospf', 'aleatoria'], choices=['ospf', 'aleatoria'])
    p.add_argument('-out', default=config.RESULTS_DIR + '/reward_bias', help='Output folder')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    e16 = make_env('GraphEnv-v16', a.f, a.g)
    e17 = make_env('GraphEnv-v17', a.f, a.g)
    print(f"=== Reward bias | {a.g} | Nodes: {e17.numNodes} | Links: {e17.numEdges} ===")

    resumen = []
    for pol in a.policy:
        rng = random.Random(SEED)
        ep_rows, step_rows = [], []
        for tm in range(a.t0, a.t0 + a.n):
            steps, ep = run_episode(e16, e17, tm, pol, rng)
            ep['politica'] = pol
            ep_rows.append(ep)
            for k, s in enumerate(steps):
                step_rows.append({'politica': pol, 'tm_id': tm, 'paso': k, **s})
        df_ep = pd.DataFrame(ep_rows)
        df_st = pd.DataFrame(step_rows)
        df_ep.to_csv(os.path.join(a.out, f"{a.g}_{pol}_episodios.csv"), index=False)
        df_st.to_csv(os.path.join(a.out, f"{a.g}_{pol}_pasos.csv"), index=False)

        fila = {
            'topologia': a.g, 'politica': pol, 'TMs': a.n,
            'pasos_por_episodio': df_ep['pasos'].mean(),
            'sesgo_medio_paso': df_st['sesgo'].mean(),
            'sesgo_std_paso': df_st['sesgo'].std(),
            'pct_pasos_sesgo_negativo': 100 * (df_st['sesgo'] < -1e-9).mean(),
            'sesgo_medio_episodio': df_ep['sesgo_episodio'].mean(),
            'retorno_medio_v16': df_ep['retorno_v16'].mean(),
            'retorno_medio_v17': df_ep['retorno_v17'].mean(),
            'max_error_telescopico_v17': df_ep['error_telescopico'].abs().max(),
        }
        resumen.append(fila)
        print(f"\n--- Policy: {pol} ---")
        for k, v in fila.items():
            if k not in ('topologia', 'politica'):
                print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")
        if pol == 'ospf' and df_ep['sin_accion_sp'].sum() > 0:
            print(f"  [WARNING] {int(df_ep['sin_accion_sp'].sum())} steps without a candidate equivalent to the shortest path.")

    pd.DataFrame(resumen).to_csv(os.path.join(a.out, f"{a.g}_resumen.csv"), index=False)
    print(f"\nResults saved in {a.out}/")


if __name__ == '__main__':
    main()
