# -*- coding: utf-8 -*-
"""
compression/pruning_sweep.py - Pruning sweep without retraining (figure fig_poda_barrido, Chapter 7)
====================================================================================================
For each pruning level, the base model is pruned with the same masks as compression/prune.py
(magnitude pruning per matrix, including the GRU) and, WITHOUT fine-tuning, the MLU improvement
over OSPF of the DRL stage is measured on the validation TMs (EVALUATE) of the three training
topologies.

In the thesis this sweep was obtained by running compression/prune.py with -episodes 0 for each
level; its values were only printed to the console. This script repeats exactly that procedure
(same functions of compression/prune.py) for all the levels and stores the values in a CSV.
The figure is drawn by figures/pruning_sweep_figure.py.

Output: results/pruning_sweep/barrido_poda.csv
        models/pruning_sweep/ckpt_PRUNED_s<level>_FINAL-1.*   (pruned actors, as with -episodes 0)

Usage (from code/):
  python -m compression.pruning_sweep
"""
import os
import argparse

import numpy as np
import pandas as pd
import gymnasium as gym
import tensorflow as tf

from compression import prune as ft
import config

TOPOS = ['BtAsiaPac', 'Goodnet', 'Garr199905']
LEVELS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]


def main():
    p = argparse.ArgumentParser(description="Pruning sweep without retraining")
    p.add_argument('-teacher_ckpt', default=config.MODELS_DIR + '/base/ckpt_ACT-282')
    p.add_argument('-levels', nargs='+', type=float, default=LEVELS, help='Sparsity levels (per matrix)')
    p.add_argument('-eval_tms', type=int, default=20, help='Validation TMs per topology')
    p.add_argument('-csv', default=config.RESULTS_DIR + '/pruning_sweep/barrido_poda.csv')
    p.add_argument('-models_out', default=config.MODELS_DIR + '/pruning_sweep',
                   help='Folder for the pruned checkpoints (empty string: do not save them)')
    a = p.parse_args()

    envs = {}
    for t in TOPOS:
        env = gym.make(ft.ENV_NAME).unwrapped
        env.generate_environment(f"{config.MAIN_TOPOLOGIES_DIR}/NEW_{t}/EVALUATE", t,
                                 ft.EPISODE_LENGTH, ft.NUM_ACTIONS, ft.PCT_DEMANDS)
        env.top_K_critical_demands = True
        envs[t] = env

    rows = []
    for s in a.levels:
        tag = f"s{int(round(s * 100))}"
        agent = ft.PrunedAgent(a.teacher_ckpt, a.teacher_ckpt.replace('ckpt_ACT', 'ckpt_CRT'), s)
        r = ft.evaluate(envs, agent, list(range(a.eval_tms)))
        zeros, total = ft.sparsity_report(agent.actor)
        row = {'poda_por_matriz_%': 100 * s, 'parametros_a_cero_%': 100 * zeros / total,
               'mejora_media_%': float(np.mean([v[1] for v in r.values()]))}
        for t in TOPOS:
            row[f'mejora_{t}_%'] = r[t][1]
            row[f'mlu_{t}'] = r[t][0]
        rows.append(row)
        print(f"  {tag}: {row['parametros_a_cero_%']:.1f} % of the actor at zero | "
              f"mean improvement {row['mejora_media_%']:+.2f}% vs OSPF", flush=True)
        if a.models_out:
            os.makedirs(a.models_out, exist_ok=True)
            tf.train.Checkpoint(model=agent.actor).write(os.path.join(a.models_out, f"ckpt_PRUNED_{tag}_FINAL-1"))
        os.makedirs(os.path.dirname(a.csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(a.csv, index=False, sep=';', decimal=',')  # incremental save
    print(f"\nResults in {a.csv}")


if __name__ == '__main__':
    main()
