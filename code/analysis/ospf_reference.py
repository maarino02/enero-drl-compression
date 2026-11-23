# -*- coding: utf-8 -*-
"""
analysis/ospf_reference.py - OSPF reference on the validation TMs of the v2 training
====================================================================================
Computes, for the same 60 TMs used by the validation of the training
(tm 0-19 of EVALUATE of BtAsiaPac, Garr199905 and Goodnet), the MLU and the normalized
standard deviation with the initial OSPF routing (without changing any demand).

As in Env17 the return of an episode is
    REW = 10*(MLU_OSPF - MLU_final) + 10*(std_OSPF - std_final),
these values give the line "REW = 0" (agent equal to OSPF) and the reference
for the MLU curves of the training. It takes about 1 minute and uses 1 core.

Usage (from code/):  python -m analysis.ospf_reference
"""
import argparse
import numpy as np
import gymnasium as gym
import gym_graph  # noqa: F401

import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-base', default=config.MAIN_TOPOLOGIES_DIR)
    p.add_argument('-n', type=int, default=20)
    a = p.parse_args()

    todas_mlu, todas_std = [], []
    for top in ['BtAsiaPac', 'Garr199905', 'Goodnet']:
        env = gym.make('GraphEnv-v17').unwrapped
        env.generate_environment(f"{a.base}/NEW_{top}/EVALUATE", top, 100, 100, 0.15)
        env.top_K_critical_demands = True
        mlu, std = [], []
        for tm in range(a.n):
            env.reset(options={'tm_id': tm})
            mlu.append(float(env.edgeMaxUti[2]))
            std.append(float(env.prev_std))
        todas_mlu += mlu
        todas_std += std
        print(f"{top:<11} MLU_OSPF mean={np.mean(mlu):.4f} max={np.max(mlu):.4f} | std_OSPF mean={np.mean(std):.4f}")

    print(f"\nTOTAL (60 TMs)  MLU_OSPF mean={np.mean(todas_mlu):.4f}  max={np.max(todas_mlu):.4f}  "
          f"std_OSPF mean={np.mean(todas_std):.4f}")


if __name__ == '__main__':
    main()
