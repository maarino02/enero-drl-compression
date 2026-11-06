# -*- coding: utf-8 -*-
"""
calcular_referencia_ospf.py - Referencia OSPF en las TMs de validacion del entrenamiento v2
=================================================================================
Calcula, para las mismas 60 TMs que usa la validacion del entrenamiento
(tm 0-19 de EVALUATE de BtAsiaPac, Garr199905 y Goodnet), la MLU y la desviacion
estandar normalizada con el enrutamiento OSPF inicial (sin tocar ninguna demanda).

Como en Env17 el retorno de un episodio es
    REW = 10*(MLU_OSPF - MLU_final) + 10*(std_OSPF - std_final),
estos valores dan la linea "REW = 0" (agente igual que OSPF) y la referencia
para las curvas de MLU del entrenamiento. Tarda aprox. 1 minuto y usa 1 nucleo.

Uso (desde codigo/):  python calcular_referencia_ospf.py
"""
import argparse
import numpy as np
import gymnasium as gym
import gym_graph  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument('-base', default='../datasets/principales')
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
    print(f"{top:<11} MLU_OSPF media={np.mean(mlu):.4f} max={np.max(mlu):.4f} | std_OSPF media={np.mean(std):.4f}")

print(f"\nTOTAL (60 TMs)  MLU_OSPF media={np.mean(todas_mlu):.4f}  max={np.max(todas_mlu):.4f}  "
      f"std_OSPF media={np.mean(todas_std):.4f}")
