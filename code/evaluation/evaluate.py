# -*- coding: utf-8 -*-
"""
evaluation/evaluate.py - Corrected and unified evaluation of the models of the thesis
=====================================================================================
Replaces the first evaluation scripts for the compression results. Fixes with respect
to the previous versions:

  1. TFLite is evaluated with the K candidate graphs (before, a single graph without
     a marked middlepoint was passed and the agent always chose action 0).
  2. All the TF models run in graph mode (tf.function with a fixed signature), like
     TFLite, so that the latency is comparable. The first calls (tracing) are
     discarded and ONLY the inference of the network is measured.
  3. The complete optimization time per TM (all the critical demands) is also
     measured and, optionally, the Hill Climbing refinement (DRL+HC).
  4. Hill Climbing computes the MLU directly from the environment, without depending
     on the sign of the values returned by Env15 after the migration to Gymnasium.
  5. The size is measured on the WEIGHTS of the actor (without optimizer state):
     parameters, non-zero, FP32 bytes and gzip-compressed bytes (standard way of
     showing the gain of pruning). For TFLite, the size of the file.
  6. The agreement of decisions (argmax) of each model with the original is measured.
  7. LP lower bound (multicommodity flow aggregated by source) computed from the
     complete traffic matrix of the environment.

Example (from code/):

  python -m evaluation.evaluate -g EliBackbone ^
     -f <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_EliBackbone/EVALUATE ^
     -orig models/base/ckpt_ACT-282 ^
     -model Podado50 pruned models/pruned_50/ckpt_PRUNED_s50_BEST-1 ^
     -model Estudiante student models/student/ckpt_STUDENT_V2_BEST-1 ^
     -model INT8_dinamico tflite models/tflite/actor_dinamico.tflite ^
     -n 50 -lp -ls

(In the VS Code terminal with PowerShell use ` instead of ^ to split lines,
 or write everything in a single line.)
"""
import os
import sys
import time
import json
import gzip
import pickle
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import gymnasium as gym
import gym_graph  # noqa: F401  (registers the environments)
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from agent import actor
from evaluation.local_search import HILL_CLIMBING
import config

ENV_DRL = 'GraphEnv-v16'
ENV_HC = 'GraphEnv-v15'
PCT_DEMANDS = 0.15
EPISODE_LENGTH = 100
NUM_ACTIONS = 100
SEED = 9
WARMUP_CALLS = 10

HPARAMS_TEACHER = {'l2': 0.0001, 'dropout_rate': 0.0, 'link_state_dim': 32, 'readout_units': 32,
                   'learning_rate': 0.0002, 'T': 8, 'num_readout_layers': 4, 'activation_fn': 'selu'}
HPARAMS_STUDENT = {'l2': 0.0001, 'dropout_rate': 0.0, 'link_state_dim': 16, 'readout_units': 16,
                   'learning_rate': 0.001, 'T': 4, 'num_readout_layers': 2, 'activation_fn': 'selu'}

np.random.seed(SEED)
tf.random.set_seed(1)
HIDDEN_INIT = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
KERNEL_INIT = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)


# ----------------------------------------------------------------------------
# Input tensor (K candidate graphs, one per middlepoint)
# ----------------------------------------------------------------------------
def old_cummax(alist, extractor):
    maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
    cummaxes = [tf.zeros_like(maxes[0])]
    for i in range(len(maxes) - 1):
        cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes


def get_graph_features(env, dim):
    n = env.numEdges
    util = np.divide(env.edge_state[:, 0], env.edge_state[:, 1])
    util = tf.reshape(tf.convert_to_tensor(util[0:n], dtype=tf.float32), [n, 1])
    cap = tf.reshape(tf.cast(env.link_capacity_feature[0:n], tf.float32), [n, 1])
    bw = tf.reshape(tf.convert_to_tensor(env.edge_state[0:n, 2], dtype=tf.float32), [n, 1])
    hidden = tf.concat([util, cap, bw], axis=1)
    link_state = tf.pad(hidden, tf.constant([[0, 0], [0, dim - 3]]), mode="CONSTANT")
    return {'link_state': link_state,
            'first': env.first[0:env.firstTrueSize],
            'second': env.second[0:env.firstTrueSize],
            'num_edges': n}


def build_tensor(env, src, dst, dim):
    feats = []
    for mp in env.src_dst_k_middlepoints[str(src) + ':' + str(dst)]:
        env.mark_action_sp(src, mp, src, dst)
        if mp != dst:
            env.mark_action_sp(mp, dst, src, dst)
        feats.append(get_graph_features(env, dim))
        env.edge_state[:, 2] = 0
    graph_ids = [tf.fill([tf.shape(v['link_state'])[0]], it) for it, v in enumerate(feats)]
    f_off = old_cummax(feats, lambda v: v['first'])
    s_off = old_cummax(feats, lambda v: v['second'])
    return {
        'link_state': tf.cast(tf.concat([v['link_state'] for v in feats], axis=0), tf.float32),
        'graph_id': tf.cast(tf.concat(graph_ids, axis=0), tf.int32),
        'first': tf.cast(tf.concat([v['first'] + m for v, m in zip(feats, f_off)], axis=0), tf.int32),
        'second': tf.cast(tf.concat([v['second'] + m for v, m in zip(feats, s_off)], axis=0), tf.int32),
        'num_edges': tf.cast(tf.math.add_n([v['num_edges'] for v in feats]), tf.int32),
    }


# ----------------------------------------------------------------------------
# Model wrappers
# ----------------------------------------------------------------------------
class TFActor:
    """Keras actor (original, pruned or student) run in graph mode."""

    def __init__(self, name, ckpt_path, hparams, eager=False):
        self.name = name
        self.kind = 'tf'
        self.path = ckpt_path
        self.dim = hparams['link_state_dim']
        self.model = actor.myModel(hparams, HIDDEN_INIT, KERNEL_INIT)
        self.model.build()
        status = tf.train.Checkpoint(model=self.model).restore(ckpt_path)
        try:
            status.assert_existing_objects_matched()
            self.restore_ok = True
        except AssertionError as e:
            print(f"\n[WARNING] {name}: the checkpoint does not contain all the weights of the model.\n{e}\n")
            self.restore_ok = False
        status.expect_partial()

        m = self.model
        spec = [tf.TensorSpec([None, self.dim], tf.float32), tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([], tf.int32)]

        @tf.function(input_signature=spec)
        def _fwd(ls, gid, f, s, n):
            return m(ls, gid, f, s, n, training=False)

        self._fwd = (lambda ls, gid, f, s, n: m(ls, gid, f, s, n, training=False)) if eager else _fwd

    def predict(self, t):
        r = self._fwd(t['link_state'], t['graph_id'], t['first'], t['second'], t['num_edges'])
        return r.numpy().reshape(-1)

    def stats(self):
        arrays = [w.numpy().astype(np.float32) for w in self.model.weights]
        n = int(sum(a.size for a in arrays))
        nz = int(sum(np.count_nonzero(a) for a in arrays))
        raw = b''.join(a.tobytes() for a in arrays)
        per_layer = {w.name: round(1.0 - np.count_nonzero(w.numpy()) / max(w.numpy().size, 1), 4)
                     for w in self.model.weights if len(w.shape) == 2}
        disk = 0
        base_dir, base = os.path.dirname(self.path), os.path.basename(self.path)
        for fn in os.listdir(base_dir or '.'):
            if fn.startswith(base + '.'):
                disk += os.path.getsize(os.path.join(base_dir, fn))
        return {'modelo': self.name, 'tipo': 'tf', 'restauracion_ok': self.restore_ok,
                'parametros': n, 'no_nulos': nz, 'esparsidad': round(1 - nz / n, 4),
                'bytes_fp32': n * 4, 'bytes_gzip': len(gzip.compress(raw, 9)),
                'bytes_checkpoint_disco': disk, 'esparsidad_por_matriz': per_layer}


class TFLiteActor:
    """TFLite actor evaluated with the complete batch of K graphs."""

    ORDER = ['link_state', 'graph', 'first', 'second', 'num_edges']

    def __init__(self, name, path, threads=None):
        self.name = name
        self.kind = 'tflite'
        self.path = path
        self.dim = 32
        self.it = tf.lite.Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        det = self.it.get_input_details()
        print(f"  [{name}] TFLite inputs: {[d['name'] for d in det]}")
        found = {}
        for d in det:
            nm = d['name'].lower()
            if 'link_state' in nm:
                found['link_state'] = d
            elif 'graph' in nm:
                found['graph'] = d
            elif 'first' in nm:
                found['first'] = d
            elif 'second' in nm:
                found['second'] = d
            elif 'edges' in nm:
                found['num_edges'] = d
        if len(found) != 5:
            print("  [WARNING] The inputs could not be identified by name; the index order is used.")
            found = dict(zip(self.ORDER, sorted(det, key=lambda d: d['index'])))
        self.inp = found
        self.dim = int(found['link_state']['shape_signature'][-1]) if found['link_state']['shape_signature'][-1] > 0 else 32
        self.out = self.it.get_output_details()[0]
        self._shapes = None

    def predict(self, t):
        ls = t['link_state'].numpy().astype(np.float32)
        gid = t['graph_id'].numpy().astype(np.int32)
        f = t['first'].numpy().astype(np.int32)
        s = t['second'].numpy().astype(np.int32)
        n = np.array(int(t['num_edges']), dtype=np.int32)
        shapes = (ls.shape, gid.shape, f.shape, s.shape)
        self.last_resize = 0.0
        if shapes != self._shapes:
            t_r = time.perf_counter()
            self.it.resize_tensor_input(self.inp['link_state']['index'], ls.shape)
            self.it.resize_tensor_input(self.inp['graph']['index'], gid.shape)
            self.it.resize_tensor_input(self.inp['first']['index'], f.shape)
            self.it.resize_tensor_input(self.inp['second']['index'], s.shape)
            self.it.allocate_tensors()
            self._shapes = shapes
            self.last_resize = time.perf_counter() - t_r   # cost of resizing and allocating memory
        self.it.set_tensor(self.inp['link_state']['index'], ls)
        self.it.set_tensor(self.inp['graph']['index'], gid)
        self.it.set_tensor(self.inp['first']['index'], f)
        self.it.set_tensor(self.inp['second']['index'], s)
        self.it.set_tensor(self.inp['num_edges']['index'], n)
        self.it.invoke()
        return self.it.get_tensor(self.out['index']).reshape(-1)

    def stats(self):
        raw = open(self.path, 'rb').read()
        return {'modelo': self.name, 'tipo': 'tflite', 'restauracion_ok': True,
                'parametros': None, 'no_nulos': None, 'esparsidad': None,
                'bytes_fp32': None, 'bytes_gzip': len(gzip.compress(raw, 9)),
                'bytes_checkpoint_disco': len(raw), 'esparsidad_por_matriz': {}}


# ----------------------------------------------------------------------------
# Network utilities
# ----------------------------------------------------------------------------
def mlu_from_env(env):
    m = 0.0
    for i in env.graph:
        for j in env.graph[i]:
            m = max(m, env.graph[i][j][0]['utilization'] / env.links_bw[i][j])
    return float(m)


def warmup(env16, agent, tm_id):
    _, info = env16.reset(options={'tm_id': tm_id})
    t = build_tensor(env16, info['src'], info['dst'], agent.dim)
    for _ in range(WARMUP_CALLS):
        agent.predict(t)


def run_drl(env16, tm_id, agent, ref=None):
    """DRL stage: one pass over the critical demands choosing the argmax."""
    _, info = env16.reset(options={'tm_id': tm_id})
    src, dst = info['src'], info['dst']
    demand = info.get('demand', info.get('bw'))
    ospf = float(env16.edgeMaxUti[2])
    best, best_routing = ospf, {}
    lat, lat_inv, agree, t_ref, steps, mlu = [], [], [], 0.0, 0, ospf
    t0 = time.perf_counter()
    while True:
        tensor = build_tensor(env16, src, dst, agent.dim)
        ti = time.perf_counter()
        logits = agent.predict(tensor)
        lat.append(time.perf_counter() - ti)
        lat_inv.append(lat[-1] - getattr(agent, 'last_resize', 0.0))   # without resizing (TFLite only)
        a = int(np.argmax(logits))
        if ref is not None and ref is not agent:
            tr = time.perf_counter()
            tref = tensor if ref.dim == agent.dim else build_tensor(env16, src, dst, ref.dim)
            agree.append(a == int(np.argmax(ref.predict(tref))))
            t_ref += time.perf_counter() - tr
        _, _, term, trunc, info = env16.step(a, demand, src, dst)
        steps += 1
        mlu = float(info['edgeMaxUti'][2])
        if mlu < best:
            best = mlu
            best_routing = dict(env16.sp_middlepoints_step)
        demand, src, dst = info['bw'], info['src'], info['dst']
        if term or trunc:
            break
    t_drl = time.perf_counter() - t0 - t_ref
    return {'ospf': ospf, 'drl_mejor': best, 'drl_final': mlu, 't_drl_s': t_drl, 'pasos': steps,
            'acuerdo': float(np.mean(agree)) if agree else np.nan,
            'lat_inv_mediana_ms': float(np.median(lat_inv)) * 1000}, best_routing, lat


def run_hc(env15, tm_id, routing, max_iters=10000):
    """Hill Climbing over the critical demands starting from 'routing' (None = OSPF)."""
    env15.reset_DRL_hill_sp(tm_id, dict(routing) if routing is not None else None, [])
    hc = HILL_CLIMBING(env15)
    cur = mlu_from_env(env15)
    it = 0
    t0 = time.perf_counter()
    while it < max_iters:
        next_val, st = hc.explore_neighbourhood_DRL_sp(env15)
        if st is None:
            break
        cand = -float(np.squeeze(next_val))
        if cand >= cur - 1e-4:
            break
        a, s, d = st
        key = str(s) + ':' + str(d)
        if key in env15.sp_middlepoints:
            mp = env15.sp_middlepoints[key]
            env15.decrease_links_utilization_sp(s, mp, s, d)
            env15.decrease_links_utilization_sp(mp, d, s, d)
            del env15.sp_middlepoints[key]
        else:
            env15.decrease_links_utilization_sp(s, d, s, d)
        env15.step_hill_sp(a, s, d)
        cur = mlu_from_env(env15)
        it += 1
    return cur, time.perf_counter() - t0, it


def solve_lp(env16, tm_id, time_limit=300):
    """Lower bound: LP of multicommodity flow aggregated by source (splittable flow)."""
    import pulp
    env16.reset(options={'tm_id': tm_id})
    TM = np.array(env16.TM, dtype=float)
    N = TM.shape[0]
    edges = [(u, v) for u in env16.graph for v in env16.graph[u]]
    prob = pulp.LpProblem("MLU", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0)
    prob += U
    f = {(s, u, v): pulp.LpVariable(f"f_{s}_{u}_{v}", lowBound=0) for s in range(N) for (u, v) in edges}
    out_e = {n: [(u, v) for (u, v) in edges if u == n] for n in range(N)}
    in_e = {n: [(u, v) for (u, v) in edges if v == n] for n in range(N)}
    for s in range(N):
        total_s = sum(TM[s, d] for d in range(N) if d != s)
        for n in range(N):
            bal = pulp.lpSum(f[(s, u, v)] for (u, v) in out_e[n]) - pulp.lpSum(f[(s, u, v)] for (u, v) in in_e[n])
            prob += (bal == (total_s if n == s else -TM[s, n]))
    for (u, v) in edges:
        prob += pulp.lpSum(f[(s, u, v)] for s in range(N)) <= float(env16.links_bw[u][v]) * U
    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    t = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    return (float(pulp.value(U)) if status == 'Optimal' else np.nan), t, status


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Corrected evaluation of compressed models (TFG)")
    p.add_argument('-g', required=True, help='Topology (e.g. EliBackbone)')
    p.add_argument('-f', required=True, help='EVALUATION dataset folder of that topology')
    p.add_argument('-orig', default=config.MODELS_DIR + '/base/ckpt_ACT-282', help='Checkpoint of the original model')
    p.add_argument('-model', action='append', nargs=3, metavar=('NAME', 'TYPE', 'PATH'), default=[],
                   help="Additional model. TYPE: pruned | student | original | tflite")
    p.add_argument('-n', type=int, default=50, help='Number of TMs')
    p.add_argument('-t0', type=int, default=0, help='First TM')
    p.add_argument('-no_hc', action='store_true', help='Do not run the Hill Climbing refinement')
    p.add_argument('-ls', action='store_true', help='Also compute Hill Climbing from OSPF (LS only)')
    p.add_argument('-lp', action='store_true', help='Compute the LP lower bound (requires pulp)')
    p.add_argument('-eager', action='store_true', help='Run the TF models in eager mode (comparison)')
    p.add_argument('-threads', type=int, default=None, help='Threads of the TFLite interpreter')
    p.add_argument('-tf_threads', type=int, default=None,
                   help='TensorFlow threads (intra and inter-op). Default: the ones chosen by TF (all the cores)')
    p.add_argument('-sin_original', action='store_true',
                   help='Do not evaluate the original model (only the -model ones). The agreement of decisions is not computed')
    p.add_argument('-out', default=config.RESULTS_DIR + '/main', help='Output folder')
    a = p.parse_args()

    if a.tf_threads is not None:
        # Must be done before running the first TensorFlow operation
        tf.config.threading.set_intra_op_parallelism_threads(a.tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(a.tf_threads)
        print(f"  TensorFlow threads: {tf.config.threading.get_intra_op_parallelism_threads()} (intra) / "
              f"{tf.config.threading.get_inter_op_parallelism_threads()} (inter)")

    os.makedirs(a.out, exist_ok=True)
    tag = a.g + ('_eager' if a.eager else '')

    print(f"=== Evaluation v2 | {a.g} | TMs {a.t0}..{a.t0 + a.n - 1} ===")
    env16 = gym.make(ENV_DRL).unwrapped
    env16.generate_environment(a.f, a.g, EPISODE_LENGTH, NUM_ACTIONS, PCT_DEMANDS)
    env16.top_K_critical_demands = True
    env15 = None
    if not a.no_hc or a.ls:
        env15 = gym.make(ENV_HC).unwrapped
        env15.generate_environment(a.f, a.g, EPISODE_LENGTH, NUM_ACTIONS, PCT_DEMANDS)
    print(f"  Nodes: {env16.numNodes} | Links: {env16.numEdges}")

    agents = [] if a.sin_original else [TFActor('Original', a.orig, HPARAMS_TEACHER, eager=a.eager)]
    for name, kind, path in a.model:
        if kind == 'tflite':
            agents.append(TFLiteActor(name, path, a.threads))
        elif kind == 'student':
            agents.append(TFActor(name, path, HPARAMS_STUDENT, eager=a.eager))
        else:
            agents.append(TFActor(name, path, HPARAMS_TEACHER, eager=a.eager))
    ref = None if a.sin_original else agents[0]
    with open(os.path.join(a.out, f"{tag}_config.json"), 'w', encoding='utf-8') as fh:
        json.dump({'argumentos': vars(a), 'tf_intra': tf.config.threading.get_intra_op_parallelism_threads(),
                   'tf_inter': tf.config.threading.get_inter_op_parallelism_threads(),
                   'fecha': time.strftime('%Y-%m-%d %H:%M:%S')}, fh, indent=2, ensure_ascii=False)

    stats = [ag.stats() for ag in agents]
    with open(os.path.join(a.out, f"{tag}_modelos.json"), 'w', encoding='utf-8') as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    for s in stats:
        print(f"  {s['modelo']:<16} params={s['parametros']} non_zero={s['no_nulos']} "
              f"gzip={s['bytes_gzip']} B restore_ok={s['restauracion_ok']}")

    for ag in agents:
        warmup(env16, ag, a.t0)

    rows, latencies = [], {ag.name: [] for ag in agents}
    csv_det = os.path.join(a.out, f"{tag}_detalle.csv")
    for tm in range(a.t0, a.t0 + a.n):
        lp_val, lp_t, lp_status = (np.nan, np.nan, 'no_calculado')
        if a.lp:
            try:
                lp_val, lp_t, lp_status = solve_lp(env16, tm)
            except Exception as e:  # noqa: BLE001
                lp_status = f'error: {e}'
        ls_val, ls_t, ls_it = (np.nan, np.nan, 0)
        if a.ls:
            ls_val, ls_t, ls_it = run_hc(env15, tm, None)
        for ag in agents:
            res, routing, lat = run_drl(env16, tm, ag, ref)
            latencies[ag.name].extend(lat)
            hc_val, hc_t, hc_it = (np.nan, 0.0, 0)
            if not a.no_hc:
                hc_val, hc_t, hc_it = run_hc(env15, tm, routing)
            lat_ms = np.array(lat) * 1000
            rows.append({'topologia': a.g, 'tm': tm, 'modelo': ag.name, **res,
                         'drl_hc': hc_val, 't_hc_s': hc_t, 'iter_hc': hc_it,
                         't_total_s': res['t_drl_s'] + hc_t,
                         'lat_mediana_ms': float(np.median(lat_ms)), 'lat_p95_ms': float(np.percentile(lat_ms, 95)),
                         'ls_solo': ls_val, 't_ls_s': ls_t, 'iter_ls': ls_it,
                         'lp': lp_val, 't_lp_s': lp_t, 'lp_estado': lp_status})
            print(f"  TM {tm:>2} | {ag.name:<16} OSPF={res['ospf']:.4f} DRL={res['drl_mejor']:.4f} "
                  f"DRL+HC={hc_val:.4f} LP={lp_val:.4f} agreement={res['acuerdo']:.2f} "
                  f"lat={np.median(lat_ms):.2f} ms t_opt={res['t_drl_s'] + hc_t:.2f} s", flush=True)
        pd.DataFrame(rows).to_csv(csv_det, index=False, sep=';', decimal=',')  # incremental save

    df = pd.DataFrame(rows)
    num = ['ospf', 'drl_mejor', 'drl_final', 'drl_hc', 'lp', 'ls_solo', 'acuerdo', 't_drl_s', 't_hc_s',
           't_total_s', 't_ls_s', 't_lp_s', 'pasos', 'iter_hc', 'lat_inv_mediana_ms']
    res = df.groupby('modelo', sort=False)[num].mean().reset_index()
    res['mejora_drl_vs_ospf_%'] = (res['ospf'] - res['drl_mejor']) / res['ospf'] * 100
    res['mejora_drlhc_vs_ospf_%'] = (res['ospf'] - res['drl_hc']) / res['ospf'] * 100
    res['brecha_drl_vs_lp_%'] = (res['drl_mejor'] - res['lp']) / res['lp'] * 100
    res['brecha_drlhc_vs_lp_%'] = (res['drl_hc'] - res['lp']) / res['lp'] * 100
    res['lat_mediana_ms'] = [float(np.median(np.array(latencies[m]) * 1000)) for m in res['modelo']]
    res['lat_p95_ms'] = [float(np.percentile(np.array(latencies[m]) * 1000, 95)) for m in res['modelo']]
    res.insert(0, 'topologia', a.g)
    res.to_csv(os.path.join(a.out, f"{tag}_resumen.csv"), index=False, sep=';', decimal=',')
    with open(os.path.join(a.out, f"{tag}_latencias.pckl"), 'wb') as fh:
        pickle.dump(latencies, fh)

    print("\n=== SUMMARY ===")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(res.round(4).to_string(index=False))
    print(f"\nFiles in {a.out}/ ({tag}_detalle.csv, {tag}_resumen.csv, {tag}_modelos.json, {tag}_latencias.pckl)")


if __name__ == '__main__':
    main()
