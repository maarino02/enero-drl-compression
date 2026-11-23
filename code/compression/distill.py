# -*- coding: utf-8 -*-
"""
compression/distill.py - Knowledge Distillation of the ENERO actor (corrected version)
======================================================================================
Differences with the first version of the distillation script:

  [D1] v2 teacher (Env17) and a SINGLE general student: distilled with the same
       training topologies as the teacher (TRAIN) and validated with their
       EVALUATE TMs. The final evaluation topologies are never seen.
  [D2] The best student is chosen by the MLU improvement over OSPF in validation
       (before: by the training loss).
  [D3] Mixed states: with probability 'mix' the environment advances with the action of
       the student and otherwise with the action of the teacher, so that the student also
       learns in the states its own decisions lead to.
  [D4] The agreement of decisions (argmax) between student and teacher is logged.
  [D5] Graph-mode inference, numerically stable log-softmax, history in CSV.

Loss (same as before):  L = alpha * T^2 * KL(p_T^(T) || p_S^(T)) + (1 - alpha) * MSE(q_T, q_S)

The defaults (-alpha 0.7 -temperature 2) are the initial configuration; the student of the
thesis uses -alpha 1 -temperature 4 -seed 9.

Example (from code/, one line):
  python -m compression.distill -teacher_ckpt models/base/ckpt_ACT-282
      -top BtAsiaPac <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_BtAsiaPac -top Garr199905 <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_Garr199905
      -top Goodnet <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_Goodnet -alpha 1 -temperature 4 -out models/student
"""
import os
import time
import math
import random
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import gymnasium as gym
import gym_graph  # noqa: F401
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from agent import actor
import config

SEED = 9
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

ENV_NAME = 'GraphEnv-v17'
PCT_DEMANDS = 0.15
EVAL_INTERVAL = 5
NUM_EVAL_TMS = 10
LR_INIT, LR_MIN = 1e-3, 5e-5

HP_T = {'l2': 0.0001, 'dropout_rate': 0.0, 'link_state_dim': 32, 'readout_units': 32,
        'learning_rate': 2e-4, 'T': 8, 'num_readout_layers': 4, 'activation_fn': 'selu'}
HP_S = {'l2': 0.0001, 'dropout_rate': 0.0, 'link_state_dim': 16, 'readout_units': 16,
        'learning_rate': LR_INIT, 'T': 4, 'num_readout_layers': 2, 'activation_fn': 'selu'}
HID = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
KER = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)


def cosine_lr(ep, n):
    return LR_MIN + 0.5 * (LR_INIT - LR_MIN) * (1 + math.cos(math.pi * ep / max(n - 1, 1)))


def old_cummax(alist, extractor):
    maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
    cummaxes = [tf.zeros_like(maxes[0])]
    for i in range(len(maxes) - 1):
        cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes


def build_tensor(env, src, dst, dim):
    feats = []
    n = env.numEdges
    for mp in env.src_dst_k_middlepoints[str(src) + ':' + str(dst)]:
        env.mark_action_sp(src, mp, src, dst)
        if mp != dst:
            env.mark_action_sp(mp, dst, src, dst)
        util = tf.reshape(tf.convert_to_tensor(np.divide(env.edge_state[:, 0], env.edge_state[:, 1])[0:n],
                                               dtype=tf.float32), [n, 1])
        cap = tf.reshape(tf.cast(env.link_capacity_feature[0:n], tf.float32), [n, 1])
        bw = tf.reshape(tf.convert_to_tensor(env.edge_state[0:n, 2], dtype=tf.float32), [n, 1])
        feats.append({'link_state': tf.pad(tf.concat([util, cap, bw], 1), [[0, 0], [0, dim - 3]]),
                      'first': env.first[0:env.firstTrueSize], 'second': env.second[0:env.firstTrueSize],
                      'num_edges': n})
        env.edge_state[:, 2] = 0
    gids = [tf.fill([tf.shape(v['link_state'])[0]], i) for i, v in enumerate(feats)]
    fo = old_cummax(feats, lambda v: v['first'])
    so = old_cummax(feats, lambda v: v['second'])
    return {'link_state': tf.cast(tf.concat([v['link_state'] for v in feats], 0), tf.float32),
            'graph_id': tf.cast(tf.concat(gids, 0), tf.int32),
            'first': tf.cast(tf.concat([v['first'] + m for v, m in zip(feats, fo)], 0), tf.int32),
            'second': tf.cast(tf.concat([v['second'] + m for v, m in zip(feats, so)], 0), tf.int32),
            'num_edges': tf.cast(tf.math.add_n([v['num_edges'] for v in feats]), tf.int32)}


def make_fwd(model, dim):
    @tf.function(input_signature=[tf.TensorSpec([None, dim], tf.float32), tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([], tf.int32)])
    def fwd(ls, gid, f, s, n):
        return tf.reshape(model(ls, gid, f, s, n, training=False), [-1])
    return fwd


def student_view(t):
    """The input tensor of the student is the one of the teacher cut to 16 columns
    (the first 3 are the actual features; the rest is zero padding)."""
    return {**t, 'link_state': t['link_state'][:, :HP_S['link_state_dim']]}


def evaluate(envs, fwd, dim, tms):
    out = {}
    for name, env in envs.items():
        mlus, ospfs = [], []
        for tm in tms:
            _, info = env.reset(options={'tm_id': tm})
            src, dst, dem = info['src'], info['dst'], info.get('demand', info.get('bw'))
            ospf = float(env.edgeMaxUti[2])
            best = ospf
            while True:
                t = build_tensor(env, src, dst, dim)
                q = fwd(t['link_state'], t['graph_id'], t['first'], t['second'], t['num_edges']).numpy()
                _, _, term, trunc, inf = env.step(int(np.argmax(q)), dem, src, dst)
                best = min(best, float(inf['edgeMaxUti'][2]))
                src, dst, dem = inf['src'], inf['dst'], inf['bw']
                if term or trunc:
                    break
            mlus.append(best)
            ospfs.append(ospf)
        out[name] = (np.mean(mlus), 100 * (np.mean(ospfs) - np.mean(mlus)) / np.mean(ospfs))
    return out


def main():
    p = argparse.ArgumentParser(description="Knowledge Distillation of the ENERO actor (v2)")
    p.add_argument('-teacher_ckpt', required=True)
    p.add_argument('-top', action='append', nargs=2, metavar=('NAME', 'FOLDER'), required=True,
                   help='Topology and folder that contains TRAIN and EVALUATE (repeatable)')
    p.add_argument('-epochs', type=int, default=40)
    p.add_argument('-tms_epoch', type=int, default=4, help='TMs per topology and epoch')
    p.add_argument('-temperature', type=float, default=2.0)
    p.add_argument('-alpha', type=float, default=0.7)
    p.add_argument('-mix', type=float, default=0.5, help='Probability of advancing with the action of the student')
    p.add_argument('-eval_tms', type=int, default=NUM_EVAL_TMS, help='Validation TMs per topology')
    p.add_argument('-seed', type=int, default=SEED,
                   help='Seed (initialization of the student, sampled TMs and mixed trajectories)')
    p.add_argument('-tf_threads', type=int, default=None,
                   help='TensorFlow threads (useful when running several distillations in parallel)')
    p.add_argument('-out', default=config.MODELS_DIR + '/student_configs/alpha0.7_tau2')
    a = p.parse_args()
    if a.tf_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(a.tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(a.tf_threads)
    os.makedirs(a.out, exist_ok=True)
    T = tf.constant(a.temperature, tf.float32)
    # The seed also affects the orthogonal initialization of the student
    tf.random.set_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    hid_s = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=a.seed)
    ker_s = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=a.seed)
    with open(os.path.join(a.out, 'config_kd.txt'), 'w', encoding='utf-8') as fh:
        fh.write(str(vars(a)) + '\n')

    teacher = actor.myModel(HP_T, HID, KER)
    teacher.build()
    st = tf.train.Checkpoint(model=teacher).restore(a.teacher_ckpt)
    st.assert_existing_objects_matched()
    st.expect_partial()
    student = actor.myModel(HP_S, hid_s, ker_s)
    student.build()
    n_t = sum(int(np.prod(v.shape)) for v in teacher.trainable_weights)
    n_s = sum(int(np.prod(v.shape)) for v in student.trainable_weights)
    print("=" * 72)
    print(f"  KD v2 | teacher {n_t} parameters -> student {n_s} ({100 * (1 - n_s / n_t):.1f} % fewer)")
    print(f"  T={a.temperature}  alpha={a.alpha}  mix={a.mix}  epochs={a.epochs}")
    print("=" * 72)

    f_t = make_fwd(teacher, HP_T['link_state_dim'])
    f_s = make_fwd(student, HP_S['link_state_dim'])
    opt = tf.keras.optimizers.Adam(learning_rate=LR_INIT)

    @tf.function(experimental_relax_shapes=True)
    def kd_step(q_t, ls, gid, f, s, n):
        with tf.GradientTape() as tape:
            q_s = tf.reshape(student(ls, gid, f, s, n, training=True), [-1])
            p_t = tf.nn.softmax(q_t / T)
            kl = tf.reduce_sum(p_t * (tf.nn.log_softmax(q_t / T) - tf.nn.log_softmax(q_s / T)))
            mse = tf.reduce_mean(tf.square(q_t - q_s))
            loss = a.alpha * (T ** 2) * kl + (1 - a.alpha) * mse
        g = tape.gradient(loss, student.trainable_variables)
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, student.trainable_variables))
        return loss, q_s

    train_envs, eval_envs = {}, {}
    for name, folder in a.top:
        folder = folder.rstrip('/\\')
        for store, sub in ((train_envs, 'TRAIN'), (eval_envs, 'EVALUATE')):
            env = gym.make(ENV_NAME).unwrapped
            env.generate_environment(f"{folder}/{sub}", name, 100, 100, PCT_DEMANDS)
            env.top_K_critical_demands = True
            store[name] = env

    tms = list(range(a.eval_tms))
    ref = evaluate(eval_envs, f_t, HP_T['link_state_dim'], tms)
    ref_m = float(np.mean([v[1] for v in ref.values()]))
    print("  Teacher (reference): " + " | ".join(f"{k}: {v[1]:+.1f}%" for k, v in ref.items()) + f" | mean {ref_m:+.2f}%")

    best, hist = -1e9, []
    for ep in range(a.epochs):
        t0 = time.time()
        opt.learning_rate.assign(cosine_lr(ep, a.epochs))
        losses, agree = [], []
        for name, env in train_envs.items():
            for tm in random.sample(range(100), a.tms_epoch):
                _, info = env.reset(options={'tm_id': tm})
                src, dst, dem = info['src'], info['dst'], info.get('demand', info.get('bw'))
                while True:
                    t = build_tensor(env, src, dst, HP_T['link_state_dim'])
                    q_t = f_t(t['link_state'], t['graph_id'], t['first'], t['second'], t['num_edges'])
                    s_t = student_view(t)
                    loss, q_s = kd_step(q_t, s_t['link_state'], s_t['graph_id'], s_t['first'], s_t['second'],
                                        s_t['num_edges'])
                    a_t, a_s = int(np.argmax(q_t.numpy())), int(np.argmax(q_s.numpy()))
                    losses.append(float(loss))
                    agree.append(a_t == a_s)
                    act = a_s if random.random() < a.mix else a_t
                    _, _, term, trunc, inf = env.step(act, dem, src, dst)
                    src, dst, dem = inf['src'], inf['dst'], inf['bw']
                    if term or trunc:
                        break
        row = {'epoca': ep + 1, 'lr': float(opt.learning_rate.numpy()), 'perdida': float(np.mean(losses)),
               'acuerdo_entrenamiento_%': 100 * float(np.mean(agree)), 't_s': time.time() - t0}
        msg = ""
        if (ep + 1) % EVAL_INTERVAL == 0 or ep + 1 == a.epochs:
            r = evaluate(eval_envs, f_s, HP_S['link_state_dim'], tms)
            m = float(np.mean([v[1] for v in r.values()]))
            row['mejora_val_%'] = m
            row.update({f'mejora_{k}_%': v[1] for k, v in r.items()})
            msg = f" | val: {m:+.2f}% vs OSPF (teacher {ref_m:+.2f}%)"
            if m > best:
                best = m
                tf.train.Checkpoint(model=student).save(os.path.join(a.out, "ckpt_STUDENT_V2_BEST"))
                msg += "  <- best"
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(a.out, "historial_kd.csv"), index=False, sep=';', decimal=',')
        print(f"  Epoch {ep + 1:>3} lr {row['lr']:.1e} loss {row['perdida']:.4f} "
              f"agreement {row['acuerdo_entrenamiento_%']:.1f}% ({row['t_s'] / 60:.1f} min){msg}", flush=True)

    tf.train.Checkpoint(model=student).save(os.path.join(a.out, "ckpt_STUDENT_V2_FINAL"))
    print("=" * 72)
    print(f"  Teacher: {ref_m:+.2f}% | best student: {best:+.2f}% vs OSPF")
    print(f"  Best: {a.out}/ckpt_STUDENT_V2_BEST-1")
    print("=" * 72)


if __name__ == '__main__':
    main()
