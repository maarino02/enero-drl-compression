# -*- coding: utf-8 -*-
"""
compression/prune.py - Magnitude pruning with manual masks + PPO fine-tuning
============================================================================
Differences with the first version of the pruning script:

  [P1] ALL the weight matrices of the actor are pruned, including the GRU:
         Message (64x32), GRU kernel (32x96), GRU recurrent_kernel (32x96),
         Readout (4 x 32x32) and output layer (32x1)  -> 12,320 of 12,737 parameters.
       With sparsity 0.5, 6,160 weights are removed = 48.4 % of the model.
       Biases and layer normalization are not pruned (417 parameters, 3.3 %).
       tfmot.prune_low_magnitude cannot wrap the GRUCell (it is called inside the
       message passing loop), so the masks are applied manually: they are computed
       once (one-shot pruning, per matrix) and reapplied after each update.
  [P2] Environment GraphEnv-v17 (corrected reward), the same as the v2 teacher model.
  [P3] A SINGLE general pruned model: fine-tuning on the same training topologies as
       the teacher (TRAIN) and validation on their EVALUATE TMs.
       The final evaluation topologies (EliBackbone, ...) are never seen.
  [P4] The critic is initialized with the trained critic of the teacher (ckpt_CRT-N),
       not from scratch.
  [P5] Inference in graph mode (tf.function) and reshape (1, -1) as in the v2 training.
  [P6] The saved checkpoint is a normal actor with zeros: it is loaded directly in
       evaluation/evaluate.py (no "strip").
  [P7] Per-episode history in CSV for the figures of the thesis.

With -episodes 0 the pruned model is only evaluated (no fine-tuning) and only the
_FINAL checkpoint is saved: this is how the pruning sweep without retraining was obtained
(compression/pruning_sweep.py repeats it for all the levels and stores the values in a CSV).

Example (from code/, in a single line in the terminal):
  python -m compression.prune -teacher_ckpt models/base/ckpt_ACT-282
      -top BtAsiaPac <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_BtAsiaPac
      -top Garr199905 <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_Garr199905
      -top Goodnet <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_Goodnet
      -sparsity 0.5 -out models/pruned_50
(the folder of each topology is the one that contains TRAIN and EVALUATE)
"""
import os
import time
import math
import random
import argparse
from collections import deque

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import gymnasium as gym
import gym_graph  # noqa: F401
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from agent import actor
from agent import critic
import config

SEED = 9
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

ENV_NAME = 'GraphEnv-v17'
EPISODE_LENGTH = 100
NUM_ACTIONS = 100
PCT_DEMANDS = 0.15
MINI_BATCH_SIZE = 32
PPO_EPOCHS = 8
BUFF_SIZE = int(np.ceil(PCT_DEMANDS * 300)) * 5   # 225 samples per episode, as in the first version of the pruning script
NUM_TRAIN_TMS_PER_EP = 3
EVAL_INTERVAL = 10
NUM_EVAL_TMS = 10

GAMMA = 0.95          # same as in the training of the v2 teacher
LMBDA = 0.95
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CLIP = 0.2
ENTROPY_BETA_INIT = 0.1
ENTROPY_BETA_FINAL = 0.01
LR_INIT = 2e-4
LR_MIN = 2e-5

HPARAMS = {'l2': 0.0001, 'dropout_rate': 0.0, 'link_state_dim': 32, 'readout_units': 32,
           'learning_rate': LR_INIT, 'T': 8, 'num_readout_layers': 4, 'activation_fn': 'selu'}
DIM = HPARAMS['link_state_dim']
HID_A = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
KER_A = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
HID_C = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
KER_C = tf.keras.initializers.Orthogonal(gain=np.sqrt(1), seed=SEED)


def cosine_lr(ep, n):
    return LR_MIN + 0.5 * (LR_INIT - LR_MIN) * (1 + math.cos(math.pi * ep / max(n - 1, 1)))


def linear_beta(ep, n):
    return ENTROPY_BETA_INIT + (ENTROPY_BETA_FINAL - ENTROPY_BETA_INIT) * ep / max(n - 1, 1)


def old_cummax(alist, extractor):
    maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
    cummaxes = [tf.zeros_like(maxes[0])]
    for i in range(len(maxes) - 1):
        cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes


def get_advantages(values, masks, rewards):
    returns, gae = [], 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + GAMMA * values[i + 1] * masks[i] - values[i]
        gae = delta + GAMMA * LMBDA * masks[i] * gae
        returns.insert(0, gae + values[i])
    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


# ----------------------------------------------------------------------------
# Pruning masks
# ----------------------------------------------------------------------------
def build_masks(model, sparsity):
    """Magnitude pruning, per matrix: sets to zero the fraction 'sparsity' of the weights
    with the lowest absolute value in each matrix (kernel) of the actor."""
    masks = []
    print("\n  Pruned matrices:")
    for v in model.trainable_weights:
        if len(v.shape) != 2 or 'kernel' not in v.name:
            continue  # biases (including the GRU one, 2x96) and LayerNorm are not pruned
        w = v.numpy()
        k = int(round(sparsity * w.size))
        m = np.ones(w.size, dtype=np.float32)
        if k > 0:
            m[np.argsort(np.abs(w).reshape(-1), kind='stable')[:k]] = 0.0
        m = m.reshape(w.shape)
        masks.append((v, tf.constant(m)))
        print(f"    {v.name:<55} {str(tuple(w.shape)):<10} {w.size:>6} weights -> {k:>5} to zero")
    n_mat = sum(int(np.prod(v.shape)) for v, _ in masks)
    n_tot = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    n_zero = sum(int(np.sum(m.numpy() == 0)) for _, m in masks)
    print(f"  Parameters in matrices: {n_mat} of {n_tot} | pruned: {n_zero} "
          f"({100 * n_zero / n_tot:.1f} % of the actor)\n")
    return masks


def apply_masks(masks):
    for v, m in masks:
        v.assign(v * m)


def sparsity_report(model):
    tot = sum(int(np.prod(v.shape)) for v in model.weights)
    zeros = sum(int(np.sum(v.numpy() == 0)) for v in model.weights)
    return zeros, tot


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------
class PrunedAgent:
    def __init__(self, teacher_ckpt, critic_ckpt, sparsity):
        self.memory = deque(maxlen=BUFF_SIZE)
        self.inds = np.arange(BUFF_SIZE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR_INIT, beta_1=0.9, epsilon=1e-05)

        self.actor = actor.myModel(HPARAMS, HID_A, KER_A)
        self.actor.build()
        st = tf.train.Checkpoint(model=self.actor).restore(teacher_ckpt)
        st.assert_existing_objects_matched()
        st.expect_partial()
        print(f"  [OK] Teacher actor loaded: {teacher_ckpt}")

        self.critic = critic.myModel(HPARAMS, HID_C, KER_C)
        self.critic.build()
        try:
            stc = tf.train.Checkpoint(model=self.critic).restore(critic_ckpt)
            stc.assert_existing_objects_matched()
            stc.expect_partial()
            print(f"  [OK] Teacher critic loaded: {critic_ckpt}")
        except Exception as e:  # noqa: BLE001
            print(f"  [WARNING] The critic could not be loaded ({e}). A new critic is used.")

        self.masks = build_masks(self.actor, sparsity)
        apply_masks(self.masks)

        a_mod, c_mod = self.actor, self.critic

        @tf.function(input_signature=[tf.TensorSpec([None, DIM], tf.float32), tf.TensorSpec([None], tf.int32),
                                      tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32),
                                      tf.TensorSpec([], tf.int32)])
        def _afwd(ls, gid, f, s, n):
            return a_mod(ls, gid, f, s, n, training=False)

        @tf.function(input_signature=[tf.TensorSpec([None, DIM], tf.float32), tf.TensorSpec([None], tf.int32),
                                      tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.int32)])
        def _cfwd(ls, f, s, n):
            return c_mod(ls, f, s, num_edges_critic=n, training=False)

        self._afwd, self._cfwd = _afwd, _cfwd

    # --- features ---
    @staticmethod
    def graph_features(env):
        n = env.numEdges
        util = tf.reshape(tf.convert_to_tensor(np.divide(env.edge_state[:, 0], env.edge_state[:, 1])[0:n],
                                               dtype=tf.float32), [n, 1])
        cap = tf.reshape(tf.cast(env.link_capacity_feature[0:n], tf.float32), [n, 1])
        bw = tf.reshape(tf.convert_to_tensor(env.edge_state[0:n, 2], dtype=tf.float32), [n, 1])
        ls = tf.pad(tf.concat([util, cap, bw], axis=1), [[0, 0], [0, DIM - 3]])
        return {'link_state': ls, 'first': env.first[0:env.firstTrueSize],
                'second': env.second[0:env.firstTrueSize], 'num_edges': n}

    @staticmethod
    def critic_features(env):
        n = env.numEdges
        util = tf.reshape(tf.convert_to_tensor(np.divide(env.edge_state[:, 0], env.edge_state[:, 1])[0:n],
                                               dtype=tf.float32), [n, 1])
        cap = tf.reshape(tf.cast(env.link_capacity_feature[0:n], tf.float32), [n, 1])
        ls = tf.pad(tf.concat([util, cap], axis=1), [[0, 0], [0, DIM - 2]])
        return {'link_state_critic': ls, 'first_critic': env.first[0:env.firstTrueSize],
                'second_critic': env.second[0:env.firstTrueSize], 'num_edges_critic': n}

    def value(self, f):
        v = self._cfwd(tf.cast(f['link_state_critic'], tf.float32), tf.cast(f['first_critic'], tf.int32),
                       tf.cast(f['second_critic'], tf.int32), tf.cast(f['num_edges_critic'], tf.int32))
        return v[0].numpy()[0]

    def policy(self, env, src, dst):
        feats = []
        for mp in env.src_dst_k_middlepoints[str(src) + ':' + str(dst)]:
            env.mark_action_sp(src, mp, src, dst)
            if mp != dst:
                env.mark_action_sp(mp, dst, src, dst)
            feats.append(self.graph_features(env))
            env.edge_state[:, 2] = 0
        gids = [tf.fill([tf.shape(v['link_state'])[0]], i) for i, v in enumerate(feats)]
        fo = old_cummax(feats, lambda v: v['first'])
        so = old_cummax(feats, lambda v: v['second'])
        t = {'graph_id': tf.cast(tf.concat(gids, 0), tf.int32),
             'link_state': tf.cast(tf.concat([v['link_state'] for v in feats], 0), tf.float32),
             'first': tf.cast(tf.concat([v['first'] + m for v, m in zip(feats, fo)], 0), tf.int32),
             'second': tf.cast(tf.concat([v['second'] + m for v, m in zip(feats, so)], 0), tf.int32),
             'num_edges': tf.cast(tf.math.add_n([v['num_edges'] for v in feats]), tf.int32)}
        r = self._afwd(t['link_state'], t['graph_id'], t['first'], t['second'], t['num_edges'])
        return tf.nn.softmax(tf.reshape(r, (1, -1))).numpy()[0], t

    # --- PPO ---
    @tf.function(experimental_relax_shapes=True)
    def _critic_step(self, ret, ls, f, s, n):
        value = self.critic(ls, f, s, num_edges_critic=n, training=True)[0]
        return tf.math.squared_difference(tf.stop_gradient(ret), value)

    @tf.function(experimental_relax_shapes=True)
    def _actor_step(self, adv, old_act, old_probs, ls, gid, f, s, n):
        r = self.actor(ls, gid, f, s, n, training=True)
        probs = tf.nn.softmax(tf.reshape(r, (1, -1)))
        p_new = tf.math.reduce_sum(old_act * probs[0])
        ratio = tf.exp(tf.math.log(p_new + 1e-10) - tf.math.log(tf.math.reduce_sum(old_act * old_probs) + 1e-10))
        loss = tf.maximum(-ratio * adv, -tf.clip_by_value(ratio, 1 - CLIP, 1 + CLIP) * adv)
        ent = -tf.math.reduce_sum(tf.math.log(probs + 1e-10) * probs[0])
        return loss, ent

    def _train_step(self, inds, beta):
        a_l, c_l, ents = [], [], []
        with tf.GradientTape() as tape:
            for i in inds:
                sm = self.memory[i]
                la, e = self._actor_step(sm['advantage'], sm['old_act'], sm['old_probs'], sm['link_state'],
                                         sm['graph_id'], sm['first'], sm['second'], sm['num_edges'])
                a_l.append(la)
                ents.append(e)
                c_l.append(self._critic_step(sm['return'], sm['link_state_critic'], sm['first_critic'],
                                             sm['second_critic'], sm['num_edges_critic']))
            c_loss = tf.math.reduce_mean(c_l)
            ent = tf.math.reduce_mean(ents)
            a_loss = tf.math.reduce_mean(a_l) - beta * ent
            total = a_loss + VF_COEF * c_loss
        tv = self.actor.trainable_weights + self.critic.trainable_weights
        grads, _ = tf.clip_by_global_norm(tape.gradient(total, tv), MAX_GRAD_NORM)
        self.optimizer.apply_gradients(zip(grads, tv))
        apply_masks(self.masks)  # [P1] the pruned weights are set back to zero after each step
        return a_loss, c_loss, ent

    def update(self, acts, probs, tensors, cfeats, returns, advs, beta):
        self.memory.clear()
        for p in range(min(len(tensors), BUFF_SIZE)):
            t, c = tensors[p], cfeats[p]
            self.memory.append({**t, **c,
                                'old_act': tf.convert_to_tensor(acts[p], tf.float32),
                                'advantage': tf.convert_to_tensor(advs[p], tf.float32),
                                'old_probs': tf.convert_to_tensor(probs[p], tf.float32),
                                'return': tf.convert_to_tensor(returns[p], tf.float32)})
        inds = np.arange(len(self.memory))
        out = (0.0, 0.0, 0.0)
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(inds)
            for st in range(0, len(inds), MINI_BATCH_SIZE):
                out = self._train_step(inds[st:st + MINI_BATCH_SIZE], beta)
        return out


def evaluate(envs, agent, tm_ids):
    """MLU (best state of the episode, greedy policy) vs OSPF, per topology."""
    res = {}
    for name, env in envs.items():
        mlus, ospfs = [], []
        for tm in tm_ids:
            _, info = env.reset(options={'tm_id': tm})
            src, dst = info['src'], info['dst']
            dem = info.get('demand', info.get('bw'))
            ospf = float(env.edgeMaxUti[2])
            best = ospf
            while True:
                probs, _ = agent.policy(env, src, dst)
                _, _, term, trunc, inf = env.step(int(np.argmax(probs)), dem, src, dst)
                best = min(best, float(inf['edgeMaxUti'][2]))
                dem, src, dst = inf['bw'], inf['src'], inf['dst']
                if term or trunc:
                    break
            mlus.append(best)
            ospfs.append(ospf)
        res[name] = (np.mean(mlus), 100 * (np.mean(ospfs) - np.mean(mlus)) / np.mean(ospfs))
    return res


def main():
    p = argparse.ArgumentParser(description="Magnitude pruning (manual masks, including the GRU) + PPO fine-tuning")
    p.add_argument('-teacher_ckpt', required=True, help='Checkpoint of the teacher actor (e.g. models/base/ckpt_ACT-282)')
    p.add_argument('-critic_ckpt', default=None, help='Default: the ckpt_CRT with the same number')
    p.add_argument('-top', action='append', nargs=2, metavar=('NAME', 'FOLDER'), required=True,
                   help='Topology and folder that contains TRAIN and EVALUATE (repeatable)')
    p.add_argument('-sparsity', type=float, default=0.5)
    p.add_argument('-episodes', type=int, default=80)
    p.add_argument('-eval_tms', type=int, default=NUM_EVAL_TMS, help='Validation TMs per topology')
    p.add_argument('-out', default=config.MODELS_DIR + '/pruned_50')
    a = p.parse_args()
    critic_ckpt = a.critic_ckpt or a.teacher_ckpt.replace('ckpt_ACT', 'ckpt_CRT')
    tag = f"s{int(round(a.sparsity * 100))}"
    os.makedirs(a.out, exist_ok=True)

    print("=" * 72)
    print(f"  PRUNING v2 | sparsity {a.sparsity:.0%} | {a.episodes} episodes | {ENV_NAME}")
    print("=" * 72)
    train_envs, eval_envs = {}, {}
    for name, folder in a.top:
        folder = folder.rstrip('/\\')
        for store, sub in ((train_envs, 'TRAIN'), (eval_envs, 'EVALUATE')):
            env = gym.make(ENV_NAME).unwrapped
            env.generate_environment(f"{folder}/{sub}", name, EPISODE_LENGTH, NUM_ACTIONS, PCT_DEMANDS)
            env.top_K_critical_demands = True
            store[name] = env
        print(f"  {name}: {train_envs[name].numNodes} nodes, {train_envs[name].numEdges} links")

    agent = PrunedAgent(a.teacher_ckpt, critic_ckpt, a.sparsity)
    eval_tms = list(range(a.eval_tms))
    train_tms = list(range(100))

    r0 = evaluate(eval_envs, agent, eval_tms)
    base = np.mean([v[1] for v in r0.values()])
    print("  After pruning, before fine-tuning: " +
          " | ".join(f"{k}: MLU {v[0]:.4f} ({v[1]:+.1f}% vs OSPF)" for k, v in r0.items()))

    best, hist = -1e9, [{'episodio': 0, 'mejora_val_%': base, **{f'mlu_{k}': v[0] for k, v in r0.items()}}]
    names = list(train_envs)
    for ep in range(a.episodes):
        t0 = time.time()
        lr, beta = cosine_lr(ep, a.episodes), linear_beta(ep, a.episodes)
        agent.optimizer.learning_rate.assign(lr)
        env = train_envs[names[ep % len(names)]]  # rotation of topologies
        tensors, cfeats, acts, vals, masks, rews, probs_l = [], [], [], [], [], [], []
        while len(tensors) < BUFF_SIZE:
            for tm in random.sample(train_tms, NUM_TRAIN_TMS_PER_EP):
                if len(tensors) >= BUFF_SIZE:
                    break
                _, info = env.reset(options={'tm_id': tm})
                dem, src, dst = info.get('demand', info.get('bw')), info['src'], info['dst']
                while True:
                    probs, t = agent.policy(env, src, dst)
                    cf = agent.critic_features(env)
                    v = agent.value(cf)
                    act = np.random.choice(len(probs), p=probs)
                    _, r, term, trunc, inf = env.step(act, dem, src, dst)
                    dem, src, dst = inf['bw'], inf['src'], inf['dst']
                    done = term or trunc
                    tensors.append(t)
                    cfeats.append(cf)
                    acts.append(tf.one_hot(act, len(probs)).numpy())
                    vals.append(v)
                    masks.append(not done)
                    rews.append(r)
                    probs_l.append(probs)
                    if done or len(tensors) >= BUFF_SIZE:
                        break
        vals.append(agent.value(agent.critic_features(env)))
        rets, advs = get_advantages(vals, masks, rews)
        al, cl, en = agent.update(acts, probs_l, tensors, cfeats, rets, advs, beta)
        row = {'episodio': ep + 1, 'topologia': names[ep % len(names)], 'lr': lr, 'beta': beta,
               'recompensa_media': float(np.mean(rews)), 'perdida_actor': float(al),
               'perdida_critico': float(cl), 'entropia': float(en), 't_s': time.time() - t0}
        msg = ""
        if (ep + 1) % EVAL_INTERVAL == 0 or ep + 1 == a.episodes:
            r = evaluate(eval_envs, agent, eval_tms)
            m = float(np.mean([v[1] for v in r.values()]))
            row['mejora_val_%'] = m
            row.update({f'mlu_{k}': v[0] for k, v in r.items()})
            msg = f" | val: mean improvement {m:+.2f}% vs OSPF"
            if m > best:
                best = m
                path = tf.train.Checkpoint(model=agent.actor).save(os.path.join(a.out, f"ckpt_PRUNED_{tag}_BEST"))
                z, tot = sparsity_report(agent.actor)
                msg += f"  <- best ({path}; {z}/{tot} weights at zero)"
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(a.out, f"historial_poda_{tag}.csv"), index=False, sep=';', decimal=',')
        print(f"  Ep {ep + 1:>3} [{row['topologia']:<10}] lr {lr:.1e} beta {beta:.3f} "
              f"rew {row['recompensa_media']:+.4f} actor {float(al):+.4f} critic {float(cl):.4f} "
              f"({row['t_s'] / 60:.1f} min){msg}", flush=True)

    final = tf.train.Checkpoint(model=agent.actor).save(os.path.join(a.out, f"ckpt_PRUNED_{tag}_FINAL"))
    z, tot = sparsity_report(agent.actor)
    print("=" * 72)
    print(f"  Mean improvement after pruning (no FT): {base:+.2f}% | best after FT: {best:+.2f}%")
    print(f"  Weights at zero: {z} of {tot} ({100 * z / tot:.1f} %)")
    print(f"  Best: {a.out}/ckpt_PRUNED_{tag}_BEST-1 | Final: {final}")
    print("=" * 72)


if __name__ == '__main__':
    main()
