# -*- coding: utf-8 -*-
"""
compression/quantize.py - Post-training quantization of the ENERO actor (corrected)
====================================================================================
Generates up to three TFLite variants from the original FP32 model:

  dinamico : INT8 weights, float activations (Dynamic Range Quantization).
             This is what the first version of the quantization script did, although it did not use the dataset.
  fp16     : float16 weights.
  int8     : REAL calibrated quantization: INT8 weights and activations with a
             representative dataset obtained by interacting with the environment
             (falling back to float in the operations that do not support INT8).

For each variant it measures the FIDELITY with respect to FP32 on states not used in the
calibration: mean absolute error of the logits and % of agreement of the argmax.

Example (from code/):
  python -m compression.quantize -teacher_ckpt models/base/ckpt_ACT-282 ^
     -cal BtAsiaPac <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_BtAsiaPac/TRAIN ^
     -cal Garr199905 <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_Garr199905/TRAIN ^
     -cal Goodnet <DATASETS_DIR>/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_Goodnet/TRAIN ^
     -out models/tflite
"""
import os
import json
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import gymnasium as gym
import gym_graph  # noqa: F401
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from agent import actor
from evaluation.evaluate import build_tensor, TFLiteActor, HPARAMS_TEACHER, HIDDEN_INIT, KERNEL_INIT
import config

PCT_DEMANDS = 0.15


def collect_states(model, envs, tm_ids, max_samples):
    """Runs episodes guided by the FP32 model (argmax) and stores the input tensors."""
    out = []
    for env in envs:
        start = len(out)
        for tm in tm_ids:
            if len(out) - start >= max_samples:
                break
            _, info = env.reset(options={'tm_id': tm})
            src, dst = info['src'], info['dst']
            demand = info.get('demand', info.get('bw'))
            while True:
                t = build_tensor(env, src, dst, 32)
                out.append([t['link_state'].numpy().astype(np.float32), t['graph_id'].numpy().astype(np.int32),
                            t['first'].numpy().astype(np.int32), t['second'].numpy().astype(np.int32),
                            np.array(int(t['num_edges']), dtype=np.int32)])
                a = int(np.argmax(model(t['link_state'], t['graph_id'], t['first'], t['second'],
                                        t['num_edges'], training=False).numpy()))
                _, _, term, trunc, info = env.step(a, demand, src, dst)
                demand, src, dst = info['bw'], info['src'], info['dst']
                if term or trunc or len(out) - start >= max_samples:
                    break
    return out


def fidelity(model, tfl_path, states):
    q = TFLiteActor('q', tfl_path)
    errs, agree = [], []
    for ls, gid, f, s, n in states:
        t = {'link_state': tf.constant(ls), 'graph_id': tf.constant(gid), 'first': tf.constant(f),
             'second': tf.constant(s), 'num_edges': tf.constant(n)}
        ref = model(t['link_state'], t['graph_id'], t['first'], t['second'], t['num_edges'],
                    training=False).numpy().reshape(-1)
        out = q.predict(t)
        errs.append(float(np.mean(np.abs(ref - out))))
        agree.append(int(np.argmax(ref)) == int(np.argmax(out)))
    return {'mae_logits': float(np.mean(errs)), 'acuerdo_argmax_%': 100 * float(np.mean(agree)),
            'n_estados': len(states)}


def main():
    p = argparse.ArgumentParser(description="PTQ quantization of the ENERO actor (v2)")
    p.add_argument('-teacher_ckpt', required=True)
    p.add_argument('-cal', action='append', nargs=2, metavar=('TOPOLOGY', 'FOLDER'), required=True,
                   help='Topology and TRAIN folder used for calibration (repeatable)')
    p.add_argument('-out', default=config.MODELS_DIR + '/tflite')
    p.add_argument('-modes', nargs='+', default=['fp32', 'dinamico', 'fp16', 'int8'],
                   help='fp32 = non-quantized TFLite (size and latency reference of the format)')
    p.add_argument('-cal_tms', type=int, default=10, help='TMs per topology used for calibration')
    p.add_argument('-cal_samples', type=int, default=100, help='Maximum number of states per topology')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = actor.myModel(HPARAMS_TEACHER, HIDDEN_INIT, KERNEL_INIT)
    model.build()
    tf.train.Checkpoint(model=model).restore(a.teacher_ckpt).expect_partial()

    envs = []
    for name, folder in a.cal:
        env = gym.make('GraphEnv-v16').unwrapped
        env.generate_environment(folder, name, 100, 100, PCT_DEMANDS)
        env.top_K_critical_demands = True
        envs.append(env)

    print("[1] Collecting calibration and test states...")
    cal_states = collect_states(model, envs, range(0, a.cal_tms), a.cal_samples)
    test_states = collect_states(model, envs, range(a.cal_tms, a.cal_tms + 5), 40)
    print(f"    calibration: {len(cal_states)} states | fidelity test: {len(test_states)} states")

    spec = [tf.TensorSpec([None, 32], tf.float32), tf.TensorSpec([None], tf.int32),
            tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.int32)]

    @tf.function(input_signature=spec)
    def model_forward(link_state, states_graph_ids, states_first, states_second, sates_num_edges):
        return model(link_state, states_graph_ids, states_first, states_second, sates_num_edges, training=False)

    concrete = model_forward.get_concrete_function()
    inf_path = os.path.join(a.out, 'informe_cuantizacion.json')
    report = json.load(open(inf_path, encoding='utf-8')) if os.path.exists(inf_path) else {}  # previous content is kept
    for mode in a.modes:
        print(f"\n[2] Converting mode '{mode}'...")
        conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
        if mode != 'fp32':
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if mode == 'fp16':
            conv.target_spec.supported_types = [tf.float16]
        elif mode == 'int8':
            conv.representative_dataset = lambda: iter(cal_states)
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        try:
            blob = conv.convert()
        except Exception as e:  # noqa: BLE001
            print(f"    [ERROR] Conversion '{mode}' failed: {e}")
            report[mode] = {'error': str(e)}
            continue
        path = os.path.join(a.out, f"actor_{mode}.tflite")
        with open(path, 'wb') as fh:
            fh.write(blob)
        fid = fidelity(model, path, test_states)
        report[mode] = {'fichero': path, 'bytes': len(blob), **fid}
        print(f"    OK -> {path} ({len(blob)} B) | MAE logits={fid['mae_logits']:.5f} "
              f"| argmax agreement={fid['acuerdo_argmax_%']:.1f}%")

    with open(inf_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReport saved in {a.out}/informe_cuantizacion.json")


if __name__ == '__main__':
    main()
