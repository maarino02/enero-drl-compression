# -*- coding: utf-8 -*-
"""
cuantizar_a_tflite.py - Cuantizacion post-entrenamiento del Actor de ENERO (corregida)
===============================================================================
Genera hasta tres variantes TFLite a partir del modelo original FP32:

  dinamico : pesos INT8, activaciones en float (Dynamic Range Quantization).
             Es lo que hacia quantize_calibrated.py, aunque no usaba el dataset.
  fp16     : pesos en float16.
  int8     : cuantizacion calibrada REAL: pesos y activaciones en INT8 con un
             dataset representativo obtenido interactuando con el entorno
             (con vuelta a float en las operaciones que no admitan INT8).

Para cada variante mide la FIDELIDAD frente al FP32 en estados no usados en la
calibracion: error absoluto medio de los logits y % de coincidencia del argmax.

Ejemplo:
  python cuantizar_a_tflite.py -teacher_ckpt ../modelos/base/ckpt_ACT-282 ^
     -cal BtAsiaPac ../datasets/principales/NEW_BtAsiaPac/TRAIN ^
     -cal Garr199905 ../datasets/principales/NEW_Garr199905/TRAIN ^
     -cal Goodnet ../datasets/principales/NEW_Goodnet/TRAIN ^
     -out ../modelos/cuantizados
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

import actor_gnn as actor
from evaluar_modelos import build_tensor, TFLiteActor, HPARAMS_TEACHER, HIDDEN_INIT, KERNEL_INIT

PCT_DEMANDS = 0.15


def collect_states(model, envs, tm_ids, max_samples):
    """Recorre episodios guiados por el modelo FP32 (argmax) y guarda los tensores de entrada."""
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
    p = argparse.ArgumentParser(description="Cuantizacion PTQ del actor ENERO (v2)")
    p.add_argument('-teacher_ckpt', required=True)
    p.add_argument('-cal', action='append', nargs=2, metavar=('TOPOLOGIA', 'CARPETA'), required=True,
                   help='Topologia y carpeta TRAIN para calibrar (se puede repetir)')
    p.add_argument('-out', default='../modelos/cuantizados')
    p.add_argument('-modes', nargs='+', default=['fp32', 'dinamico', 'fp16', 'int8'],
                   help='fp32 = TFLite sin cuantizar (referencia de tamano y latencia del formato)')
    p.add_argument('-cal_tms', type=int, default=10, help='TMs por topologia para calibrar')
    p.add_argument('-cal_samples', type=int, default=100, help='Estados maximos por topologia')
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

    print("[1] Recogiendo estados de calibracion y de prueba...")
    cal_states = collect_states(model, envs, range(0, a.cal_tms), a.cal_samples)
    test_states = collect_states(model, envs, range(a.cal_tms, a.cal_tms + 5), 40)
    print(f"    calibracion: {len(cal_states)} estados | prueba de fidelidad: {len(test_states)} estados")

    spec = [tf.TensorSpec([None, 32], tf.float32), tf.TensorSpec([None], tf.int32),
            tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.int32)]

    @tf.function(input_signature=spec)
    def model_forward(link_state, states_graph_ids, states_first, states_second, sates_num_edges):
        return model(link_state, states_graph_ids, states_first, states_second, sates_num_edges, training=False)

    concrete = model_forward.get_concrete_function()
    inf_path = os.path.join(a.out, 'informe_cuantizacion.json')
    report = json.load(open(inf_path, encoding='utf-8')) if os.path.exists(inf_path) else {}  # se conserva lo anterior
    for mode in a.modes:
        print(f"\n[2] Convirtiendo modo '{mode}'...")
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
            print(f"    [ERROR] La conversion '{mode}' ha fallado: {e}")
            report[mode] = {'error': str(e)}
            continue
        path = os.path.join(a.out, f"actor_{mode}.tflite")
        with open(path, 'wb') as fh:
            fh.write(blob)
        fid = fidelity(model, path, test_states)
        report[mode] = {'fichero': path, 'bytes': len(blob), **fid}
        print(f"    OK -> {path} ({len(blob)} B) | MAE logits={fid['mae_logits']:.5f} "
              f"| acuerdo argmax={fid['acuerdo_argmax_%']:.1f}%")

    with open(inf_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nInforme guardado en {a.out}/informe_cuantizacion.json")


if __name__ == '__main__':
    main()
