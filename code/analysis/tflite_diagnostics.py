"""
analysis/tflite_diagnostics.py - Diagnostics of the TFLite latency
==================================================================
Compares, with 1 thread, the latency of the TF actor and of the TFLite FP32 actor with inputs of
the same size and of alternating sizes (K*E changes with each demand), the cost of invoke() alone
and the operations of the TFLite graph. Its output is in results/thread_latency/diagnostico_tflite.txt.

Usage (from code/):  python -m analysis.tflite_diagnostics
"""
import sys, time


def main():
    sys.argv=['x']
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    import numpy as np, gymnasium as gym, gym_graph
    import config
    from evaluation.evaluate import build_tensor, TFLiteActor, TFActor, HPARAMS_TEACHER
    D=config.MAIN_TOPOLOGIES_DIR + '/NEW_Janetbackbone/EVALUATE'
    e=gym.make('GraphEnv-v16').unwrapped; e.generate_environment(D,'Janetbackbone',100,100,0.15); e.top_K_critical_demands=True
    _,info=e.reset(options={'tm_id':0})
    ts=[]; src,dst,dem=info['src'],info['dst'],info['demand']
    for _ in range(6):
        t=build_tensor(e,src,dst,32); ts.append(t)
        _,_,_,_,inf=e.step(0,dem,src,dst); src,dst,dem=inf['src'],inf['dst'],inf['bw']
    print("sizes K*E:", [int(t['link_state'].shape[0]) for t in ts])
    tfa=TFActor('o',config.MODELS_DIR + '/base/ckpt_ACT-282',HPARAMS_TEACHER)
    for name, ag in [('TF', tfa), ('TFLite_FP32', TFLiteActor('f',config.MODELS_DIR + '/tflite/actor_fp32.tflite',1))]:
        for _ in range(10): ag.predict(ts[0])
        t0=time.perf_counter()
        for _ in range(50): ag.predict(ts[0])
        same=(time.perf_counter()-t0)/50*1000
        t0=time.perf_counter()
        for i in range(50): ag.predict(ts[i%len(ts)])
        alt=(time.perf_counter()-t0)/50*1000
        print(f"{name:12s} same size: {same:6.2f} ms | alternating sizes: {alt:6.2f} ms")
    # Invoke without copying the inputs
    q=TFLiteActor('f',config.MODELS_DIR + '/tflite/actor_fp32.tflite',1); q.predict(ts[0])
    t0=time.perf_counter()
    for _ in range(50): q.it.invoke()
    print(f"TFLite invoke() only: {(time.perf_counter()-t0)/50*1000:.2f} ms")
    # Operations of the TFLite graph
    try:
        from tensorflow.lite.python import analyzer
    except Exception: analyzer=None
    ops = {}
    for d in q.it._get_ops_details():
        ops[d['op_name']] = ops.get(d['op_name'],0)+1
    print("TFLite ops:", dict(sorted(ops.items(), key=lambda x:-x[1])))


if __name__ == '__main__':
    main()
