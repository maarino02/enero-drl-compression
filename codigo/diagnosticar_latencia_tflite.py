import sys, time; sys.path.insert(0,'.'); sys.argv=['x']
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np, gymnasium as gym, gym_graph
from evaluar_modelos import build_tensor, TFLiteActor, TFActor, HPARAMS_TEACHER
D='../datasets/principales/NEW_Janetbackbone/EVALUATE'
e=gym.make('GraphEnv-v16').unwrapped; e.generate_environment(D,'Janetbackbone',100,100,0.15); e.top_K_critical_demands=True
_,info=e.reset(options={'tm_id':0})
ts=[]; src,dst,dem=info['src'],info['dst'],info['demand']
for _ in range(6):
    t=build_tensor(e,src,dst,32); ts.append(t)
    _,_,_,_,inf=e.step(0,dem,src,dst); src,dst,dem=inf['src'],inf['dst'],inf['bw']
print("tamanos K*E:", [int(t['link_state'].shape[0]) for t in ts])
tfa=TFActor('o','../modelos/base/ckpt_ACT-282',HPARAMS_TEACHER)
for name, ag in [('TF', tfa), ('TFLite_FP32', TFLiteActor('f','../modelos/cuantizados/actor_fp32.tflite',1))]:
    for _ in range(10): ag.predict(ts[0])
    t0=time.perf_counter()
    for _ in range(50): ag.predict(ts[0])
    same=(time.perf_counter()-t0)/50*1000
    t0=time.perf_counter()
    for i in range(50): ag.predict(ts[i%len(ts)])
    alt=(time.perf_counter()-t0)/50*1000
    print(f"{name:12s} mismo tamano: {same:6.2f} ms | tamanos alternos: {alt:6.2f} ms")
# Invoke sin copiar entradas
q=TFLiteActor('f','../modelos/cuantizados/actor_fp32.tflite',1); q.predict(ts[0])
t0=time.perf_counter()
for _ in range(50): q.it.invoke()
print(f"TFLite solo invoke(): {(time.perf_counter()-t0)/50*1000:.2f} ms")
# Operaciones del grafo TFLite
try:
    from tensorflow.lite.python import analyzer
except Exception: analyzer=None
ops = {}
for d in q.it._get_ops_details():
    ops[d['op_name']] = ops.get(d['op_name'],0)+1
print("ops TFLite:", dict(sorted(ops.items(), key=lambda x:-x[1])))
