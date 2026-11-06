import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import gymnasium as gym
import gym_graph
import random
import argparse
import time as tt
import tensorflow as tf
import actor_gnn as actor
import pickle
import sys

sys.setrecursionlimit(2000)

ENV_MIDDROUT_AGENT_SP = 'GraphEnv-v16'
ENV_SIMM_ANEAL_AGENT = 'GraphEnv-v15'
SEED = 9

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(1)
random.seed(SEED)

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

class COMPRESSED_AGENT:
    def __init__(self, model_path, model_type):
        self.model_type = model_type
        self.link_state_dim = 16 if model_type == 'student' else 32
        
        if model_type == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
        else:
            if model_type == 'student':
                hparams = {'l2': 0.0001, 'dropout_rate': 0.0, 'link_state_dim': 16, 'readout_units': 16, 'learning_rate': 0.001, 'T': 4, 'num_readout_layers': 2, 'activation_fn': 'selu'}
            else:
                hparams = {'l2': 0.0001, 'dropout_rate': 0.0, 'link_state_dim': 32, 'readout_units': 32, 'learning_rate': 0.0002, 'T': 8, 'num_readout_layers': 4, 'activation_fn': 'selu'}
            hidden_init = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
            kernel_init = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
            self.model = actor.myModel(hparams, hidden_init, kernel_init)
            dummy_link = tf.ones([1, self.link_state_dim])
            dummy_graph = tf.zeros([1], dtype=tf.int32)
            dummy_f = tf.zeros([1], dtype=tf.int32)
            dummy_s = tf.zeros([1], dtype=tf.int32)
            dummy_num = tf.constant(1, dtype=tf.int32)
            self.model(dummy_link, dummy_graph, dummy_f, dummy_s, dummy_num, training=False)
            checkpoint = tf.train.Checkpoint(model=self.model)
            checkpoint.restore(model_path).expect_partial()

    def get_graph_features(self, env, source, destination):
        bw_allocated_feature = env.edge_state[:,2]
        utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'bw_allocated': tf.convert_to_tensor(value=bw_allocated_feature, dtype=tf.float32),
            'utilization': tf.convert_to_tensor(value=np.divide(utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['bw_allocated'] = tf.reshape(sample['bw_allocated'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['utilization'], sample['capacity'], sample['bw_allocated']], axis=1)
        paddings = tf.constant([[0, 0], [0, self.link_state_dim - 3]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']], 'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}
        return inputs

    def pred_action_node_distrib_sp(self, env, source, destination):
        list_k_features = list()
        middlePointList = env.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        itMidd = 0
        
        while itMidd < len(middlePointList):
            env.mark_action_sp(source, middlePointList[itMidd], source, destination)
            if middlePointList[itMidd]!=destination:
                env.mark_action_sp(middlePointList[itMidd], destination, source, destination)
            
            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)
            env.edge_state[:,2] = 0
            itMidd = itMidd + 1

        vs = [v for v in list_k_features]
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = old_cummax(vs, lambda v: v['first'])
        second_offset = old_cummax(vs, lambda v: v['second'])

        tensor = {
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
        }        

        if self.model_type == 'tflite':
            # Map input tensors to interpreter
            def get_idx(name):
                for detail in self.input_details:
                    if name in detail['name']: return detail['index']
                return -1
            self.interpreter.set_tensor(get_idx('link_state'), tensor['link_state'])
            self.interpreter.set_tensor(get_idx('graph_id'), tensor['graph_id'])
            self.interpreter.set_tensor(get_idx('first'), tensor['first'])
            self.interpreter.set_tensor(get_idx('second'), tensor['second'])
            self.interpreter.set_tensor(get_idx('num_edges'), tf.reshape(tensor['num_edges'], []))
            self.interpreter.invoke()
            r = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            r = self.model(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'], tensor['num_edges'], training=False)
            
        listQValues = tf.reshape(r, (1, len(r)))
        softMaxQValues = tf.nn.softmax(listQValues)
        return softMaxQValues.numpy()[0], tensor


class HILL_CLIMBING:
    def __init__(self, env):
        self.num_actions = env.K 

    def get_value_sp(self, env, source, destination, action):
        middlePointList = list(env.src_dst_k_middlepoints[str(source) +':'+ str(destination)])
        middlePoint = middlePointList[action]
        env.allocate_to_destination_sp(source, middlePoint, source, destination)
        if middlePoint!=destination:
            env.allocate_to_destination_sp(middlePoint, destination, source, destination)
            env.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        currentValue = -1000000 
        position = 0
        for i in env.graph:
            for j in env.graph[i]:
                link_capacity = env.links_bw[i][j]
                if env.edge_state[position][0]/link_capacity>currentValue:
                    currentValue = env.edge_state[position][0]/link_capacity
                position = position + 1
        
        if str(source)+':'+str(destination) in env.sp_middlepoints:
            middlepoint = env.sp_middlepoints[str(source)+':'+str(destination)]
            env.decrease_links_utilization_sp(source, middlepoint, source, destination)
            env.decrease_links_utilization_sp(middlepoint, destination, source, destination)
            del env.sp_middlepoints[str(source)+':'+str(destination)] 
        else: 
            env.decrease_links_utilization_sp(source, destination, source, destination)
        
        return -currentValue
    
    def explore_neighbourhood_DRL_sp(self, env):
        dem_iter = 0
        nextVal = -1000000
        next_state = None

        for elem in env.list_eligible_demands:
            source = elem[0]
            dest = elem[1]
            for action in range(len(env.src_dst_k_middlepoints[str(source)+':'+str(dest)])):
                middlepoint = -1
                if str(source)+':'+str(dest) in env.sp_middlepoints: 
                    middlepoint = env.sp_middlepoints[str(source)+':'+str(dest)] 
                    env.decrease_links_utilization_sp(source, middlepoint, source, dest)
                    env.decrease_links_utilization_sp(middlepoint, dest, source, dest)
                    del env.sp_middlepoints[str(source)+':'+str(dest)] 
                else: 
                    env.decrease_links_utilization_sp(source, dest, source, dest)

                evalState = self.get_value_sp(env, source, dest, action)
                if evalState > nextVal:
                    nextVal = evalState
                    next_state = (action, source, dest) 
                
                if middlepoint>=0:
                    env.allocate_to_destination_sp(source, middlepoint, source, dest)
                    env.allocate_to_destination_sp(middlepoint, dest, source, dest)
                    env.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                else:
                    env.allocate_to_destination_sp(source, dest, source, dest)
        return nextVal, next_state

def play_middRout_games_sp(tm_id, env_middRout_sp, agent, timesteps):
    demand, info = env_middRout_sp.reset(options={'tm_id': tm_id})
    source = info['src']
    destination = info['dst']
    
    initMaxUti = env_middRout_sp.edgeMaxUti
    initMaxUti_val = initMaxUti[2] if isinstance(initMaxUti, tuple) else float(initMaxUti)
    OSPF_init = initMaxUti_val
    
    best_routing = env_middRout_sp.sp_middlepoints_step.copy()
    list_of_demands_to_change = env_middRout_sp.list_eligible_demands
    timesteps.append((0, initMaxUti_val))

    start = tt.time()
    time_start_DRL = start
    while 1:
        action_dist, tensor = agent.pred_action_node_distrib_sp(env_middRout_sp, source, destination)
        action = np.argmax(action_dist)
        obs, reward, done, truncated, info = env_middRout_sp.step(action, demand, source, destination)
        
        demand = info["bw"]
        source = info["src"]
        destination = info["dst"]
        
        maxLinkUti = info["edgeMaxUti"]
        maxLinkUti_val = maxLinkUti[2] if isinstance(maxLinkUti, tuple) else float(maxLinkUti)
        
        if maxLinkUti_val < initMaxUti_val:
            initMaxUti_val = maxLinkUti_val 
            best_routing = env_middRout_sp.sp_middlepoints_step.copy()
            timesteps.append((tt.time()-time_start_DRL, initMaxUti_val))
        if done:
            break
    end = tt.time()
    return initMaxUti_val, end-start, OSPF_init, best_routing, list_of_demands_to_change, time_start_DRL

def play_DRL_GNN_sp_hill_climbing_games(tm_id, best_routing, list_of_demands_to_change, timesteps, time_start_DRL, args):
    env_hill_climb = gym.make(ENV_SIMM_ANEAL_AGENT).unwrapped
    env_hill_climb.generate_environment(args.f, args.g, 100, 100, 0.15)
    
    currentVal, info = env_hill_climb.reset_DRL_hill_sp(tm_id, best_routing, list_of_demands_to_change)
    
    # Asegurarnos de que currentVal sea un float escalar negativo
    if isinstance(currentVal, tuple):
        currentVal = currentVal[2]
    currentVal = -abs(float(np.squeeze(currentVal)))
    
    hill_climb_agent = HILL_CLIMBING(env_hill_climb)
    start = tt.time()
    while 1:
        nextVal, next_state = hill_climb_agent.explore_neighbourhood_DRL_sp(env_hill_climb)
        nextVal = -abs(float(np.squeeze(nextVal)))
        
        if nextVal<=currentVal or (abs(nextVal-currentVal)<1e-4):
            break
        
        action = next_state[0]
        source = next_state[1]
        dest = next_state[2]
       
        if str(source)+':'+str(dest) in env_hill_climb.sp_middlepoints:
            middlepoint = env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)]
            env_hill_climb.decrease_links_utilization_sp(source, middlepoint, source, dest)
            env_hill_climb.decrease_links_utilization_sp(middlepoint, dest, source, dest)
            del env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)] 
        else:
            env_hill_climb.decrease_links_utilization_sp(source, dest, source, dest)
        
        step_val = env_hill_climb.step_hill_sp(action, source, dest)
        if isinstance(step_val, tuple):
            step_val = step_val[0]
        currentVal = -abs(float(np.squeeze(step_val)))
        
        timer = tt.time()
        timesteps.append((timer-time_start_DRL, abs(currentVal)))
    end = tt.time()
    return abs(currentVal), end-start


def main():
    parser = argparse.ArgumentParser("Evaluar DRL Model en Topología Única - FULL EVAL PIPELINE")
    parser.add_argument('-g', help='Nombre de la Topología', default='EliBackbone')
    parser.add_argument('-f', help='Ruta al Dataset de evaluación', default='../datasets/zoo/EliBackbone')
    parser.add_argument('-model_path', help='Ruta del modelo a evaluar', required=True)
    parser.add_argument('-model_type', help='Tipo: original | pruned | student | tflite', required=True)
    parser.add_argument('-t', help='ID Matriz tráfico', type=int, default=0)
    parser.add_argument('-n', help='No. de Acciones Máx', type=int, default=100)
    args = parser.parse_args()
    
    args.o = f'./results_COMPRESSED/{args.g}_{args.model_type}'
    
    print(f"=== INICIANDO EVALUACIÓN MASIVA ({args.g}) - MODELO: {args.model_type} ===")
    
    env = gym.make(ENV_MIDDROUT_AGENT_SP).unwrapped
    env.generate_environment(args.f, args.g, 100, args.n, 0.15)
    
    agent = COMPRESSED_AGENT(args.model_path, args.model_type)
    
    timesteps = []
    
    # 1. Enrutamiento Inicial GNN (play_middRout_games_sp)
    initMaxUti, _, OSPF_init, best_routing, list_eligible, time_start_DRL = play_middRout_games_sp(args.t, env, agent, timesteps)
    
    # 2. Refinamiento Hill Climbing (play_DRL_GNN_sp_hill_climbing_games)
    finalMaxUti, _ = play_DRL_GNN_sp_hill_climbing_games(args.t, best_routing, list_eligible, timesteps, time_start_DRL, args)
    
    final_mlu = finalMaxUti[2] if isinstance(finalMaxUti, tuple) else float(finalMaxUti)
    
    # Guardar resultados
    res = {'topology': args.g, 'tm_id': args.t, 'mlu': final_mlu}
    out_file = os.path.join(args.o, f"res_{args.g}_{args.t}.pckl")
    os.makedirs(args.o, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(res, f)
    
    print(f"[+] TM {args.t} Finalizada. MLU Final (tras Hill Climbing): {res['mlu']:.4f}")

if __name__ == "__main__":
    main()
