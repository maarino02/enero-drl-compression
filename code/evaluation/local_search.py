"""
evaluation/local_search.py - Hill Climbing local search (LS) of ENERO
=====================================================================
HILL_CLIMBING explores, for the current routing of an environment GraphEnv-v15, all the
candidate middlepoints of the eligible (critical) demands and returns the best neighbour.
It is used by evaluation/evaluate.py, both after the DRL stage (DRL+HC) and from OSPF (LS only).
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import tensorflow as tf
import sys

sys.setrecursionlimit(2000)

SEED = 9

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(1)
random.seed(SEED)


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
