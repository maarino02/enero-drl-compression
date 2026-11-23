# =============================================================================
# training/train_worker.py  (TFG - corrected version of ENERO's training script)
# Launched by training/train.py for each batch of episodes.
# Changes with respect to the original training script:
#  [v2-1] Environment GraphEnv-v17: unbiased reward (previous std of the complete state).
#  [v2-2] Graph mode enabled (before: run_functions_eagerly(True)): much faster.
#  [v2-3] tf.reshape(r, (1, -1)) instead of len(r) (needed in graph mode).
#  [v2-4] Inference of the actor and the critic during the collection in tf.function.
#  [v2-5] Validation on the TMs of EVALUATE (before: TMs of TRAIN).
#  [v2-6] Fixed the validation loop of the 2nd and 3rd topology (it did not move to the next demand).
#  [v2-7] Additional log: mean validation MLU and std/min of the normalized utilizations.
#  [v2-8] Models in models/base and log in logs/training_log.txt.
# =============================================================================
import multiprocessing as mp
import os
import time

# 1. Set the start method before anything else
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
import numpy as np
import gymnasium as gym

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import gym_graph
import random
from agent import critic
from agent import actor
import config
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress extra warnings
tf.config.run_functions_eagerly(False) # [v2-2] graph mode
from collections import deque
import argparse
import pickle
from keras import backend as K

# Use BtAsiaPac, EliBackbone and Goodnet for training

'''
This script is run for each batch of training episodes, performing a given number of episodes
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # NVIDIA specific

# In this experiment we learn how to pick the best action(middlepoint) by marking for each middlepoint
# the action in the topology edges. Rewards are given per time-step.
# We also remove the SP paths that can create a loop with the source node!

ENV_NAME = 'GraphEnv-v17'  # [v2-1]

# Indicates how many time-steps has an episode
EPISODE_LENGTH = 100 # We are not using it now
SEED = 9
MINI_BATCH_SIZE = 32
experiment_letter = "_B_NEW"
take_critic_demands = True # True if we want to take the demands from the most critical links, True if we want to take the largest
# Percentage of all the demands that are considered as critical demands
percentage_demands = 15 # Percentage of demands that will be used in the optimization
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100

EVALUATION_EPISODES = 20 # As the demand selection is deterministic, it doesn't make sense to evaluate multiple times over the same TM
PPO_EPOCHS = 8
num_samples_top1 = int(np.ceil(percentage_demands*380))*5
num_samples_top2 = int(np.ceil(percentage_demands*506))*4
num_samples_top3 = int(np.ceil(percentage_demands*272))*6

BUFF_SIZE = num_samples_top1+num_samples_top2+num_samples_top3 # Experience buffer size. Careful to don't have more samples from one TM!

def make_env(dataset_folder, topology_name, ep_length, k_actions, pct_demands):
    def _init():
        # Imported inside to avoid serialization problems with multiprocessing
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import gymnasium as gym
        import gym_graph
        
        env = gym.make(ENV_NAME)
        # unwrapped is accessed to configure our custom environment
        env.unwrapped.generate_environment(
            dataset_folder, 
            topology_name, 
            ep_length, 
            k_actions, 
            pct_demands
        )
        return env
    return _init
'''
This script is run for each batch of episodes ('episode_iters = 20' episodes, which are the ones run in this script).
The learning rate decays along the training episodes: it is decayed at the beginning of each episode.
'''
# The DECAY_STEPS must be a multiple of args.e (episode_iters)
DECAY_STEPS = 60 # The second value is to indicate every how many PPO EPISODES we decay the lr
DECAY_RATE = 0.96

CRITIC_DISCOUNT = 0.8

# if agent struggles to explore the environment, increase BETA
# if the agent instead is very random in its actions, not allowing it to take good decisions, you should lower it
ENTROPY_BETA = 0.1
ENTROPY_STEP = 240 # From this episode on, the entropy coefficient is reduced (less exploration, so that the DRL agent focuses on a narrower region of the search space)

clipping_val = 0.2

max_grad_norm = 0.5

'''
Folder where each version of the agent (actor+critic) is stored
'''
#differentiation_str = "Enero_3top_"+str_perctg_demands+experiment_letter
differentiation_str = "V2"  # [v2-8]
checkpoint_dir = config.MODELS_DIR + "/base"

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.random.set_seed(1)


NUM_ACTIONS = 100 # For now we have dynamic action space. This means that we consider all nodes as actions but removing the repeated paths

hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
hidden_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(1), seed=SEED)



'''
Receives a list of dictionaries (each one with the information of a candidate middlepoint) and the 'extractor', the function
that extracts the value of a given key from each dictionary. Here the extractor extracts the sender links of each dictionary.

'reduce_max' takes the maximum over the given dimensions.

Returns a list with as many elements as candidate middlepoints for the current demand, where each element is the sum of:
    * Position of the sender link with the highest position, in the graph with the current middlepoint
    * Position of the sender link with the highest position, in each of the graphs of the previous middlepoints

(cumulative sum)
'''

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'): # Defines a name scope

        # For each candidate middlepoint, position of the sender link with the highest position (plus 1, i.e. 1-based) in the graph with that middlepoint
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        # Cumulative sum of the elements of maxes
        # First element
        cummaxes = [tf.zeros_like(maxes[0])]
        # Remaining elements
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

# Decays the learning rate at the beginning of each training episode with an exponential decay (faster the more episodes have been run)
def decayed_learning_rate(step):
    lr = hparams['learning_rate']*(DECAY_RATE ** (step / DECAY_STEPS))
    # A minimum learning rate is always kept
    if lr<10e-5:
        lr = 10e-5
    return lr

class PPOActorCritic:
    def __init__(self):
        self.memory = deque(maxlen=BUFF_SIZE)
        # Indices of the states of the trajectory followed by the agent to collect experience
        self.inds = np.arange(BUFF_SIZE)
        self.listQValues = None
        self.softMaxQValues = None

        self.action = None
        self.softMaxQValues = None
        self.listQValues = None

        self.utilization_feature = None
        self.bw_allocated_feature = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], beta_1=0.9, epsilon=1e-05)
        self.actor = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
        self.actor.build()

        self.critic = critic.myModel(hparams, hidden_init_critic, kernel_init_critic)
        self.critic.build()

        # [v2-4] Graph-mode inference with a fixed signature (no retracing because of the size)
        d = hparams['link_state_dim']
        a_mod, c_mod = self.actor, self.critic
        @tf.function(input_signature=[tf.TensorSpec([None, d], tf.float32), tf.TensorSpec([None], tf.int32),
                                      tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32),
                                      tf.TensorSpec([], tf.int32)])
        def _actor_fwd(ls, gid, f, s, n):
            return a_mod(ls, gid, f, s, n, training=False)
        @tf.function(input_signature=[tf.TensorSpec([None, d], tf.float32), tf.TensorSpec([None], tf.int32),
                                      tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.int32)])
        def _critic_fwd(ls, f, s, n):
            return c_mod(ls, f, s, num_edges_critic=n, training=False)
        self._actor_fwd = _actor_fwd
        self._critic_fwd = _critic_fwd

    def critic_value(self, features):
        # [v2-4] value of the critic for the current state (graph mode)
        v = self._critic_fwd(tf.cast(features['link_state_critic'], tf.float32), tf.cast(features['first_critic'], tf.int32),
                             tf.cast(features['second_critic'], tf.int32), tf.cast(features['num_edges_critic'], tf.int32))
        return v[0].numpy()[0]
    
    '''
    This method builds the graphs that result from applying each of the possible middlepoints.
    Returns:
        * Probability of choosing each middlepoint (action)
        * Information of the state of the graph after applying each middlepoint
    '''
    def pred_action_distrib_sp(self, env, source, destination):
        # List of graph features that are used in the cummax() call
        '''
        This list stores one element per possible middlepoint, such that for each middlepoint:
            * The graph is (abstractly) adjusted assuming that this middlepoint is used to send the data from source to destination
            * A dictionary with the information of that graph is obtained
            * The dictionary is stored in list_k_features
        '''
        list_k_features = list()

        # We get the K-middlepoints between source-destination
        # I.e. the list of intermediate nodes that can be used as middlepoints
        middlePointList = env.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        itMidd = 0
        
        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        # Now we compute the state of the graph if each of the possible middlepoints were applied
        while itMidd < len(middlePointList):
            # Mark the path followed if the middlepoint with index 'itMidd' is used
            env.mark_action_sp(source, middlePointList[itMidd], source, destination)
            # If we allocated to a middlepoint that is not the final destination
            if middlePointList[itMidd]!=destination:
                env.mark_action_sp(middlePointList[itMidd], destination, source, destination)

            # Features of the graph with the middlepoint applied, as a dictionary (the source-destination arguments are actually not used)
            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)

            # We desmark the bw_allocated
            # This holds, for the links of the source-middlepoint-destination path, the usage considering only the source-destination demand
            env.edge_state[:,2] = 0
            itMidd = itMidd + 1

        vs = [v for v in list_k_features] # list with the features of the graph (as a dictionary) for each candidate middlepoint of the current demand

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.

        '''
        graph_ids: list with one element per candidate middlepoint, where each element is:
            a 1-D tensor with as many components as links in the graph, filled with the index of that middlepoint

        EXAMPLE: with 3 candidate middlepoints for the current demand and a graph with 5 links, 'graph_ids' would be:
        [ [0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2] ]
        '''
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]

        # Cumulative sum of the positions of the sender links with the highest position, over the graphs of the different middlepoints
        first_offset = old_cummax(vs, lambda v: v['first'])
        # The same for the receiver links
        second_offset = old_cummax(vs, lambda v: v['second'])

        '''
        The supergraph that contains the different graphs passed to the GNN is modelled with a dictionary:

            * graph_id: 1-D tensor obtained by concatenating the 1-D tensors with the ids of the middlepoints
                - Example: with 3 middlepoints and a graph with 2 links: [0,0,1,1,2,2]

            * link_state: concatenates the hidden state matrices (2-D tensors of shape (number of links x link_state_dim)) along the rows

            * first: concatenates the sender links of each graph with a candidate middlepoint, along the rows. 'old_cummax' is used to shift them.
            The reason is that we do not work on the hidden state matrix of each graph separately, but on a single matrix obtained by
            concatenating the hidden state matrices of all the candidate middlepoints.
            Therefore, the indices must be shifted to work on the part of the global matrix that holds the hidden states of the graph
            with the corresponding middlepoint.
            For example, to get the hidden state of the sender in position 0 of the graph with the second candidate middlepoint we could:
                * Take directly the first row of the 'link_state' of that candidate graph
                * Or, working with the global 'link_state' (that concatenates the hidden state matrices of all the candidates), first skip
                the rows of the 'link_state' of the graph with the first middlepoint, and then take the first row after that offset.
            That is why the 'first' of each graph with a candidate middlepoint is shifted accordingly. The same for 'second'

            * second: the same for the receiver links

            * num_edges: total number of links, adding up the links of the graph of each candidate middlepoint
        '''

        # Initialize dictionary to structure global supergraph
        tensor = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0), # 1-D tensors are concatenated along their only dimension
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]), # math.add_n adds tensors quickly
            }
        )        

        # Predict qvalues for all graphs within tensors
        # The actor is evaluated on the graphs of each middlepoint, giving one value per possible middlepoint
        r = self._actor_fwd(tf.cast(tensor['link_state'], tf.float32), tf.cast(tensor['graph_id'], tf.int32),
                            tf.cast(tensor['first'], tf.int32), tf.cast(tensor['second'], tf.int32),
                            tf.cast(tensor['num_edges'], tf.int32))  # [v2-4]

        self.listQValues = tf.reshape(r, (1, -1))  # [v2-3]
        # Softmax gives the probability of choosing each of the possible middlepoints to send the source-destination data
        self.softMaxQValues = tf.nn.softmax(self.listQValues)

        # Return action distribution
        # Return the probability distribution of the actions and the tensor with the information of all the possible middlepoints
        return self.softMaxQValues.numpy()[0], tensor
    
    '''
    Returns the features of the graph in the current state of the environment.

    Used mainly after (abstractly) marking a candidate middlepoint (i.e. marking the path that would be followed with that middlepoint (edge_state[2]), without changing anything else).
    Note: the link usage is not recomputed with the new route. It returns the initial usage of each link of the topology and, separately, edge_state[2] to know which links make up the path
    of a given middlepoint
    '''
    def get_graph_features(self, env, source, destination):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        # Usage of the link considering only the demand we are working on in a given state of the environment
        self.bw_allocated_feature = env.edge_state[:,2]
        # Traffic that goes through the link
        self.utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature, # capacity of the link relative to the capacity of the link with the highest capacity
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        '''
        For the utilization, capacity and bandwidth of the middlepoint:
            * The 1-D tensor (row vector) is converted into a 2-D tensor (the 2nd dimension with a single value) --> matrix with a single column
        '''
        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['bw_allocated'] = tf.reshape(sample['bw_allocated'][0:sample['num_edges']], [sample['num_edges'], 1])

        # The column vectors are concatenated along the columns, giving a matrix with three columns: utilization, capacity and bandwidth of the middlepoint path (to identify the links of the path)
        hiddenStates = tf.concat([sample['utilization'], sample['capacity'], sample['bw_allocated']], axis=1)

        # The padding is defined with a 2x2 tensor of the form
        # (  0       0          )  --> zeros to add before/after the values of the first dimension of hiddenStates (rows, i.e. number of links)
        # (  0   hparams[...]-3 ) ----> the same for the dimension with the features (utilization, capacity, bandwidth)
        
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 3]]) # Tensor of the form [ [0,0], [0,hparams[...]-3] ], one element per dimension of the tensor to pad
        # The hidden state of each link is extended to dimension 'link_state_dim', which is the input dimension of the model
        # I.e. a tensor that looks like a matrix with one row per link, and for each row (link) a vector of 'link_state_dim' components
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        '''
        Returns a dictionary with:
            * link_state: tensor (matrix) with the hidden state of each link, adapted to the input dimension of the model
            * first and second: senders and receivers for the message passing phase
            * number of links of the graph
        '''
        inputs = {
            'link_state': link_state,
            'first': sample['first'][0:sample['length']],
            'second': sample['second'][0:sample['length']], 
            'num_edges': sample['num_edges']
        }

        return inputs

    # Also returns the features of the current state of the graph.
    # Unlike the previous method, the marked path is not taken into account.
    # Used mainly to store the current state of the graph (without considering the possible middlepoints of the current demand),
    # before finally applying the selected middlepoint, which would make us lose the initial state of the graph.
    def critic_get_graph_features(self, env):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        # Bandwidth currently carried by each link
        self.utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            # we store the utilization (allocated bandwidth / maximum capacity) of each link
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['utilization'], sample['capacity']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state_critic': link_state, 'first_critic': sample['first'][0:sample['length']],
                'second_critic': sample['second'][0:sample['length']], 'num_edges_critic': sample['num_edges']}

        return inputs
    

    '''
    In short, the difference between the two previous methods is:
        * get_graph_features: initial state of the graph + path of a given middlepoint
        * critic_get_graph_features: initial state of the graph (without any candidate middlepoint)
    '''
    

    
    @tf.function(experimental_relax_shapes=True)
    def _critic_step(self, ret, link_state_critic, first_critic, second_critic, num_edges_critic):
        ret = tf.stop_gradient(ret)

        value = self.critic(link_state_critic, first_critic, second_critic,
                    num_edges_critic=num_edges_critic, training=True)[0]
        # Loss: squared difference
        critic_sample_loss = tf.math.squared_difference(ret,value)
        return critic_sample_loss
    
    @tf.function(experimental_relax_shapes=True)

    # Simulates a step of the actor. After collecting experience along a trajectory, the loss is computed for each state of the trajectory
    def _actor_step(self, advantage, old_act, old_policy_probs, link_state, graph_id, \
                    first, second, num_edges):
        # Variables that must not be updated when the gradient is computed and used
        adv = tf.stop_gradient(advantage)
        old_act = tf.stop_gradient(old_act) # One-hot vector with only the applied middlepoint activated
        old_policy_probs = tf.stop_gradient(old_policy_probs)

        # The actor is evaluated on the graphs with the candidate middlepoints
        r = self.actor(link_state, graph_id, first, second, num_edges, training=True)
        qvalues = tf.reshape(r, (1, -1))  # [v2-3]
        # New probabilities of choosing each middlepoint
        newpolicy_probs = tf.nn.softmax(qvalues)
        # From these, the new probability of applying the middlepoint selected by the agent when collecting experience, given this initial state
        # That is, newpolicy_probs2 --> prob of applying the same action given the same initial state, but with the new policy.
        newpolicy_probs2 = tf.math.reduce_sum(old_act * newpolicy_probs[0])
        '''
        Therefore, the probability of applying the action given the initial state is:
            * For the old policy --> tf.math.reduce_sum(old_act * old_policy_probs)
            * For the new policy --> newpolicy_probs2
        '''

        '''
        The loss function is evaluated on the current parameters of the model, to know the loss of the new policy for a given action applied on a given state.
        Following the PPO loss, either the original formula or 'logs' (as here) can be used.
        To do so:
            * Compute the ratio: divide the probability of choosing the same action selected in a state,
            with the old and with the new policy --> 'ratio'
            * Multiply the ratio by the advantage obtained in that initial state --> surr1
        '''
        ratio = tf.exp(tf.math.log(newpolicy_probs2) - tf.math.log(tf.math.reduce_sum(old_act*old_policy_probs)))
        surr1 = -ratio*adv
        surr2 = -tf.clip_by_value(ratio, clip_value_min=1 - clipping_val, clip_value_max=1 + clipping_val) * adv
        '''
        The loss is --> min( ratio*advantage // clip(...)*advantage ), i.e. max(-ratio*advantage, -clip(...)*advantage)
        '''
        loss_sample = tf.maximum(surr1, surr2)


        entropy_sample = -tf.math.reduce_sum(tf.math.log(newpolicy_probs) * newpolicy_probs[0])
        return loss_sample, entropy_sample

    '''
    inds: indices of the states used to update the actor and the critic

    This method updates the agent with a batch of size len(inds):
        * The loss of each state of the batch is computed, both for the actor and for the critic
        * The mean loss of the actor and of the critic (over the batch) are added, giving the mean total loss of the current policy
        * Gradient descent is applied
    '''
    def _train_step_combined(self, inds):
        entropies = []
        actor_losses = []
        critic_losses = []
        # Optimize weights
        with tf.GradientTape() as tape:
            # The actor/critic are applied again on the states of the current batch
            for minibatch_ind in inds:
                # For each selected state, take all the experience collected by the agent in that state
                sample = self.memory[minibatch_ind]

                # ACTOR
                '''
                To update the actor with a given state, pass:
                    * Advantage obtained
                    * Applied action (one-hot vector)
                    * Probability of each possible middlepoint under the policy followed by the agent in that state
                    * State vectors of the links of the graph, for each of the graphs of each possible middlepoint (marked path)
                    * Ids of the different graphs or middlepoints
                    * first and second: for each link, the links that send it messages
                '''
                loss_sample, entropy_sample = self._actor_step(sample["advantage"], sample["old_act"], sample["old_policy_probs"], \
                            sample["link_state"], sample["graph_id"], sample["first"], sample["second"], sample["num_edges"])
                actor_losses.append(loss_sample) # Loss of this element of the batch
                entropies.append(entropy_sample)
                # The loss of this state has just been computed
                
                # Same process for the critic
                '''
                To update the critic, the update method receives:
                    * Value/quality of the policy followed at each time step (with the aggregated advantage of each time step)
                    * State vector of each link of the graph
                    * 'first_critic' and 'second_critic': sender/receiver links of the initial graph, for each time step of the trajectory
                    * number of links of the initial graph, for each time step
                It returns the loss for a given state of the trajectory
                '''
                critic_sample_loss = self._critic_step(sample["return"], sample["link_state_critic"], sample["first_critic"], sample["second_critic"], sample["num_edges_critic"])
                critic_losses.append(critic_sample_loss)
        
            # After computing the actor/critic loss for each state of the batch, the mean actor and critic losses are computed
            # I.e. mean loss of the actor/critic for this batch
            critic_loss = tf.math.reduce_mean(critic_losses)
            final_entropy = tf.math.reduce_mean(entropies)
            actor_loss = tf.math.reduce_mean(actor_losses) - ENTROPY_BETA * final_entropy
            # The mean total loss of the batch weights the critic loss so that it does not overshadow the actor
            total_loss = actor_loss + vf_coef * critic_loss

        '''
        Gradient of the loss function with respect to the following parameters only (on which the partial derivatives are computed):
            * Trainable parameters of the actor
            * Trainable parameters of the critic
        tape.gradient() receives two parameters:
            * Total loss of the batch, on which the gradient is computed
            * Input variables with respect to which the partial derivatives are computed
        Output: 'grad' ---> list of partial derivatives with respect to those variables
        '''
        grad = tape.gradient(total_loss, sources=self.actor.trainable_weights + self.critic.trainable_weights)
        #gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
        grad, _grad_norm = tf.clip_by_global_norm(grad, max_grad_norm)
        # The trainable parameters of the actor and the critic (on which the gradient was computed) are updated with the gradient,
        # using 'Adam' as optimization algorithm
        self.optimizer.apply_gradients(zip(grad, self.actor.trainable_weights + self.critic.trainable_weights))
        entropies.clear()
        actor_losses.clear()
        critic_losses.clear()
        # Return the mean loss of the actor and the critic for this batch of states, and the final entropy
        return actor_loss, critic_loss, final_entropy

    def ppo_update(self, actions, actions_probs, tensors, critic_features, returns, advantages):

        # Length of the trajectory followed to collect experience
        for pos in range(0, int(BUFF_SIZE)): # For each of the states visited by the agent...

            tensor = tensors[pos] # information of each graph with a candidate middlepoint applied
            critic_feature = critic_features[pos] # initial state of the graph (without any middlepoint)
            action = actions[pos] # one-hot vector activating the selected action or middlepoint
            ret_value = returns[pos] # value of the critic updated with the advantage
            adv_value = advantages[pos] # advantage obtained
            action_dist = actions_probs[pos] # probability distribution of each candidate middlepoint
            
            final_tensors = ({
                'graph_id': tensor['graph_id'],
                'link_state': tensor['link_state'], # hidden state (utilization, capacity, bandwidth of the middlepoint path) of each link, by column
                'first': tensor['first'],
                'second': tensor['second'],
                'num_edges': tensor['num_edges'],
                'link_state_critic': critic_feature['link_state_critic'],
                'old_act': tf.convert_to_tensor(action, dtype=tf.float32),
                'advantage': tf.convert_to_tensor(adv_value, dtype=tf.float32),
                'old_policy_probs': tf.convert_to_tensor(action_dist, dtype=tf.float32),
                'first_critic': critic_feature['first_critic'],
                'second_critic': critic_feature['second_critic'],
                'num_edges_critic': critic_feature['num_edges_critic'],
                'return': tf.convert_to_tensor(ret_value, dtype=tf.float32),
            })      
            # The experience obtained in this state is stored in the memory of the agent
            # This is repeated for all the states of the trajectory
            self.memory.append(final_tensors)  

        
        # All the experience obtained by the agent along the trajectory has been stored
        # Now the actor and the critic are updated.

        # To update the agent, the complete trajectory is traversed 'PPO_EPOCHS' times
        '''
        In this iteration, after collecting experience along the trajectory of length BUFF_SIZE, the actor and the critic
        are updated:
        The complete trajectory is traversed a given number of times and, each time:
            * The trajectory is split into batches (sets of time steps), and the agent is updated once per batch
        '''

        for i in range(PPO_EPOCHS):
            # In each PPO_EPOCH the states of the trajectory are shuffled, so that the agent is updated with different batches in each PPO_EPOCH
            np.random.shuffle(self.inds)

            # Update by batches in this episode
            for indice_lote, start in enumerate(range(0, BUFF_SIZE, MINI_BATCH_SIZE)):
                end = start + MINI_BATCH_SIZE
                # Mean actor/critic loss over this batch of states
                actor_loss, critic_loss, final_entropy = self._train_step_combined(self.inds[start:end])

        
        
        self.memory.clear()
        # return the mean actor/critic loss of the last batch of the trajectory in the last PPO_EPOCH
        return actor_loss, critic_loss

'''
Computes the advantage of each of the states of the trajectory along which experience was collected.
The advantage of time step 'i' uses the rewards obtained in the time steps after 'i' (within the trajectory)

 0   1   2   3   4   5
A_0 A_1 A_2 A_3 A_4 A_5

where

A_0 : uses the states i= {0,1,2,3,4,5}
A_1 : uses the states i= {1,2,3,4,5}
...
A__5: uses only i = {5}

'mask' is activated when the last critical demand of a traffic matrix has been rerouted, to avoid mixing actions of the agent on different traffic matrices.
I.e. when computing the advantage of the state of the last critical demand, the next state 't+1' is not considered, because it is the end of the episode for that traffic matrix.

'''
def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        # Delta used in the advantage
        scaled_reward = rewards[i] * reward_scale
        delta = scaled_reward + gamma * values[i + 1] * masks[i] - values[i]
        # Advantage of state 'i' --> delta of state 'i' + advantage of 'i+1' (multiplied by gamma*lambda)
        gae = delta + gamma * lmbda * masks[i] * gae
        # Stored as the first element of the list, pushing all the others forward, so that the advantages are ordered from the first to the last time step
        # The value of state 'i' is added to the advantage, to update the value function.
        returns.insert(0, gae + values[i])

    # The advantages of each time step are also returned, without adding the value.
    adv = np.array(returns) - values[:-1]
    # Normalize advantages to reduce variance
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

if __name__ == "__main__":
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-i', help='iters', type=int, required=True)
    parser.add_argument('-c', help='counter model', type=int, required=True)
    parser.add_argument('-e', help='episode iterations', type=int, required=True)
    parser.add_argument('-f1', help='dataset folder name topology 1', type=str, required=True, nargs='+')
    parser.add_argument('-f2', help='dataset folder name topology 2', type=str, required=True, nargs='+')
    parser.add_argument('-f3', help='dataset folder name topology 3', type=str, required=True, nargs='+')
    
    # New hyperparameters
    parser.add_argument('-lr', help='Learning rate', type=float, default=0.0002)
    parser.add_argument('-T', help='Message passing iterations', type=int, default=5)
    parser.add_argument('-dim', help='Network capacity (link state & readout)', type=int, default=20)
    parser.add_argument('-dropout', help='Dropout rate', type=float, default=0.0)
    parser.add_argument('-readout_layers', help='Depth of readout MLP', type=int, default=2)
    parser.add_argument('-gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('-lmbda', help='GAE lambda', type=float, default=0.95)
    parser.add_argument('-vf_coef', help='Critic loss coefficient', type=float, default=0.5)
    parser.add_argument('-reward_scale', help='Reward scaling multiplier', type=float, default=1.0)
    parser.add_argument('-activation', help='Activation function for Readout', type=str, default='selu')
    args = parser.parse_args()

    gamma = args.gamma
    lmbda = args.lmbda
    vf_coef = args.vf_coef
    reward_scale = args.reward_scale

    hparams = {
        'l2': 0.0001,
        'link_state_dim': args.dim,
        'readout_units': args.dim,
        'learning_rate': args.lr,
        'T': args.T,
        'dropout_rate': args.dropout,
        'num_readout_layers': args.readout_layers,
        'activation_fn': args.activation
    }

    # Location of the datasets (see config.py)
    BASE_PATH_DATASETS = config.DATASETS_DIR

    dataset_folder_name1 = f"{BASE_PATH_DATASETS}/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_"+args.f1[0]+"/TRAIN"
    dataset_folder_name2 = f"{BASE_PATH_DATASETS}/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_"+args.f2[0]+"/TRAIN"
    dataset_folder_name3 = f"{BASE_PATH_DATASETS}/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_"+args.f3[0]+"/TRAIN"
    # [v2-5] VALIDATION folders (TMs not seen during training)
    eval_folder_name1 = dataset_folder_name1.replace("/TRAIN", "/EVALUATE")
    eval_folder_name2 = dataset_folder_name2.replace("/TRAIN", "/EVALUATE")
    eval_folder_name3 = dataset_folder_name3.replace("/TRAIN", "/EVALUATE")

    # All the GPUs are hidden from TensorFlow to force a stable execution on the CPU RAM
    tf.config.set_visible_devices([], 'GPU')

    print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))

    # Get the environment and extract the number of actions.
    env_training1 = gym.make(ENV_NAME)
    env_training1.unwrapped.generate_environment(dataset_folder_name1, "BtAsiaPac", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training1.unwrapped.top_K_critical_demands = take_critic_demands

    env_training2 = gym.make(ENV_NAME)
    env_training2.unwrapped.generate_environment(dataset_folder_name2, "Garr199905", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training2.unwrapped.top_K_critical_demands = take_critic_demands

    env_training3 = gym.make(ENV_NAME)
    env_training3.unwrapped.generate_environment(dataset_folder_name3, "Goodnet", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training3.unwrapped.top_K_critical_demands = take_critic_demands

    env_eval1 = gym.make(ENV_NAME)
    env_eval1.unwrapped.generate_environment(eval_folder_name1, "BtAsiaPac", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval1.unwrapped.top_K_critical_demands = take_critic_demands

    env_eval2 = gym.make(ENV_NAME)
    env_eval2.unwrapped.generate_environment(eval_folder_name2, "Garr199905", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval2.unwrapped.top_K_critical_demands = take_critic_demands

    env_eval3 = gym.make(ENV_NAME)
    env_eval3.unwrapped.generate_environment(eval_folder_name3, "Goodnet", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval3.unwrapped.top_K_critical_demands = take_critic_demands

    # If the folder where the version of the DRL agent is stored at the end of each episode (model checkpoints) does not exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Open (or create) in append mode the log file, which keeps a record of the training along the current episode
    fileLogs = open(config.LOGS_DIR + "/training_log.txt", "a")

    # Load maximum reward from previous iterations and the current lr
    '''
    At the beginning of this new batch of episodes, the temporary file is opened to know the maximum reward obtained so far (considering all the past episodes)
    and the learning rate used in the last episode.
    '''
    if os.path.exists(config.CODE_DIR + "/tmp/" + differentiation_str + "tmp.pckl"):
        f = open(config.CODE_DIR + "/tmp/" + differentiation_str + "tmp.pckl", 'rb')
        max_reward, _ = pickle.load(f)
        f.close()
    else:
        # The temporary file does not exist ---> this is the first training episode
        max_reward = -1000

    if args.i>=ENTROPY_STEP:
        ENTROPY_BETA = ENTROPY_BETA/10 # Exploration is reduced
    
    # The agent is initialized
    agent = PPOActorCritic()

    # Checkpoint locations of the actor and of the critic of the DRL agent
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # Objects that save/restore the state of the models and of their optimizers
    checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
    checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)

    # If this is not the first episode, i.e. the DRL agent has already been trained for at least one complete episode, the state of the model at the end of the previous episode is loaded
    if args.i>0:
        # -1 because the current value is to store the model that we train in this iteration
        '''
        The last saved state of the actor/critic and of their optimizers (which use a previously changed learning rate) is restored
        The '.index' (indices of the parameters of the model) and '.data' (actual values of the parameters indexed by .index) files are used

        IMPORTANT: the last version saved at the end of the previous episode is restored, and this restored version is used at the beginning of the new episodes
        '''
        try:
            checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
            checkpoint_actor.restore(checkpoint_dir + "/ckpt_ACT-" + str(args.c-1)) #..../ckpt_ACT-<counter_store_model-1>, because the '-c' argument of this script is the new version of the model that we start with
            checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)
            checkpoint_critic.restore(checkpoint_dir + "/ckpt_CRT-" + str(args.c-1))
        except tf.errors.NotFoundError:
            print("The previous model was not found (possibly corrupted). Initializing the agent weights from scratch.")
            pass
    # The last saved state of the actor/critic has been loaded at the beginning of this new episode

    reward_id = 0
    evalMeanReward = 0
    counter_store_model = args.c  # New version of the model, incremented along the episodes of this batch

    rewards_test = np.zeros(EVALUATION_EPISODES*3)
    error_links = np.zeros(EVALUATION_EPISODES*3)
    max_link_uti = np.zeros(EVALUATION_EPISODES*3) # Stores the utilization of the most loaded link for each validation matrix of each topology (i.e. after rerouting all the critical demands)
    min_link_uti = np.zeros(EVALUATION_EPISODES*3)
    uti_std = np.zeros(EVALUATION_EPISODES*3)

    training_tm_ids = set(range(100))

    for iters in range(args.e): # Train args.e episodes (process a batch of episodes)
        current_ep = args.i + iters
        current_lr = args.lr * (DECAY_RATE ** (current_ep / DECAY_STEPS))
        if current_lr < 1e-5:
            current_lr = 1e-5
        agent.optimizer.learning_rate.assign(current_lr)
        
        start_time_episode = time.time()
        states = []
        critic_features = []
        tensors = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []

        print("MIDDLEPOINT ROUTING(3 TOP Topologies Enero "+experiment_letter+") PPO EPISODE: ", args.i+iters)
        number_samples_reached = False

        '''
        Then: a trajectory of length num_samples1 + num_samples2 + num_samples3 is followed (the number refers to the topology), such that for each topology:
            * The traffic matrix used in the current episode is selected
            * All the critical demands of that traffic matrix are rerouted. This is repeated (on the same traffic matrix) a given number of times, until the number of applied actions is num_samples<number>

        '''

        # The length of the trajectory followed to collect experience = 'num_samples_top1'
        # IMPORTANT!! --> The agent is applied a given number of times on different traffic matrices
        print("Starting training with the 1st topology...")
        while not number_samples_reached:
            tm_id = random.sample(list(training_tm_ids), 1)[0]
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 1
            ######

            # The following 'while' loop runs as many iterations as critical demands selected for rerouting in that traffic matrix
            obs, info = env_training1.unwrapped.reset(options={"tm_id": tm_id})
            demand = info.get("demand", info.get("bw"))
            source = info['src']
            destination = info['dst']
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)

                # Predict probabilities over middlepoints
                '''
                The agent receives:
                    - Environment (of the first topology)
                    - Source and destination of the demand to reroute
                The agent is an MPNN applied on as many graphs as possible middlepoints,
                such that each graph has the path of a given middlepoint marked.

                Output: probabilities of each action (=middlepoint) and the information of each action (graph with the middlepoint marked)
                '''
                action_dist, tensor = agent.pred_action_distrib_sp(env_training1.unwrapped, source, destination)
                # Before finally applying a middlepoint to the graph in the current state and moving to the next state, which would make us lose
                # the initial state of the graph, the following function stores the features of the initial state of the graph.
                features = agent.critic_get_graph_features(env_training1.unwrapped)

                q_value = agent.critic_value(features)  # [v2-4]

                # An action is taken according to its probabilities. This is done only during training, to explore more and not always follow the apparently "best path"
                action = np.random.choice(len(action_dist), p=action_dist)
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                '''
                The chosen action (index of the middlepoint) is applied to the environment
                'done' is a boolean that is activated when all the selected critical demands have been routed
                After each step, as long as there are critical demands not yet rerouted, the environment sets the next critical demand in 'patMaxBandwth'
                '''
                obs, reward, terminated, truncated, info = env_training1.unwrapped.step(action, demand, source, destination)
                new_demand = info.get("bw", info.get("demand"))
                new_source = info["src"]
                new_destination = info["dst"]
                done = terminated or truncated
                mask = not done

                states.append((env_training1.unwrapped.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                # When the agent has visited a number of states of the environment = num_samples_top1 (number of decisions taken), both loops are exited (number_samples_reached is activated),
                # and we move to the next topology.
                if len(states) == num_samples_top1:
                    number_samples_reached = True
                    break

                # Once all the selected critical demands of this traffic matrix have been rerouted, the inner 'while' is exited and another iteration of the outer 'while' is run, restoring the OSPF routing
                # of the demands of the same traffic matrix of topology 1
                if done:
                    break

        number_samples_reached = False
        print("Starting training with the 2nd topology...")
        while not number_samples_reached:
            tm_id = random.sample(list(training_tm_ids), 1)[0]
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 2
            ######

            obs, info = env_training2.unwrapped.reset(options={"tm_id": tm_id})
            demand = info.get("demand", info.get("bw"))
            source = info['src']
            destination = info['dst']
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)
                # Predict probabilities over middlepoints
                action_dist, tensor = agent.pred_action_distrib_sp(env_training2.unwrapped, source, destination)
                features = agent.critic_get_graph_features(env_training2.unwrapped)

                q_value = agent.critic_value(features)  # [v2-4]

                action = np.random.choice(len(action_dist), p=action_dist)
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                obs, reward, terminated, truncated, info = env_training2.unwrapped.step(action, demand, source, destination)
                new_demand = info.get("bw", info.get("demand"))
                new_source = info["src"]
                new_destination = info["dst"]
                done = terminated or truncated
                mask = not done

                states.append((env_training2.unwrapped.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                # If we have enough samples
                if len(states) == num_samples_top1+num_samples_top2:
                    number_samples_reached = True
                    break

                if done:
                    break

        number_samples_reached = False
        print("Starting training with the 3rd topology...")
        while not number_samples_reached:
            tm_id = random.sample(list(training_tm_ids), 1)[0]
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 3
            ######

            obs, info = env_training3.unwrapped.reset(options={"tm_id": tm_id})
            demand = info.get("demand", info.get("bw"))
            source = info['src']
            destination = info['dst']
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)
                # Predict probabilities over middlepoints
                action_dist, tensor = agent.pred_action_distrib_sp(env_training3.unwrapped, source, destination)
                features = agent.critic_get_graph_features(env_training3.unwrapped)

                q_value = agent.critic_value(features)  # [v2-4]

                action = np.random.choice(len(action_dist), p=action_dist)
                # One-hot vector of length 'number of possible middlepoints for the current demand', activating only the middlepoint finally used.
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                obs, reward, terminated, truncated, info = env_training3.unwrapped.step(action, demand, source, destination)
                new_demand = info.get("bw", info.get("demand"))
                new_source = info["src"]
                new_destination = info["dst"]
                done = terminated or truncated
                mask = not done

                states.append((env_training3.unwrapped.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                # If we have enough samples
                if len(states) == num_samples_top1+num_samples_top2+num_samples_top3:
                    number_samples_reached = True
                    break

                if done:
                    break

        features = agent.critic_get_graph_features(env_training3.unwrapped)
        # The critic receives the information of the initial graph (without a candidate middlepoint) + sender links + receiver links
        # This is equivalent to asking the critic how good the future result of applying the current policy from the initial state of the graph is
        q_value = agent.critic_value(features)  # [v2-4]       
        values.append(q_value)

        # After applying the agent several times on each topology (with different training traffic matrices):
        # Updated values for each time step of the trajectory, and the advantage obtained in each time step
        returns, advantages = get_advantages(values, masks, rewards)
        '''
        The agent is updated at the end of the iteration of the episode (after being applied on each of the 3 topologies, with different TMs on each topology)
        Parameters:
            * actions --> for each state: one-hot vector (one component per middlepoint that can be chosen for the demand) activating the applied action/middlepoint
            * action_probs --> for each state along the trajectory --> probability distribution of the different actions
            * tensors --> for each state: the graphs of the different candidate middlepoints
            * critic_features --> features of the graph in each initial state, before applying the selected middlepoint
            * returns --> updated critic values
            * advantages --> advantage obtained in each state

        Summary: for each state of the trajectory followed to collect experience, before updating anything, we collect:
        the selected middlepoint among the possible ones, the probability of each middlepoint, the initial state of the graph with each of the candidate paths marked,
        the initial state of the graph (without any candidate middlepoint), the updated critic value (with the advantage) and the advantage obtained
        '''

        print("\n ---- Updating the actor/critic parameters ----")
        actor_loss, critic_loss = agent.ppo_update(actions, actions_probs, tensors, critic_features, returns, advantages)
        # The mean loss of the actor/critic in this episode is stored here
        fileLogs.write("a," + str(actor_loss.numpy()) + ",\n")
        fileLogs.write("c," + str(critic_loss.numpy()) + ",\n")
        fileLogs.flush()

        '''
        After updating the agent at the end of the episode, the agent is evaluated on 20 of the 50 traffic matrices used for validation.
        These validation traffic matrices are assumed to be uniformly distributed.
        '''

        print("\nValidating...")
        # Evaluate on FIRST TOPOLOGY, here 20 episodes (the 20 TMs selected from the validation set)
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            obs, info = env_eval1.unwrapped.reset(options={"tm_id": tm_id})
            demand = info.get("demand", info.get("bw"))
            source = info['src']
            destination = info['dst']
            done = False
            rewardAddTest = 0
            # The agent is applied as many times as critical demands were selected in this TM
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval1.unwrapped, source, destination)
                # Now, instead of selecting the middlepoint probabilistically, the one with the
                # highest probability is selected (deterministic selection)
                action = np.argmax(action_dist)
                obs, reward, terminated, truncated, info_step = env_eval1.unwrapped.step(action, demand, source, destination)
                done = terminated or truncated
                error_eval_links = info_step.get("error_evaluation", 0.0)
                maxLinkUti = info_step["edgeMaxUti"]
                minLinkUti = info_step.get("edgeMinUti", 0.0)
                utiStd = info_step.get("std_norm", info_step["std"])  # [v2-7]
                demand = info_step.get("bw", info_step.get("demand"))
                source = info_step["src"]
                destination = info_step["dst"]

                rewardAddTest += reward
                if done:
                    break
            # The reward of episode 'eps' is the sum of the rewards obtained after rerouting each of the selected critical demands
            # This is the actual cumulative future reward obtained after following a trajectory of a given length.
            rewards_test[eps] = rewardAddTest
            error_links[eps] = error_eval_links
            max_link_uti[eps] = maxLinkUti[2] # Utilization of the most loaded link after rerouting all the critical demands of this traffic matrix (metric)
            min_link_uti[eps] = minLinkUti
            uti_std[eps] = utiStd
        
        # Evaluate on SECOND TOPOLOGy
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            posi = EVALUATION_EPISODES+eps
            obs, info = env_eval2.unwrapped.reset(options={"tm_id": tm_id})
            demand = info.get("demand", info.get("bw"))
            source = info['src']
            destination = info['dst']
            done = False
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval2.unwrapped, source, destination)
                
                action = np.argmax(action_dist)
                obs, reward, terminated, truncated, info_step = env_eval2.unwrapped.step(action, demand, source, destination)
                done = terminated or truncated
                error_eval_links = info_step.get("error_evaluation", 0.0)
                maxLinkUti = info_step["edgeMaxUti"]
                minLinkUti = info_step.get("edgeMinUti", 0.0)
                utiStd = info_step.get("std_norm", info_step["std"])  # [v2-7]
                demand = info_step.get("bw", info_step.get("demand"))  # [v2-6]
                source = info_step["src"]
                destination = info_step["dst"]

                rewardAddTest += reward
                if done:
                    break
            rewards_test[posi] = rewardAddTest
            error_links[posi] = error_eval_links
            max_link_uti[posi] = maxLinkUti[2]
            min_link_uti[posi] = minLinkUti
            uti_std[posi] = utiStd

        # Evaluate on THIRD TOPOLOGY
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            posi = EVALUATION_EPISODES*2+eps
            obs, info = env_eval3.unwrapped.reset(options={"tm_id": tm_id})
            demand = info.get("demand", info.get("bw"))
            source = info['src']
            destination = info['dst']
            done = False
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval3.unwrapped, source, destination)
                
                action = np.argmax(action_dist)
                obs, reward, terminated, truncated, info_step = env_eval3.unwrapped.step(action, demand, source, destination)
                done = terminated or truncated
                error_eval_links = info_step.get("error_evaluation", 0.0)
                maxLinkUti = info_step["edgeMaxUti"]
                minLinkUti = info_step.get("edgeMinUti", 0.0)
                utiStd = info_step.get("std_norm", info_step["std"])  # [v2-7]
                demand = info_step.get("bw", info_step.get("demand"))  # [v2-6]
                source = info_step["src"]
                destination = info_step["dst"]

                rewardAddTest += reward
                if done:
                    break
            # Rerouting finished
            rewards_test[posi] = rewardAddTest
            error_links[posi] = error_eval_links
            max_link_uti[posi] = maxLinkUti[2] # Utilization of the most loaded link after rerouting all the selected critical demands of this matrix
            min_link_uti[posi] = minLinkUti
            uti_std[posi] = utiStd

        evalMeanReward = np.mean(rewards_test)
        fileLogs.write(";," + str(np.mean(uti_std)) + ",\n")
        fileLogs.write("+," + str(np.mean(error_links)) + ",\n")
        fileLogs.write("<," + str(np.amax(max_link_uti)) + ",\n")
        fileLogs.write("MLUmean," + str(np.mean(max_link_uti)) + ",\n")  # [v2-7]
        fileLogs.write(">," + str(np.amax(min_link_uti)) + ",\n")
        fileLogs.write("ENTR," + str(ENTROPY_BETA) + ",\n")
        #fileLogs.write("-," + str(agent.epsilon) + ",\n")
        fileLogs.write("REW," + str(evalMeanReward) + ",\n")
        fileLogs.write("lr," + str(current_lr) + ",\n")
  
        # If at the end of the training episode, after validating, the mean reward is higher than the best one so far,
        # the log file records that a new model with a better reward has been found.
        if evalMeanReward>max_reward:
            max_reward = evalMeanReward
            reward_id = counter_store_model
            # The reward of the new best model and its id are written to the log file, so that the new best model can be taken at any time
            fileLogs.write("MAX REWD: " + str(max_reward) + " REWD_ID: " + str(reward_id) +",\n")
        
        fileLogs.flush()
        
        # Store trained model
        # Storing the model and the tape.gradient make the memory increase
        # .save stores both the internal variables of the Checkpoint (including save_counter) and the trackable objects
        checkpoint_actor.save(checkpoint_prefix+'_ACT')
        checkpoint_critic.save(checkpoint_prefix+'_CRT')
        counter_store_model = counter_store_model + 1
        K.clear_session()
        
        end_time_episode = time.time()
        duration_total = end_time_episode - start_time_episode
        print(f"\n[INFO] Total time of episode {args.i + iters}: {duration_total/60:.2f} min ({duration_total:.2f} s)\n")
        
        #gc.collect()
        # ---------------- END OF EPISODE ---------

    '''
    At the end of the batch of episodes, the temporary file is opened in binary write mode, so that each time it is opened in this mode, the information
    it contained is overwritten with the new one

    This temporary file is used during the training of the model to store at the end of each batch:
        * Maximum reward obtained so far, considering all the past episodes
        * Last learning rate used (in the last episode before this new one)
    '''
    f = open(config.CODE_DIR + "/tmp/" + differentiation_str + "tmp.pckl", 'wb')
    # Initialize current_lr if the loop was not executed
    if 'current_lr' not in locals():
        current_lr = hparams['learning_rate']
    pickle.dump((max_reward, current_lr), f)
    f.close()

