# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

# This class models the actor
class myModel(tf.keras.Model):
    '''
    Defines the architecture of the actor without specifying the input dimension of each layer (that is done in build())
        * hparams: dictionary with the configuration parameters
        * hidden_init_actor and kernel_init_actor: initializers of the parameters of the actor
    '''
    def __init__(self, hparams, hidden_init_actor, kernel_init_actor):
        super(myModel, self).__init__()
        self.hparams = hparams

        # --- First part of the model: MESSAGE PASSING
        self.Message = tf.keras.models.Sequential()
        # The output dimension of this layer is 'link_state_dim' because it returns the message that a sender link sends to a
        # receiver link. The message is a vector of 'link_state_dim' components, so that it can then be combined with the hidden
        # state of the receiver link when updating its state.
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            kernel_initializer=hidden_init_actor,
                                            activation=tf.nn.selu, name="FirstLayer"))
        # Second part: update the hidden state of each link from the received messages. This layer returns the updated hidden
        # state of the receiver link. Its size is the size of the hidden state 'h'
        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ActorLayerNorm")

        # The information accumulated in the graph is interpreted here (readout), with n dense layers:
        self.Readout = tf.keras.models.Sequential()
        activation_fn = tf.keras.activations.get(self.hparams['activation_fn'])
        for i in range(self.hparams['num_readout_layers']):
            self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                                activation=activation_fn,
                                                kernel_initializer=hidden_init_actor,
                                                kernel_regularizer=regularizers.l2(self.hparams['l2']),
                                                name=f"Readout_{i+1}"))
            if self.hparams['dropout_rate'] > 0:
                self.Readout.add(keras.layers.Dropout(rate=self.hparams['dropout_rate']))
                
        # The last dense layer has a single neuron and returns the value that represents each candidate graph
        self.Readout.add(keras.layers.Dense(1, kernel_initializer=kernel_init_actor, name="Readout_Final"))
        '''
        Given a state of the environment (and the demand to route), the actor is applied as many times as candidate middlepoints there are.
        For each candidate (modelled as the initial graph with the path of that middlepoint marked), the agent represents the graph with a value.
        '''

    '''
    '__init__' defines the architecture of the model, giving the output dimension of each layer.
    'build' specifies the input dimension of each of the phases
    '''
    def build(self, input_shape=None):
        # Create the weights of the layer
        # The input dimension is given for the message passing, update and readout phases
        
        '''
        Message passing works on the hidden states of both the receiver link and the sender links.
        The input is therefore a 2-D tensor:
            * First dimension: 'None', i.e. the number of rows is not fixed (one row per sender-receiver pair).
            * Second dimension: 'link_state_dim' * 2, the concatenation of the hidden state of the sender and of the receiver link.

        That is, the input is a matrix with one row per sender link, and each row contains the hidden states of the sender and
        the receiver links, which is what is needed to model the message that is sent.
        '''
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))

        '''
        Unlike a GRU, which processes a whole sequence of tokens, a GRUCell processes a single token of the sequence.
            * Here each link receives a sequence of a single token: the aggregation of the messages received from its neighbours.
            * The initial hidden state is the state vector of the link (before the update).

            - The output is a tensor of shape [None, 'link_state_dim'], with the updated state vector of each element of the batch (each link)
        '''
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))
        self.LayerNorm.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']]))

        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    #@tf.function
    '''
    Each time the supergraph with the graphs of the candidate middlepoints is processed with the actor (GNN), the following
    process is run (FORWARD PROPAGATION). The actor receives:
    self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'],tensor['num_edges'])

    The process is:
    During T iterations:
        1. The messages that each link receives from its neighbours are generated and added up
        2. Each link is updated with the GRUCell, which works with a sequence of length 1 (the aggregated message) and the state
        of the link as initial hidden state, giving as output the new state of the link
        3. Step 1 is repeated, generating the new messages that each link receives from its neighbours with their new states
    '''
    # training= ... --> whether the model is used for training or for inference, so that the Dropout layers are used or not
    def call(self, link_state, states_graph_ids, states_first, states_second, sates_num_edges, training=False):

        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main edges with the neighbours

            # Hidden states of the links that send messages, for each graph with a candidate middlepoint.
            # This works because the input tensor is preprocessed, shifting the positions of the senders so that we can work on the global 'link_state'
            mainEdges = tf.gather(link_state, states_first)
            # Hidden states of the receiver links
            neighEdges = tf.gather(link_state, states_second)

            # To generate the message, the hidden states of the sender and of the receiver are concatenated (one row per pair)
            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            ### 1.a Message passing for link with all it's neighbours
            outputs = self.Message(edgesConcat)
            # Messages generated for each sender-receiver pair: tensor of shape (sender-receiver pairs, message)

            '''
            Now the messages received by each link are added up...
            The messages (vectors) received by the same receiver link are summed, i.e. all the rows of 'outputs' that are messages sent
            to the same receiver link. This is a segmented sum, adding the rows that belong to the same segment (receiver link)
            '''
            ### 1.b Sum of output values according to link id index
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=sates_num_edges)


            ### 2. Update for each link
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            '''
            The update layer (GRUCell) receives:
                * 2-D tensor with the sum of the messages received by each link, for each graph with a candidate middlepoint
                * List with a single element: global matrix with the initial hidden state of each link of each graph
            Output:
                * outputs: output of the GRUCell, which here is the new hidden state 'h'
                * link_state_list: list with a single element (tensor) with the new hidden state 'h' (new state) of each link
            '''
            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            # New hidden state of each link of the graph
            link_state = links_state_list[0]
            link_state = self.LayerNorm(link_state)

        # After message passing, the state vectors of the links of each candidate graph are added up,
        # and from that sum we try to obtain a score of the candidate graph
        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)
        
        # Each candidate graph (middlepoint) is represented with a single numerical value
        r = self.Readout(edges_combi_outputs,training=training)
        return r
