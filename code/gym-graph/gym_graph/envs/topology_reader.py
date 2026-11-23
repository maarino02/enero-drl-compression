import numpy as np
import networkx as nx


class TopologyReader:
    """
    Reads a topology (.graph file) and its traffic matrices (.demands files) in the
    format of the ENERO datasets.

    After construction it exposes:
        * net_size     -> number of nodes
        * Gbase        -> networkx MultiDiGraph with one edge per directed link
        * links_bw     -> links_bw[src][dst] = capacity of the directed link src->dst
        * links_weight -> links_weight[src][dst] = weight of the directed link src->dst
                          (all links have the same weight)
    """

    net_size = 0
    links_bw = None
    links_weight = None
    Gbase = None

    def __init__(self, graph_file):
        self.graph_file = graph_file
        self.Gbase = nx.MultiDiGraph()
        self.process_graph_file()

    def process_graph_file(self):
        with open(self.graph_file) as fd:
            line = fd.readline()
            camps = line.split(" ")
            self.net_size = int(camps[1])  # Number of nodes of the network
            # Remove : label x y
            line = fd.readline()

            # Skip the node lines (one per node, e.g. "N0 x y")
            for i in range(self.net_size):
                line = fd.readline()

            self.links_bw = []
            self.links_weight = []
            for i in range(self.net_size):
                self.links_bw.append({})
                self.links_weight.append({})
            for line in fd:
                if (not line.startswith("Link_") and not line.startswith("edge_")):
                    continue
                # Link line: 'label src dest weight bw delay'. Only src, dest, weight and bw are used
                camps = line.split(" ")
                src = int(camps[1])  # Index of the source node of the directed link
                dst = int(camps[2])  # Index of the destination node of the directed link
                weight = int(camps[3])  # Weight of the directed link
                bw = float(camps[4])  # Capacity of the link
                # The directed graph is built directly from its links; the end nodes are created
                # automatically if they do not exist yet
                self.Gbase.add_edge(src, dst)
                self.links_bw[src][dst] = bw
                self.links_weight[src][dst] = weight

    def _get_traffic_matrix(self, traffic_file):
        tm = np.zeros((self.net_size, self.net_size))
        with open(traffic_file) as fd:
            fd.readline()
            fd.readline()
            for line in fd:
                camps = line.split(" ")
                # We force that the bws are integers
                tm[int(camps[1]), int(camps[2])] = np.floor(float(camps[3]))
        return (tm)
