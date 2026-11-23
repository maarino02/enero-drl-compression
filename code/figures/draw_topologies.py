"""
figures/draw_topologies.py - Draws the structure of a topology of the ENERO datasets
====================================================================================
Output: figures/output/topologies/<Topology>/<Topology>_estructura.png

Usage (from code/):  python -m figures.draw_topologies -g EliBackbone
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
import argparse

import config

def parse_graph_file(filepath):
    G = nx.Graph()
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    mode = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('NODES'):
            mode = 'NODES'
            continue
        elif line.startswith('EDGES'):
            mode = 'EDGES'
            continue
            
        if mode == 'NODES':
            # Skip header line like 'label x y'
            if line.startswith('label') or line.startswith('id'):
                continue
            parts = line.split()
            # Depending on format, sometimes it's "N0 0.0 0.0", sometimes just "0 0.0 0.0"
            node_id = parts[0].replace('N', '')
            G.add_node(int(node_id))
            
        elif mode == 'EDGES':
            if line.startswith('label') or line.startswith('src'):
                continue
            parts = line.split()
            try:
                # If the first column is a text (e.g. Link_0), src and dst are shifted
                if not parts[0].replace('N', '').isnumeric():
                    src = int(parts[1].replace('N', ''))
                    dst = int(parts[2].replace('N', ''))
                else:
                    src = int(parts[0].replace('N', ''))
                    dst = int(parts[1].replace('N', ''))
                G.add_edge(src, dst)
            except:
                pass
                
    return G

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw the structure of a ZOO topology')
    parser.add_argument('-g', help='Name of the topology (e.g. Aarnet)', type=str, required=True)
    parser.add_argument('-f', help='Datasets folder', type=str, default=config.ZOO_TOPOLOGIES_DIR)
    
    args = parser.parse_args()
    topology = args.g
    
    filepath = os.path.join(args.f, topology, f"{topology}.graph")
    
    if not os.path.exists(filepath):
        print(f"Error: file {filepath} not found")
        exit(1)
        
    G = parse_graph_file(filepath)
    
    # Compute the density to show it
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)
    
    # Create the folder if it does not exist
    output_dir = os.path.join(config.FIGURES_OUTPUT_DIR, "topologies", topology)
    os.makedirs(output_dir, exist_ok=True)
    
    # Draw
    plt.figure(figsize=(10, 8))
    plt.title(f"Estructura Física - {topology}\nNodos: {nodes} | Enlaces: {edges} | Densidad: {density:.3f}", 
              fontsize=16, fontweight='bold', pad=20)
    
    # The 'spring' layout separates the nodes by repelling them so that the structure is clearly visible
    pos = nx.spring_layout(G, seed=42) 
    
    # Draw nodes, edges and labels
    nx.draw_networkx_nodes(G, pos, node_color='#3b82f6', node_size=500, edgecolors='white', linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{topology}_estructura.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Topology {topology} drawn successfully!")
    print(f"Image saved in: {output_path}")
