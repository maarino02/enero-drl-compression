import os
import networkx as nx
import matplotlib.pyplot as plt
import argparse

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
                # Si la primera columna es un texto (ej: Link_0), src y dst se desplazan
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
    parser = argparse.ArgumentParser(description='Dibujar estructura de topología ZOO')
    parser.add_argument('-g', help='Nombre de la topología (ej: Aarnet)', type=str, required=True)
    parser.add_argument('-f', help='Carpeta de datasets', type=str, default='../datasets/zoo/')
    
    args = parser.parse_args()
    topology = args.g
    
    filepath = os.path.join(args.f, topology, f"{topology}.graph")
    
    if not os.path.exists(filepath):
        print(f"Error: No se ha encontrado el archivo {filepath}")
        exit(1)
        
    G = parse_graph_file(filepath)
    
    # Calcular densidad para mostrarla
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)
    
    # Crear carpeta si no existe
    output_dir = f"../resultados/topologias/{topology}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dibujar
    plt.figure(figsize=(10, 8))
    plt.title(f"Estructura Física - {topology}\nNodos: {nodes} | Enlaces: {edges} | Densidad: {density:.3f}", 
              fontsize=16, fontweight='bold', pad=20)
    
    # El layout 'spring' separa los nodos repeliéndolos para que se vea bien la estructura
    pos = nx.spring_layout(G, seed=42) 
    
    # Dibujar nodos, bordes y etiquetas
    nx.draw_networkx_nodes(G, pos, node_color='#3b82f6', node_size=500, edgecolors='white', linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{topology}_estructura.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Topología {topology} dibujada con éxito!")
    print(f"Imagen guardada en: {output_path}")
