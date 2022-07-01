import dgl
import matplotlib
import torch
import pandas as pd
import networkx as nx
def build_karate_club_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = pd.read_csv('/testdata')
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    return g

G = build_karate_club_graph()
G.ndata['feat'] = torch.eye(34)



