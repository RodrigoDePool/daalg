import grafos05 as gr
import networkx as nx
# 1 for matrix 2 for dg 3 for ng

def compare_dists(d1,d2):
    for ele in d1:
        if d2[ele] != d1[ele]:
            raise Exception()

def compare_paths(p1,p2):
    for ele in p1:
        if p1[ele] != p2[ele]:
            raise Exception()


def test_dijkstra(g1, dijks1, g2, dijks2):
    d1, p1 = dijks1(g1, 0)
    d2, p2 = dijks2(g2, 0)
    if dijks1 != nx.single_source_dijkstra:
        p1 = gr.min_paths(p1)
    if dijks2 != nx.single_source_dijkstra:
        p2 = gr.min_paths(p2)
    compare_dists(d1, d2)
    compare_paths(p1, p2)

def full_dijkstra(g):
    d = {1: gr.dijkstra_m, 2: gr.dijkstra_d, 3: nx.single_source_dijkstra}
    for i in range(1,len(g)):
        test_dijkstra(g[i-1][0], d[g[i-1][1]], g[i][0], d[g[i][1]])
    print('TODO FUNCIONA')

def test(node, sparse_factor):
    graphs = []
    g = gr.rand_matr_pos_graph(nodes,sparse_factor)
    graphs.append((g, 1))
    g = gr.m_g_2_d_g(g)
    graphs.append((g,2))
    g = gr.d_g_2_m_g(g)
    graphs.append((g,1))
    g = gr.m_g_2_d_g(g)
    graphs.append((g,2))
    g = gr.d_g_2_nx_g(g)
    graphs.append((g,3))
    g = gr.nx_g_2_d_g(g)
    graphs.append((g,2))
    gr.save_object(g)
    g = gr.read_object('obj.pklz')
    graphs.append((g,2))
    g = gr.d_g_2_m_g(g)
    graphs.append((g,1))
    gr.save_object(g)
    g = gr.read_object('obj.pklz')
    graphs.append((g,1))
    g = gr.m_g_2_d_g(g)
    graphs.append((g,2))
    gr.d_g_2_TGF(g,'a.tgf')
    g = gr.TGF_2_d_g('a.tgf')
    graphs.append((g,2))
    full_dijkstra(graphs)


for nodes in range(10,30,5):
    test(nodes,0.8)
    test(nodes,0.9)


