import sys
import time
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import gzip
import os

from sklearn.linear_model import LinearRegression

import networkx as nx

def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50, decimals=0):
    graph = [[np.inf for i in range(0,n_nodes)] for j in range(0,n_nodes)]
    for i in range(0,n_nodes):
        for j in range(0,n_nodes):
            if i!=j and random.random()<=sparse_factor:
                graph[i][j] = random.randint(1,max_weight)
    return graph


def cuenta_ramas(m_g):
    l=0
    for i in range(0, len(m_g)):
        for j in range(0, len(m_g)):
            if m_g[i][j]<float('inf'):
                l += 1
    return l


def calculate_sparse_factor(g):
    return (cuenta_ramas(g))/(len(g)*(len(g)-1.0))


def check_sparse_factor(n_grafos, n_nodes, sparse_factor):
    values = [calculate_sparse_factor(rand_matr_pos_graph(n_nodes,sparse_factor)) for i in range(0, n_grafos)]
    return sum(values)/(n_grafos)



def m_g_2_d_g(m_g):
    graph = {}
    for i in range(0,len(m_g)):
        graph[i] = {}
        for j in range(0, len(m_g)):
            if m_g[i][j] < np.inf:
                graph[i][j] = m_g[i][j]
    return graph


def d_g_2_m_g(d_g):
    graph = [[np.inf for i in range(0,len(d_g))] for j in range(0,len(d_g))]
    for v1 in d_g:
        for v2 in d_g[v1]:
            graph[v1][v2] = d_g[v1][v2]
    return graph



def save_object(obj, f_name='obj.pklz', save_path='.'):
    complete_name = os.path.join(save_path, f_name)
    with gzip.open(complete_name) as f:
        f.write(pickle.dump(obj))
        
        
def read_object(f_name, save_path='.'):
    complete_name = os.path.join(save_path, f_name)
    obj = None
    with gzip.open(complete_name) as f:
        obj = pickle.load(f.read())
    return obj


# Trivial Graph Format (TGF)
def d_g_2_TGF(d_g, f_name):
    out = ''
    for v1 in d_g:
        out += str(v1)+'\n'
    out += '#\n'
    for v1 in d_g:
        for v2 in d_g[v1]:
            out += str(v1)+' '+str(v2)+' '+str(d_g[v1][v2])+'\n'
    with open(f_name,'w') as f:
        f.write(out)        
        

def TGF_2_d_g(f_name):
    with open(f_name) as f:
        tgf = f.read()
    nodes, edges = tgf.split('#\n')
    nodes = nodes.split('\n')[:-1] # Last is empty
    edges = list(map(lambda x: x.split(' '), edges.split('\n')[:-1]))
    graph = {}
    for node in nodes:
        graph[int(node)] = {}
    for v1, v2, weight in edges:
        graph[int(v1)][int(v2)] = float(weight)
    return graph



# DIJKSTRA!
from queue import PriorityQueue

def dijkstra_d(d_g, u):
    dist = {}
    prev = {}
    visited = {}
    for ele in d_g:
        dist[ele] = np.inf
        prev[ele] = -1
        visited[ele] = False
    q = PriorityQueue()
    dist[u] = 0
    q.put((0, u))
    while not q.empty():
        w, curr = q.get()
        if not visited[curr]:
            visited[curr] = True
            for node in d_g[curr]:
                if d_g[curr][node]+w < dist[node]:
                    dist[node] = d_g[curr][node]+w
                    prev[node] = curr
                    q.put((dist[node], node))
    return dist, prev
    
    
def dijkstra_m(m_g, u):
    dist = {}
    prev = {}
    visited = {}
    for ele in range(len(m_g)):
        dist[ele] = np.inf
        prev[ele] = -1
        visited[ele] = False
    q = PriorityQueue()
    dist[u] = 0
    q.put((0, u))
    while not q.empty():
        w, curr = q.get()
        if not visited[curr]:
            visited[curr] = True
            for node in range(len(m_g[curr])):
                if m_g[curr][node]+w < dist[node]:
                    dist[node] = m_g[curr][node]+w
                    prev[node] = curr
                    q.put((dist[node], node))
    return dist, prev


def min_path(d_prev, v):
    if d_prev[v]<0:
        return [] # There is no path
    path = [v]
    while d_prev[v]>=0:
        v = d_prev[v]
        path.append(v)
    path.reverse()
    return path    
    
def min_paths(d_prev):
    d_path = {}
    for ele in d_prev:
        d_path[ele] = min_path(d_prev, ele)
    return d_path



from time import time


def time_dijkstra(n_graphs, n_nodes_ini, n_nodes_fin, step, generate, dijks, sparse_factor=.25):
    times = [] # Returning list
    for nodes in range(n_nodes_ini, n_nodes_fin+1, step):
        meanTime = 0
        for _ in range(n_graphs):
            graph = generate(nodes, sparse_factor) 
            time_ini = time()
            for i in range(nodes):
                dijks(graph,i)
            meanTime += (time()-time_ini)
        times.append(meanTime/(n_graphs*nodes))
    return times
    
    
def time_dijkstra_m(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    return time_dijkstra(n_graphs, n_nodes_ini, n_nodes_fin, step, rand_matr_pos_graph, dijkstra_m, sparse_factor=sparse_factor)
    
    
def time_dijkstra_d(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    f = lambda x,y: m_g_2_d_g(rand_matr_pos_graph(x,y))
    return time_dijkstra(n_graphs, n_nodes_ini, n_nodes_fin, step, f, dijkstra_d, sparse_factor=sparse_factor)



def fit_plot(l, func_2_fit, size_ini, size_fin, step, label=None):
    l_func_values =[i*func_2_fit(i) for i in range(size_ini, size_fin+1, step)]
    
    lr_m = LinearRegression()
    X = np.array(l_func_values).reshape( len(l_func_values), -1 )
    lr_m.fit(X, l)
    y_pred = lr_m.predict(X)
    if label:
        plt.plot(l, '*', y_pred, '-', label=label)
        return
    plt.plot(l, '*', y_pred, '-')
    
def n2_log_n(n):
    return n**2. * np.log(n)



def edges(d_g):
    e = []
    for u in d_g:
        for v in d_g[u]:
            e.append((u,v,d_g[u][v]))
    return e


def d_g_2_nx_g(d_g):
    g = nx.DiGraph()
    g.add_weighted_edges_from(edges(d_g))
    return g

def nx_g_2_d_g(nx_g):
    d_g = {}
    for u in nx_g:
        d_g[u] = {}
        for v in nx_g[u]:
            d_g[u][v] = nx_g[u][v]['weight']
            
def time_dijkstra_nx(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):    
    f = lambda x,y: d_g_2_nx_g(m_g_2_d_g(rand_matr_pos_graph(x,y)))
    return time_dijkstra(n_graphs, n_nodes_ini, n_nodes_fin, step, f, nx.single_source_dijkstra, sparse_factor=sparse_factor)

     
def print_m_g(m_g):
    print("graph_from_matrix:\n")
    n_v = m_g.shape[0]
    for u in range(n_v):
        for v in range(n_v):
            if v != u and m_g[u, v] != np.inf:
                print("(", u, v, ")", m_g[u, v])  
def print_d_g(d_g):
    print("\ngraph_from_dict:\n")
    for u in d_g.keys():
        for v in d_g[u].keys():
            print("(", u, v, ")", d_g[u][v])
           