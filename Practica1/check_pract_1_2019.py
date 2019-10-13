#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string, random
import numpy as np
import pandas as pd
import sys
import os
import argparse

import networkx as nx

from sklearn.linear_model import LinearRegression

import grafos_2019 as gr

#def fit_plot(l, func_2_fit, size_ini, size_fin, step):
#    l_func_values =[i*func_2_fit(i) for i in range(size_ini, size_fin+1, step)]
#    
#    lr_m = LinearRegression()
#    X = np.array(l_func_values).reshape( len(l_func_values), -1 )
#    lr_m.fit(X, l)
#    y_pred = lr_m.predict(X)
#    
#    plt.plot(l, '*', y_pred, '-')
#
#def n2_log_n(n):
#    return n**2. * np.log(n)
#
#l_values =[i*n2_log_n(i) *(1 + 0.25* np.random.rand()) for i in range(10, 500+1, 10)]
#fit_plot(l_values, n2_log_n, 10, 500, 10)

  
####################################### main
def main(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor):
    """Prueba las funciones de qs.py.
    
    Args: n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor
    """
    
    #if len(args) != 5:
    #    print ("args: n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor")
    #    sys.exit(0)  
    #n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor = int(args[0]), int(args[1]), int(args[2]), int(args[3]), float(args[4])
    
    np.set_printoptions(precision=3)
    
    print("\ncheck basic graph functions ....................")
    print("\ncomprobamos la generación de grafos aleatorios ..........")
    est_sp_f = gr.check_sparse_factor(n_graphs, n_nodes_ini, sparse_factor)
    print("\n\testimated sp_f %5.2f sobre %d graphs with %d n_nodes and spars. fact. %5.2f" %  (est_sp_f, n_graphs, n_nodes_ini, sparse_factor) )
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\ncomprobamos las funciones de conversión de grafo a matriz y vice versa ..........")
    m_g = gr.rand_matr_pos_graph(n_nodes=5, sparse_factor=0.5, max_weight=10.)    
    print("\nmatriz del grafo generado\n", m_g)
    
    d_g = gr.m_g_2_d_g(m_g)
    print("\nfrom m to d")
    gr.print_d_g(d_g)
    
    m_g = gr.d_g_2_m_g(d_g)
    print("\nfrom d to m\n", m_g)
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\tgeneramos_grafo_aleatorio ..........")
    m_g = gr.rand_matr_pos_graph(n_nodes=5, sparse_factor=0.5, max_weight=10.)    
    d_g = gr.m_g_2_d_g(m_g)
    gr.print_d_g(d_g)
    
    print("\ncomprobamos las funciones de guardar grafos como TFG ..........")
    
    print("\nguardamos como TFG y mostramos archivo..........\n")
    f_name = 'my_graph.tfg'
    gr.d_g_2_TGF(d_g, f_name)
    os.system("cat %s" % f_name)
    
    print("\nreleemos TFG y comprobamos ..........")
    d_g2 = gr.TGF_2_d_g(f_name)
    
    gr.print_d_g(d_g2)
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\nsingle source Dijkstra ....................")
    print("\ncomprobamos la corrección de Dijkstra contra laa soluciones de \
           NetworkX  ..........")
    
    #d_g = {
    #       0: {1: 10, 2: 1}, 
    #       1: {2: 1}, 
    #       2: {3: 1},
    #       3: {1: 1}
    #      }
    
    print("\tgeneramos_grafo_aleatorio ..........")
    m_g = gr.rand_matr_pos_graph(n_nodes=5, sparse_factor=0.75, max_weight=10.)    
    d_g = gr.m_g_2_d_g(m_g)
    nx_g = gr.d_g_2_nx_g(d_g)
    
    print("\tcalculamos d y p mediante nuestro Dijkstra y mediante networkx ..........")
        
    l_difs = []
    for u in d_g.keys():
        d, p = gr.dijkstra_d(d_g, u)
        d_n, p_n = nx.single_source_dijkstra(nx_g, u, weight='weight')
        
        print( ("\ndistancias from %d:\n" % u),  d)
        print( ("\ndistancias_nx from %d:\n" % u),  d_n)
        
        #calculamos y guardamos diferencias entre distancias 
        df = pd.DataFrame.from_dict(d, orient='index', columns=['dijks']).sort_index()
        df_n = pd.DataFrame.from_dict(d_n, orient='index', columns=['dijks']).sort_index()
        l_difs.append(abs(df['dijks'].values - df_n['dijks'].values).max())
        
    _ = input("\npulsar Intro para continuar ....................\n")
    print("\ncomprobamos que todas las distancias son iguales ..........")
    print("max_dif_distancias", abs(np.array(l_difs)).max())
    if abs(np.array(l_difs)).max() > 0:
        idx_dif = np.where(abs(np.array(l_difs)) != 0.)[0]
        print("\tdifs en vertices:", idx_dif)
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\najuste de tiempos dijkstra ....................")
    
    l_t = gr.time_dijkstra_d(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor)
    #gr.fit_plot(l_t, gr.n2_log_n, size_ini=n_nodes_ini, size_fin=n_nodes_fin, step=step)        

    l_t_nx = gr.time_dijkstra_nx(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor)
    #gr.fit_plot(l_t_nx, n2_log_n, size_ini=n_nodes_ini, size_fin=n_nodes_fin, step=step)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="comprobador de la práctica 1.")
    
    parser.add_argument("-ng", "--num_graphs", type=int, default=10)
    parser.add_argument("-is", "--initial_size", type=int, default=10)
    parser.add_argument("-fs", "--final_size", type=int, default=20)
    parser.add_argument("-s", "--step", type=int, default=2)    
    parser.add_argument("-sf", "--sparse_factor", type=float, default=0.5)    
    
    args = parser.parse_args()
    
    main(args.num_graphs, args.initial_size, args.final_size, args.step, args.sparse_factor)
    