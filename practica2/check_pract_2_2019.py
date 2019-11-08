#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string, random
import numpy as np
import sys
import os
import argparse

from sklearn.linear_model import LinearRegression

import grafos_2019 as gr
  
####################################### main
def main(n_graphs, n_nodes_ini, n_nodes_fin, step, prob):
    """Prueba las funciones de qs.py.
    
    Args: n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor
    """
    
    #if len(args) != 5:
    #    print ("args: n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor")
    #    sys.exit(0)  
    #n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor = int(args[0]), int(args[1]), int(args[2]), int(args[3]), float(args[4])
    
    print("\ncomprobar corrección de PAs ....................")
    print("\nleemos grafo en formato TFG y convertimos a nueva EdD para multigrafos ..........")
    f_name = 'pda.tgf'
    d_g = gr.TGF_2_d_g(f_name)
    d_mg = gr.graph_2_multigraph(d_g)
    
    gr.print_d_g(d_mg)
    print(d_mg.keys())
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\ncomprobamos la corrección de PdA ..........")
    p, o, a = gr.p_o_a_driver(d_mg, u=1)
    print(p, o, a)

    l_pdas = gr.check_pda(p, o, a)
    print(l_pdas)
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\najuste de tiempos PDA ....................")
    
    l_t = gr.time_pda(n_graphs, n_nodes_ini, n_nodes_fin, step, prob)
    print("times_PDA\n", l_t)
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\ncomprobar corrección de Kruskal ....................")
    print("\nleemos grafo en formato TFG y convertimos a nueva EdD para multigrafos ..........")
    f_name = 'kruskal.tgf'
    d_g = gr.TGF_2_d_g(f_name)
    d_mg = gr.graph_2_multigraph(d_g)
    
    gr.print_d_g(d_mg)
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\ncomprobamos la corrección de Kruskal ..........")
    mst = gr.kruskal(d_mg)
    if mst:
        print("mst")
        gr.print_d_mg(mst)
    
    _ = input("\npulsar Intro para continuar ....................\n")
    
    print("\najuste de tiempos kruskal sin o con CC....................")
    print("\najuste de tiempos kruskal sin CC....................")
    
    l_t = gr.time_kruskal(n_graphs, n_nodes_ini, n_nodes_fin, step, prob, fl_cc=False)
    print("kruskal sin CC\n", l_t)
    
    print("\najuste de tiempos kruskal con CC....................")
    l_t_cc = gr.time_kruskal_2(n_graphs, n_nodes_ini, n_nodes_fin, step, prob, fl_cc=True)
    print("kruskal con CC\n", l_t_cc)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="comprobador de la práctica 2")
    
    parser.add_argument("-ng", "--num_graphs", type=int, default=10)
    parser.add_argument("-is", "--initial_size", type=int, default=10)
    parser.add_argument("-fs", "--final_size", type=int, default=20)
    parser.add_argument("-s", "--step", type=int, default=2)    
    parser.add_argument("-p", "--prob", type=float, default=0.5)    
    
    args = parser.parse_args()
    
    main(args.num_graphs, args.initial_size, args.final_size, args.step, args.prob)
    