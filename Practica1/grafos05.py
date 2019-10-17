import sys
import time
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import gzip
import os
from time import time
from queue import PriorityQueue

from sklearn.linear_model import LinearRegression

import networkx as nx


def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50, decimals=0):
    """ Genera un grafo aleatorio en forma de matriz de adyacencia.

    Argumentos:
        n_nodes -- Numero de nodos del grafo (del 0 al n_nodes-1).
        sparse_factor -- Factor de dispersion APROXIMADO de la matriz generada.
        max_weight -- Peso maximo de las ramas del grafo (son enteros).
        decimals -- ??.
    Retorno:
        Una matriz de adyacencias que representa al grafo generado.
    """
    graph = [[np.inf for i in range(0, n_nodes)] for j in range(0, n_nodes)]
    for i in range(0, n_nodes):
        for j in range(0, n_nodes):
            if i != j and random.random() <= sparse_factor:
                graph[i][j] = random.randint(1, max_weight)
    return graph


def cuenta_ramas(m_g):
    """ Cuenta el numero de ramas en grafo representado como matriz.

    Argumentos:
        m_g -- Matriz de adyacencia que representa al grafo.
    Retorno:
        Numero (entero) de ramas.
    """
    l = 0
    for i in range(0, len(m_g)):
        for j in range(0, len(m_g)):
            if m_g[i][j] < float('inf'):
                l += 1
    return l


def calculate_sparse_factor(g):
    """Calcula el factor de dispersion de un grafo representado como matriz.

    Argumentos:
        g -- Matriz de adyacencias del grafo.
    Retorno:
        Factor de dispersion del grafo (float).
    """
    return (cuenta_ramas(g)) / (len(g) * (len(g) - 1.0))


def check_sparse_factor(n_grafos, n_nodes, sparse_factor):
    """Funcion que comprueba que los factores de dispersion generados concuerdan
    con los esperados.

    Argumentos:
        n_grafos -- Número de grafos a generar para la comprobacion.
        n_nodes -- Número de nodos de cada grafo generado.
        sparse_factor -- Factor de dispersion esperado.
    Retorno:
        Media de los factores de dispersion obtenidos (float).
    Nota:
        Idealmente el retorno y el sparse_factor dado como argumento son
        muy parecidos.
    """
    values = [
        calculate_sparse_factor(rand_matr_pos_graph(n_nodes, sparse_factor))
        for i in range(0, n_grafos)
    ]
    return sum(values) / (n_grafos)


def m_g_2_d_g(m_g):
    """Transforma una matriz de adyacencias en listas de adyacencias.

    Argumentos:
        m_g -- Grafo representado como matriz de adyacencias.
    Retorno:
        Grafo represenado como diccionario de listas de adyacencias
    """
    graph = {}
    for i in range(0, len(m_g)):
        graph[i] = {}
        for j in range(0, len(m_g)):
            if m_g[i][j] < np.inf:
                graph[i][j] = m_g[i][j]
    return graph


def d_g_2_m_g(d_g):
    """Transforma una listas de adyacencias en una matriz de adyacencia.

    Argumentos:
        d_g -- Grafo representado como lista de adyacencias.
    Retorno:
        Grafo representado como matriz de adyacencias.
    """
    graph = [[np.inf for i in range(0, len(d_g))] for j in range(0, len(d_g))]
    for v1 in d_g:
        for v2 in d_g[v1]:
            graph[v1][v2] = d_g[v1][v2]
    return graph


def save_object(obj, f_name='obj.pklz', save_path='.'):
    """Serializa con pickle y guarda un objeto en un fichero comprimido.

    Argumentos:
        obj -- Objeto que queremos serializar.
        f_name -- Nombre del fichero en el que escribiremos el objeto.
        save_path -- Ruta hasta el directorio donde se escribe el fichero.
    """
    complete_name = os.path.join(save_path, f_name)
    with gzip.open(complete_name, 'wb') as f:
        pickle.dump(obj, f)


def read_object(f_name, save_path='.'):
    """ Carga un objeto serializado con pickle de un fichero comprimido.

    Argumentos:
        f_name -- Nombre del fichero que contiene el objeto.
        save_path -- Ruta al directorio donde esta f_name.
    Retorno:
        Objeto cargado.
    """
    complete_name = os.path.join(save_path, f_name)
    obj = None
    with gzip.open(complete_name) as f:
        obj = pickle.load(f)
    return obj


def d_g_2_TGF(d_g, f_name):
    """Escribe un grafo representado como lista de adyacencias en TGF.

    Argumentos:
        d_g -- Grafo representado como lista de adyacencias.
        f_name -- Fichero en el que se volcara el grafo como TGF.
    """
    out = ''
    for v1 in d_g:
        out += str(v1) + '\n'
    out += '#\n'
    for v1 in d_g:
        for v2 in d_g[v1]:
            out += str(v1) + ' ' + str(v2) + ' ' + str(d_g[v1][v2]) + '\n'
    with open(f_name, 'w') as f:
        f.write(out)


def TGF_2_d_g(f_name):
    """Convierte un grafo en TGF a lista de adyacencias.

    Argumentos:
        f_name -- Fichero en el que se encuentra  el grafo en TGF.
    Retorno:
        Grafo representado como lista de adyacencias.
    """
    with open(f_name) as f:
        tgf = f.read()
    nodes, edges = tgf.split('#\n')
    nodes = nodes.split('\n')[:-1]  # Last is empty
    edges = list(map(lambda x: x.split(' '), edges.split('\n')[:-1]))
    graph = {}
    for node in nodes:
        graph[int(node)] = {}
    for v1, v2, weight in edges:
        graph[int(v1)][int(v2)] = float(weight)
    return graph


# DIJKSTRA!
def dijkstra_d(d_g, u):
    """Ejecuta dijkstra sobre un grafo como lista de adyacencias.

    Argumentos:
        d_g -- Grafo representado como lista de adyacencias.
        u -- Nodo desde el que se inicia el algoritmo.
    Retorno:
        (a,b):
            a -- Lista de distancias minimas (a[i] distancia minima de u a i).
            b -- Lista de padres en caminos minimos (a[i] padre de i).
    """
    dist = {}
    prev = {}
    visited = {}
    for ele in d_g:
        dist[ele] = np.inf
        if ele != u:
            prev[ele] = -1
        else:
            prev[ele] = -2
        visited[ele] = False
    q = PriorityQueue()
    dist[u] = 0
    q.put((0, u))
    while not q.empty():
        w, curr = q.get()
        if not visited[curr]:
            visited[curr] = True
            for node in d_g[curr]:
                if d_g[curr][node] + w < dist[node]:
                    dist[node] = d_g[curr][node] + w
                    prev[node] = curr
                    q.put((dist[node], node))
    return dist, prev


def dijkstra_m(m_g, u):
    """Ejecuta dijkstra sobre un grafo como matriz de adyacencias.

    Argumentos:
        m_g -- Grafo representado como matriz de adyacencias.
        u -- Nodo desde el que se inicia el algoritmo.
    Retorno:
        (a,b):
            a -- Lista de distancias minimas (a[i] distancia minima de u a i).
            b -- Lista de padres en caminos minimos (a[i] padre de i).
    """
    dist = {}
    prev = {}
    visited = {}
    for ele in range(len(m_g)):
        dist[ele] = np.inf
        if ele != u:
            prev[ele] = -1
        else:
            prev[ele] = -2
        visited[ele] = False
    q = PriorityQueue()
    dist[u] = 0
    q.put((0, u))
    while not q.empty():
        w, curr = q.get()
        if not visited[curr]:
            visited[curr] = True
            for node in range(len(m_g[curr])):
                if m_g[curr][node] + w < dist[node]:
                    dist[node] = m_g[curr][node] + w
                    prev[node] = curr
                    q.put((dist[node], node))
    return dist, prev


def min_path(d_prev, v):
    """Recupera el camino minimo hasta v dado el retorno de Dijkstra.

    Argumento:
        d_prev -- Lista de padres devuelta por Dijkstra.
        v -- Nodo final del camino buscado.
    Retorno:
        Lista con el camino de u a v ([u x1 x2 ... xk v]) si existe.
        Si no hay camino devuelve [].
    """
    if d_prev[v] == -2:
        return [v] # This is the first node of Dijks
    elif d_prev[v] < 0:
        return []  # There is no path
    path = [v]
    while d_prev[v] >= 0:
        v = d_prev[v]
        path.append(v)
    path.reverse()
    return path


def min_paths(d_prev):
    """Devuelve un hash con todos los caminos minimos encontrados en Dijkstra.

    Argumentos:
        d_prev -- Lista de padres devuelta por Dijkstra.
    Retorno:
        Un hash que asocia a cada nodo su camino minimo ({u: min_path(u)}).
    """
    d_path = {}
    for ele in d_prev:
        d_path[ele] = min_path(d_prev, ele)
    return d_path


def time_dijkstra(n_graphs,
                  n_nodes_ini,
                  n_nodes_fin,
                  step,
                  generate,
                  dijks,
                  sparse_factor=.25):
    """Funcion que calcula los tiempos de ejecucion de Dijkstra para varios tamaños.

    Argumentos:
        n_graphs -- Número de grafos a generar por cada tamaño.
        n_nodes_ini -- Primer tamaño de grafo utilizado.
        n_nodes_fin -- Último tamaño de grafo utilizado.
        step -- Incremento del número de nodos en cada iteracion.
        generate -- Funcion que genera un grafo aleatorio dado el número de
                    nodos y el factor de dispersion.
        dijks -- Dijkstra a utilizar. Recibe el grafo generado 
                 y el nodo de partida.
        sparse_factor -- Factor de dispersion de los grafos generados.
    Retorno:
        Devuelva una lista de tiempos en segundos. El primer tiempo corresponde
        con el tiempo medio necesario para ejecutar Dijkstra desde todos los
        nodos en un grafo de n_nodes_ini, el segundo correspondera con un grafo 
        de n_nodes_ini+step, asi sucesivamente hasta n_nodes_fin.
    Nota:
        Los tiempos medios devueltos corresponden con el tiempo medio para
        ejecutar Dijkstra desde TODOS los nodos (NO desde un solo nodo).
        Se toma esta decision porque la funcion de fit_plot proporcionada
        busca ajustar una n^3*log(n).
    Nota2:
        Utilizamos el mismo codigo tanto para diccionarios, como para matrices
        como para grafos de networkx, entonces los vertices tienen que ser nums
        de entre 0 y numero de nodos  -1
    """
    times = []  # Returning list
    for nodes in range(n_nodes_ini, n_nodes_fin + 1, step):
        meanTime = 0
        for _ in range(n_graphs):
            graph = generate(nodes, sparse_factor)
            time_ini = time()
            for i in range(nodes):
                dijks(graph, i)
            meanTime += (time() - time_ini)
        times.append(
            meanTime / n_graphs
        )  # No dividimos por n_nodes porque la aproximacion de fit_plot lo toma en cuenta
    return times


def time_dijkstra_m(n_graphs,
                    n_nodes_ini,
                    n_nodes_fin,
                    step,
                    sparse_factor=.25):
    """Instancia de time_dijkstra para matrices de adyacencia.

    NOTA: Detalles del funcionamiento en time_dijkstra.
    """
    return time_dijkstra(
        n_graphs,
        n_nodes_ini,
        n_nodes_fin,
        step,
        rand_matr_pos_graph,
        dijkstra_m,
        sparse_factor=sparse_factor)


def time_dijkstra_d(n_graphs,
                    n_nodes_ini,
                    n_nodes_fin,
                    step,
                    sparse_factor=.25):
    """Instancia de time_dijkstra para lista de adyacencia.

    NOTA: Detalles del funcionamiento en time_dijkstra.
    """
    f = lambda x, y: m_g_2_d_g(rand_matr_pos_graph(x, y))
    return time_dijkstra(
        n_graphs,
        n_nodes_ini,
        n_nodes_fin,
        step,
        f,
        dijkstra_d,
        sparse_factor=sparse_factor)


## FUNCIONES DEL CUADERNO AUXILIAR


# La siguiente funcion la modificamos ligeramente para que aceptase algún caso
# adicional, como el ajuste cambiando el factor de dispersion. Tambien agregamos
# algunos campos para configurar la grafica.
def fit_plot(l,
             func_2_fit,
             size_ini,
             size_fin,
             step,
             label=None,
             rho=False,
             xlabel=None,
             ylabel=None):
    if not rho:
        x_vals = [i for i in range(size_ini, size_fin + 1, step)]
        l_func_values = [
            i * func_2_fit(i) for i in range(size_ini, size_fin + 1, step)
        ]
    else:
        # Para el fitting del rho que requiere de iteracion con floats
        # Nota: No  multiplicamos por el número de nodos porque en este caso es cte
        l_func_values = []
        x_vals = []
        i = size_ini
        while i <= size_fin:
            x_vals.append(i)
            l_func_values.append(func_2_fit(i))
            i += step

    lr_m = LinearRegression()
    X = np.array(l_func_values).reshape(len(l_func_values), -1)
    lr_m.fit(X, l)
    y_pred = lr_m.predict(X)

    if label:
        plt.plot(x_vals, l, '*', x_vals, y_pred, '-', label=label)
    else:
        plt.plot(x_vals, l, '*', x_vals, y_pred, '-', label=label)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)


def n2_log_n(n):
    return n**2. * np.log(n)


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


## FIN DE FUNCIONES DEL CUADERNO AUXILIAR


def edges(d_g):
    # Funcion auxiliar privada. Devuelve una lista de aristas de una grafo
    # representado como matriz
    e = []
    for u in d_g:
        for v in d_g[u]:
            e.append((u, v, d_g[u][v]))
    return e


def d_g_2_nx_g(d_g):
    """Convierte un grafo en lista de adyacencias en uno de NetworkX.

    Argumentos:
        d_g -- Grafo representado como lista de adyacencias.
    Retorno:
        Grafo como objeto de libreria de NetworkX.
    """
    g = nx.DiGraph()
    g.add_weighted_edges_from(edges(d_g))
    return g


def nx_g_2_d_g(nx_g):
    """Convierte un grafo de NetworkX en uno como lista de adyacencias.

    Argumentos:
        nx_g -- Grafo como objeto de NetworkX.
    Retorno:
        Grafo como lista de adyacencias.
    """
    d_g = {}
    for u in nx_g:
        d_g[u] = {}
        for v in nx_g[u]:
            d_g[u][v] = nx_g[u][v]['weight']
    return d_g


def time_dijkstra_nx(n_graphs,
                     n_nodes_ini,
                     n_nodes_fin,
                     step,
                     sparse_factor=.25):
    """Instancia de time_dijkstra para grafos de NetworkX.

    NOTA: Detalles del funcionamiento en time_dijkstra.
    """
    f = lambda x, y: d_g_2_nx_g(m_g_2_d_g(rand_matr_pos_graph(x, y)))
    return time_dijkstra(
        n_graphs,
        n_nodes_ini,
        n_nodes_fin,
        step,
        f,
        nx.single_source_dijkstra,
        sparse_factor=sparse_factor)


##### APeNDICE


def time_dijks_rho(rho_ini,
                   rho_fin,
                   step,
                   dijks,
                   generate,
                   n_nodes=100,
                   n_graphs=30,
                   for_all_nodes=True):
    """Calcula tiempos de ejecucion de Dijkstra iterando sobre el 
    factor de dispersion.

    Argumentos:
        rho_ini -- Factor de dispersion inicial.
        rho_fin -- Factor de dispersion final.
        step -- Incremento del factor de dispersion en cada iteracion.
        dijks -- Algoritmo de Dijkstra a utilizar. Recibe el grafo y el nodo
                 inicial.
        generate -- Generador de grafos aleatorios. Recibe número de nodos y
                    factor de dispersion.
        n_nodes -- Número fijo de nodos con los que se genera el grafo aleatorio.
        n_graphs -- Número de grafos generados en cada iteracion.
        for_all_nodes -- Si Dijkstra se ejecuta desde todos los nodos o solo desde
                         el inicial.
    Retorno:
        Lista de tiempos medios correspondientes a cada factor de dispersion desde 
        rho_ini a rho_fin (incluidos ambos) a pasos de step.
    Nota:
        En caso de que for_all_nodes sea True entonces el tiempo de ejecucion corresponde
        con el tiempo que tarda en ejecutar Dijkstra desde TODOS los nodos.
    """
    if for_all_nodes:
        # Iterate Dijkstra for all nodes
        iters = n_nodes
    else:
        # Only dijkstra from 0
        iters = 1
    times = []
    rho = rho_ini
    while rho <= rho_fin:
        mean_time = 0
        for _ in range(n_graphs):
            g = generate(n_nodes, rho)
            time_ini = time()
            for i in range(n_nodes):
                try:
                    dijks(g, i)
                except:
                    pass  # If no path is found, just continue with next node
            mean_time += (time() - time_ini)
        times.append(mean_time / n_graphs)
        rho += step
    return times
