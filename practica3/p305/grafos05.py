import sys
import time
# import matplotlib.pyplot as plt
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
    graph = np.array([
        np.array([np.inf for i in range(0, n_nodes)])
        for j in range(0, n_nodes)
    ])
    for i in range(0, n_nodes):
        for j in range(0, n_nodes):
            if i != j and np.random.random() <= sparse_factor:
                graph[i][j] = np.round(np.random.random() * max_weight,
                                       decimals)
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
        return [v]  # This is the first node of Dijks
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
        generate (funcion)-- Funcion que genera un grafo aleatorio dado el número de
                    nodos y el factor de dispersion.
        dijks (funcion)-- Dijkstra a utilizar. Recibe el grafo generado 
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
    """Función aportada en el notebook auxiliar
    """
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

    #if label:
    #    plt.plot(x_vals, l, '*', x_vals, y_pred, '-', label=label)
    #else:
    #    plt.plot(x_vals, l, '*', x_vals, y_pred, '-', label=label)
    #if xlabel:
    #    plt.xlabel(xlabel)
    #if ylabel:
    #    plt.ylabel(ylabel)
    #plt.legend(loc=0)


def n2_log_n(n):
    """Función aportada en el notebook auxiliar
    """
    return n**2. * np.log(n)


def print_m_g(m_g):
    """Función aportada en el notebook auxiliar
    """
    print("graph_from_matrix:\n")
    n_v = m_g.shape[0]
    for u in range(n_v):
        for v in range(n_v):
            if v != u and m_g[u, v] != np.inf:
                print("(", u, v, ")", m_g[u, v])


def print_d_g(d_g):
    """Función aportada en el notebook auxiliar
    """
    print("\ngraph_from_dict:\n")
    for u in d_g.keys():
        for v in d_g[u].keys():
            print("(", u, v, ")", d_g[u][v])


## FIN DE FUNCIONES DEL CUADERNO AUXILIAR


def edges(d_g):
    """Dado un grafo representado como diccionario devuelve una lista de sus ramas

    Argumentos:
        d_g -- Grafo representado como diccionario de diccionarios
    Retorno:
        Una lista de 3-tuplas con todas las ramas 
        [(nodo_origen1, nodo_destino1, peso1), ...]
    """
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
        dijks (funcion)-- Algoritmo de Dijkstra a utilizar. Recibe el grafo y el nodo
                 inicial.
        generate (funcion)-- Generador de grafos aleatorios. Recibe número de nodos y
                    factor de dispersion.
        n_nodes -- Número fijo de nodos con los que se genera el grafo aleatorio.
        n_graphs -- Número de grafos generados en cada iteracion.
        for_all_nodes (boolean)-- Si Dijkstra se ejecuta desde todos los nodos o solo desde
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
            for i in range(iters):
                try:
                    dijks(g, i)
                except:
                    pass  # If no path is found, just continue with next node
            mean_time += (time() - time_ini)
        times.append(mean_time / n_graphs)
        rho += step
    return times


def print_d_mg(g):
    print(g)


## PRACTICA 2


def graph_2_multigraph(d_mg):
    """Recibe un grafo como lista de adyacencias y devuelve un multigrafo
    
    Argumentos:
        d_mg ({int:{int:int}}) -- Lista de adyacencias de un grafo
    Retorno:
        Retorna las listas de adyacencias del multigrafo resultante 
        ({int:{int:{int:int}}})
    """
    outp = {}
    for u in d_mg:
        outp[u] = {}
        for v in d_mg[u]:
            outp[u][v] = {0: d_mg[u][v]}
    return outp


def rand_weighted_multigraph(n_nodes,
                             prob=0.2,
                             num_max_multiple_edges=1,
                             max_weight=50.,
                             decimals=0,
                             fl_unweighted=False,
                             fl_diag=True):
    """Genera un multigrafo aleatorio dirigido

    Argumentos:
        n_nodes (int) -- Número de nodos del multigrafo
        prob (float) -- Factor de dispersión (entre 0 y 1)
        num_max_multiple_edges (int) -- Número máximo de ramas entre dos nodos
        max_weight (float) -- Peso  máximo de una arista
        decimals (int) -- Número de decimales de los pesos de cada rama
        fl_unweighted (bool) -- False si el grafo es ponderado, True si no lo es
        fl_dia (bool) -- False si no se permiten aristas ciclos, True si se permiten
    Retorno:
        Retorna las listas de adyacencias del multigrafo resultante 
        ({int:{int:{int:int}}})
    """
    mdg = {u: {} for u in range(n_nodes)}
    for u in range(n_nodes):
        for v in range(n_nodes):
            if v != u or fl_diag:  # Caso de ciclo
                if np.random.random() <= prob:
                    mdg[u][v] = {}
                    # Num de ramas uniforme entre 1 y num_max
                    num_ramas = np.random.randint(1,
                                                  num_max_multiple_edges + 1)
                    for i in range(num_ramas):
                        if fl_unweighted:
                            mdg[u][v][i] = np.random.randint(0, 2)
                        else:
                            mdg[u][v][i] = np.round(
                                np.random.random() * max_weight, decimals)
    return mdg


def rand_weighted_undirected_multigraph(n_nodes,
                                        prob=0.2,
                                        num_max_multiple_edges=1,
                                        max_weight=50.,
                                        decimals=0,
                                        fl_unweighted=False,
                                        fl_diag=True):
    """Genera un multigrafo aleatorio dirigido

    Argumentos:
        n_nodes (int) -- Número de nodos del multigrafo
        prob (float) -- Factor de dispersión (entre 0 y 1)
        num_max_multiple_edges (int) -- Número máximo de ramas entre dos nodos
        max_weight (float) -- Peso  máximo de una arista
        decimals (int) -- Número de decimales de los pesos de cada rama
        fl_unweighted (bool) -- False si el grafo es ponderado, True si no lo es
        fl_dia (bool) -- False si no se permiten aristas ciclos, True si se permiten
    Retorno:
        Retorna las listas de adyacencias del multigrafo resultante 
        ({int:{int:{int:int}}})
    """
    mdg = {u: {} for u in range(n_nodes)}
    for u in range(n_nodes):
        for v in range(u, n_nodes):
            if v != u or fl_diag:  # Caso de ciclo
                if np.random.random() <= prob:
                    mdg[u][v] = {}
                    mdg[v][u] = {}
                    # Num de ramas uniforme entre 1 y num_max
                    num_ramas = np.random.randint(1,
                                                  num_max_multiple_edges + 1)
                    for i in range(num_ramas):
                        if fl_unweighted:
                            r = np.random.randint(0, 2)
                        else:
                            r = np.round(np.random.random() * max_weight,
                                         decimals)
                        mdg[u][v][i] = r
                        mdg[v][u][i] = r
    return mdg


def o_a_tables(u, d_mg, p, s, o, a, c):
    """Modificación de DFS para calcular las tablas o y a
    
    Argumentos:
        u -- Un vértice del grafo
        d_mg -- Diccionario de un multigrafo en formato
        p ({nodo:nodo}) -- Tabla de previos
        s ({nodo:bool}) -- Tabla de vistos
        o ({nodo:int}) -- Tabla orden de paso
        a ({nodo:int}) -- Tabla de orden de ascenso
        c (int) -- Contador para actualizar la tabla de orden
    Retorno:
        Devuelve el contador de la tabla de orden (int)
    """
    s[u] = True
    o[u] = c
    a[u] = o[u]
    c += 1
    for v in d_mg[u]:
        if s[v] and p[u] != v:
            a[u] = min(a[u], o[v])
    for v in d_mg[u]:
        if not s[v]:
            p[v] = u
            c = o_a_tables(v, d_mg, p, s, o, a, c)
    for v in d_mg[u]:
        if p[v] == u:
            a[u] = min(a[u], a[v])
    return c


def p_o_a_driver(d_mg, u=0):
    """Calcula las tabla P, O y A de un multigrafo

    Argumentos:
        d_mg -- Lista de adyacencias de multigrafo
        u -- Nodo inicial desde el cual iniciar el árbol de DFS
    Retorno:
        Devuelve tres diccionarios:
            p ({nodo:nodo}) -- Dict de padres del árbol DFS
            o ({nodo:int}) -- Órden de descubrimiento del árbol
            a ({nodo:int}) -- Órden de ascenso de cada nodo
    """
    p = {u: None}
    s = {i: False for i in d_mg}
    o = {i: np.inf for i in d_mg}
    a = {i: np.inf for i in d_mg}
    o_a_tables(u, d_mg, p, s, o, a, 0)
    return p, o, a


def hijos_bp(u, p):
    """Calcula los hijos de un nodo en un árbol DFS

    Argumentos:
        u -- Nodo del que se quieren saber sus hijos
        p -- Diccionario de padres del árbol DFS
    Returno:
        Un array con los hijos del nodo en el árbol
    """
    return np.array([v for v in p if p[v] == u])


def check_pda(p, o, a):
    """Devuelve los puntos de articulación de un multigrafo

    Argumentos:
        p ({nodo:nodo}) -- Dict de padres del árbol DFS
        o ({nodo:int}) -- Órden de descubrimiento del árbol
        a ({nodo:int}) -- Órden de ascenso de cada nodo
    Retorno:
        Si el grafo no es conexo devuelve None.
        Si el grafo es conexo entonces devuelve una lista
        con los puntos de articulación.
    """
    if len(p) == len(o):
        # Hay solución
        p_articulacion = []
        for u in p:
            hijos = hijos_bp(u, p)
            if p[u] == None:  # Raiz
                if len(hijos) > 1:
                    p_articulacion.append(u)
            else:  # No raiz
                if len(hijos) > 0 and any(o[u] <= a[h] for h in hijos):
                    p_articulacion.append(u)
        return np.array(p_articulacion)
    return None


def time_pda(n_graphs, n_nodes_ini, n_nodes_fin, step, prob):
    """ Calcula los tiempos de ejecución del cómputo de las tablas P, O y A.

    Argumentos:
        n_graphs (int) -- Número de grafos a generar por cada tamaño
        n_nodes_ini (int) -- Tamaño inicial (de nodos) del grafo
        n_nodes_fin (int) -- Tamaño final (de nodos) del grafo
        step (int) -- Salto de tamaños entre pruebas
        prob (float) -- Factor de dispersión.
    Retorno:
        Devuelve un array con los tiempos medios de ejecución por cada tamaño 
        de grafo. SOLO se toman en cuenta los tiempos de ejecución de los grafos
        conexos.
        Si para alguno de los tamaños pasa que todos los grafos son no conexos
        entonces la función devuelve una lista vacía.
    """
    meanTimes = [0 for _ in range(n_nodes_ini, n_nodes_fin + 1, step)]
    i = 0
    for nodes in range(n_nodes_ini, n_nodes_fin + 1, step):
        counter = 0
        for _ in range(n_graphs):
            g = rand_weighted_undirected_multigraph(nodes, prob)
            t_ini = time()
            p, o, a = p_o_a_driver(g, 0)
            if len(p) == len(o):  # Si es conexo
                meanTimes[i] += (time() - t_ini)
                counter += 1
        if counter == 0:
            return []
        meanTimes[i] /= counter
        i += 1
    return meanTimes


def init_cd(d_g):
    """Iniciamos la estructura de conjuntos disjuntos

    Argumentos:
        d_g -- Lista de adyacencias de un multigrafo/grafo
    Retorno:
        Devuelve la estructura cd incializada
    """
    return {u: -1 for u in d_g}


def union(rep_1, rep_2, d_cd):
    """Une dos conjuntos de la estructura CD.

    Argumentos:
        rep_1 -- Representante de un conjunto 
        rep_2 -- Representante de otro conjunto
        d_cd -- Estructura de conjunto disjunto
    Retorno:
        El representante resultante de unir ambos conjuntos.
    """
    if d_cd[rep_2] < d_cd[rep_1]:
        d_cd[rep_1] = rep_2
        return rep_2
    elif d_cd[rep_2] > d_cd[rep_1]:
        d_cd[rep_2] = rep_1
        return rep_1
    else:
        d_cd[rep_2] = rep_1
        d_cd[rep_1] -= 1
        return rep_1


def find(ind, d_cd, fl_cc):
    """Encuentra el representante de un elemento en una estructura de CD.

    Argumento:
        ind -- Elemento del que se quiere buscar su representante
        d_cd -- Estructura de conjunto disjunto
        fl_cc (bool) -- Si debería, o no, realizar compresión de caminos.
    Retorno:
        Devuelve el representante del conjunto al que pertenece ind en el CD.
    """
    #Buscamos el representante
    rep = ind
    while type(d_cd[rep]) is not int or d_cd[rep] >= 0:
        # Los nodos pueden ser cualquier cosa menos enteros negativos
        rep = d_cd[rep]

    if fl_cc:
        #Compresión de caminos
        while type(d_cd[ind]) is not int or d_cd[ind] >= 0:
            aux = d_cd[ind]
            d_cd[ind] = rep
            ind = aux
    return rep


def insert_pq(d_mg, q):
    """Inserta todas las ramas de un multigrafo no dirigido.

    Argumentos:
        d_mg -- Lista de adyacencia de multigrafo NO dirigido
        q -- Cola en la que se insertarán las ramas
    Nota:
        - Cada rama del multigrafo no dirigido se introduce en un solo sentido
        - No se introducen los ciclos (no son útiles para Kruskal)
    """
    for u in d_mg:
        for v in d_mg[u]:
            if u < v:
                for edge in d_mg[u][v]:
                    q.put((d_mg[u][v][edge], u, v))


def kruskal(d_g, fl_cc=True):
    """Devuelve el árbol abarcador mínimo calculado con Kruskal

    Argumentos:
        d_g -- Lista de adyacencias del multigrafo no dirigido.
        fl_cc -- Si deberíamos utilizar, o no, compresión de caminos en el CD.
    Retorno:
        Devuelve un multigrafo con el MST, o None si el grafo no es conexo.
    """
    cd = init_cd(d_g)
    q = PriorityQueue()
    insert_pq(d_g, q)
    mintree = {u: {} for u in d_g}
    num_ramas = 0
    while not q.empty():
        w, u, v = q.get()
        repu, repv = find(u, cd, fl_cc), find(v, cd, fl_cc)
        if repu != repv:
            union(repu, repv, cd)
            mintree[u][v] = {0: w}
            mintree[v][u] = {0: w}
            num_ramas += 1
    if num_ramas != len(d_g) - 1:  # Si no es conexo
        return None
    return mintree


def kruskal2(d_g, fl_cc=True):
    """Devuelve el árbol abarcador mínimo calculado con Kruskal
    Además, devuelve el tiempo de ejecución del algoritmo 
    (sin contar la gestión de la cola de prioridad)

    Argumentos:
        d_g -- Lista de adyacencias del multigrafo no dirigido.
        fl_cc -- Si deberíamos utilizar, o no, compresión de caminos en el CD.
    Retorno:
        Si el grafo es conexo:
            MST -- Multigrafo con el MST
            t -- Tiempo de ejecución del algoritmo sin la inserción inicial a la cola
        Si el grafo no es conexo:
            None
            -1
    """
    cd = init_cd(d_g)
    q = PriorityQueue()
    insert_pq(d_g, q)
    mintree = {u: {} for u in d_g}
    num_ramas = 0
    ttime = 0
    while not q.empty():
        w, u, v = q.get()
        t_ini = time()
        repu, repv = find(u, cd, fl_cc), find(v, cd, fl_cc)
        if repu != repv:
            union(repu, repv, cd)
            mintree[u][v] = {0: w}
            mintree[v][u] = {0: w}
            num_ramas += 1
        ttime += (time() - t_ini)
    if num_ramas != len(d_g) - 1:  # Si no es conexo
        return None, -1
    return mintree, ttime


def time_kruskal_2(n_graphs, n_nodes_ini, n_nodes_fin, step, prob, fl_cc):
    """ Calcula los tiempos de ejecución de Kruskal,
    sin tener en cuenta el tiempo de inserción inicial en la cola de prioridad.

    Argumentos:
        n_graphs (int) -- Número de grafos a generar por cada tamaño
        n_nodes_ini (int) -- Tamaño inicial (de nodos) del grafo
        n_nodes_fin (int) -- Tamaño final (de nodos) del grafo
        step (int) -- Salto de tamaños entre pruebas
        prob (float) -- Factor de dispersión.
        fl_cc (bool) -- Si se utiliza, o no, compresión de caminos en el CD.
    Retorno:
        Devuelve un array con los tiempos medios de ejecución por cada tamaño 
        de grafo. SOLO se toman en cuenta los tiempos de ejecución de los grafos
        conexos.
        Si para alguno de los tamaños pasa que todos los grafos son no conexos
        entonces la función devuelve una lista vacía.
    """
    meanTimes = [0 for _ in range(n_nodes_ini, n_nodes_fin + 1, step)]
    i = 0
    for nodes in range(n_nodes_ini, n_nodes_fin + 1, step):
        counter = 0
        for _ in range(n_graphs):
            g = rand_weighted_undirected_multigraph(nodes, prob)
            mst, t = kruskal2(g, fl_cc)
            if mst is not None:
                meanTimes[i] += t
                counter += 1
        if counter == 0:
            return []
        meanTimes[i] /= counter
        i += 1
    return meanTimes


def time_kruskal(n_graphs, n_nodes_ini, n_nodes_fin, step, prob, fl_cc):
    """ Calcula los tiempos de ejecución de Kruskal.

    Argumentos:
        n_graphs (int) -- Número de grafos a generar por cada tamaño
        n_nodes_ini (int) -- Tamaño inicial (de nodos) del grafo
        n_nodes_fin (int) -- Tamaño final (de nodos) del grafo
        step (int) -- Salto de tamaños entre pruebas
        prob (float) -- Factor de dispersión.
        fl_cc (bool) -- Si se utiliza, o no, compresión de caminos en el CD.
    Retorno:
        Devuelve un array con los tiempos medios de ejecución por cada tamaño 
        de grafo. SOLO se toman en cuenta los tiempos de ejecución de los grafos
        conexos.
        Si para alguno de los tamaños pasa que todos los grafos son no conexos
        entonces la función devuelve una lista vacía.
    """
    meanTimes = [0 for _ in range(n_nodes_ini, n_nodes_fin + 1, step)]
    i = 0
    for nodes in range(n_nodes_ini, n_nodes_fin + 1, step):
        counter = 0
        for _ in range(n_graphs):
            g = rand_weighted_undirected_multigraph(nodes, prob)
            t_ini = time()
            mst = kruskal(g, fl_cc)
            if mst is not None:
                meanTimes[i] += (time() - t_ini)
                counter += 1
        if counter == 0:
            return []
        meanTimes[i] /= counter
        i += 1
    return meanTimes



## PRACTICA 3

# Número aleatorio máximo a generar por encima de la suma
MAX_RAND_ADD = 1000

def gen_super_crec(n_terms:int):
    """TODO
    """
    assert n_terms > 0, "El número de términos ha de ser positivo"
    s = random.randint(1, MAX_RAND_ADD)
    superc = [s]
    for _ in range(n_terms-1):
        x = random.randint(1, MAX_RAND_ADD)
        superc.append(s+x)
        s = s*2 + x # Suma de todo lo anterior más el nuevo termino
    return superc

def mcd(x:int, y:int):
    """TODO
    """
    if x==0 or y==0:
        return 0
    if x<y:
        x, y = y ,x
    while y!=0:
        x,y = y, x%y 
    return x

def multiplier(mod: int , mult_ini:int ):
    """TODO
    """
    assert mod>mult_ini "El multiplicador tiene que ser menor que el mod"
    p = random.randint(mult_ini, mod)
    while mcd(mod,p)!=1:
        p = random.randint(mult_ini, mod)
    return p

def inverse(p:int, mod:int):
    """TODO
    """
    assert mod>p "El valor tiene que ser menor que el módulo"
    for q in range(1,mod):
        if (p*q)%mod == 1:
            return q
    return -1

def mod_mult_inv(l_sc):
    """TODO
    """
    s = sum(l_sc) + random.randint(1, MAX_RAND_ADD)
    p = multiplier(s, s//5 + 1)
    q = inverse(p, s)
    return p,q,s

def gen_sucesion_publica(l_sc, p:int, mod:int):
    """TODO
    """
    return [(x*p)%mod for x in l_sc]

def l_publica_2_l_super_crec(l_pub, q:int, mod:int):
    """TODO
    """
    return [(x*q)%mod for x in l_pub]

def gen_random_bit_list(n_bits):
    """TODO
    """
    return [random.randint(0,1) for _ in range(n_bits)]


def mh_encrypt(l_bits, l_pub, mod:int):
    """TODO
    """
    if len(l_bits)%len(l_pub)!=0: # Rellenamos con ceros
        l_bits += [0]*(len(l_pub)- len(l_bits)%len(l_pub))
    
    encrypted = []
    i_bits = 0
    while i_bits<len(l_bits):
        # Ciframos un bloque
        e = 0
        for _ in range(len(l_pub)):
            if l_bits[i_bits] == 1:
                e += l_pub[i_bits%len(l_pub)]
            i_bits += 1
        encrypted.append(e)
    return e

def mh_block_decrypt(c:int, l_sc, inv:int, mod:int):
    """TODO
    """
    c = (c*inv)%mod
    bits = []
    for i in range(len(l_sc)-1, -1, -1):
        if c == 0:
            bits.append(0)
        elif l_sc < c:
            bits.append(1)
            c -= l_sc
        elif l_sc > c:
            bits.append(0)
        else:
            bits.append(1)
            c = 0
    return bits

def mh_decrypt(l_cifra, l_sc, inv:int , mod:int):
    """TODO
    """
    bits = []
    for cifra  in l_cifra:
        bits += mh_block_decrypt(cifra, l_sc, inv, mod)
    return bits

# TODO FALTA PROBAR TODO ESTE CODIGOOOOO
# USO ASSERTS MÁS ESTRICTOS???

def min_coin_number(c:int, l_coins):
    """TODO
    """
    dp = [[None]*(c+1)]*len(l_coins) 
    # Condiciones frontera
    for i in range( c+1):
        dp[0][i] = i # La primera moneda es siempre 1
    for i in range(1, len(l_coins)):
        dp[i][0] = 0
    # Dynamic programming
    for i in range(1,len(l_coins)):
        for v in range(1,c+1):
            if v-l_coins[i] >= 0:
                dp[i][v] = min(dp[i-1][v], 1+dp[i][v-l_coins[i]])
            else:
                dp[i][v] = dp[i-1][v]
## TODOOOOO TERMINARRRR 
