"""
Librería para comprobar la práctica 2 de DAALG automáticamente.
Si te da algún fallito, o tienes duda, avísame porfa.
Autor: Rodrigo De Pool (rodrigodp05@gmail.com)
DISCLAIMER: El código es inentendible, fue una pruebita rápida que hice.
"""
from importlib import import_module
import networkx as nx
import sys

# TODO: Cambia el nombre de p al de tu librería
p = "grafos05"

# IMPORTAMOS LA LIBRERÍA
try:
    lib = import_module(p)
except:
    print("No  se ha podido importar tu librería ({})".format(p))
    print("Recuerda cambiar la variable p al nombre de tu librería (línea 9)")
    sys.exit()

# FIJAMOS ALGUNOS PARÁMETROS
num_max_multiple_edges = 10
decimals = 3
max_weight = 50
reps = 5


# IMPRIMIR EN FORMATO LEGIBLE LAS CONEXIONES DE UN GRAFO
def string_conexiones_grafo(g):
    out = ''
    for u in g:
        out += ('Desde {} a'.format(u))
        for v in g[u]:
            out += (' {}'.format(v))
        out += '\n'
    return out


def multigrafo_a_nx(g):
    edges = []
    s = {u: False for u in g}
    for u in g:
        for v in g[u]:
            if u != v or (u == v and not s[v]):
                if u == v: s[u] = True
                for num in g[u][v]:
                    edges.append((u, v, g[u][v][num]))
    ng = nx.MultiGraph()
    ng.add_nodes_from([u for u in g])
    ng.add_weighted_edges_from(edges)
    return ng


#COMPRUEBA LOS PA DE UN GRAFO DADO
def comprobar_pas(g):
    ng = multigrafo_a_nx(g)
    p, o, a = lib.p_o_a_driver(g)
    arts_g = lib.check_pda(p, o, a)
    arts_ng = nx.articulation_points(ng)

    if arts_g is None:
        if not nx.is_connected(ng):
            return False, False
        else:
            raise Exception(
                "Tu librería dice que un grafo conexo no lo es. Grafo:\n{}".
                format(string_conexiones_grafo(g)))

    if not nx.is_connected(ng):
        raise Exception(
            "Tu librería dice que un grafo no conexo lo es. Grafo:\n{}".format(
                string_conexiones_grafo(g)))

    a1 = set(list(arts_g))
    a2 = set(list(arts_ng))
    if a1 == a2:
        return True, len(arts_g) > 0
    else:
        raise Exception(
            "Tu librería da puntos de articulación distintos a los de NetworkX.\n\nEl grafo es:\n{}\nLos PA de tu lib: {}\n\nLos PA de NX: {}"
            .format(string_conexiones_grafo(g), a1, a2))


# TEST COMPLETO DE PAS
def comprobar_puntos_de_articulacion():
    total = 0
    conexos = 0
    conArticulacion = 0
    print("Comprobamos puntos de articulación:\n***")
    for fl_diag in [True, False]:
        for fl_unweighted in [True, False]:
            for n_nodes in range(5, 35, 5):
                prob = 0.2
                while prob <= 0.8:
                    for _ in range(reps):
                        g = lib.rand_weighted_undirected_multigraph(
                            n_nodes,
                            prob=prob,
                            num_max_multiple_edges=num_max_multiple_edges,
                            max_weight=max_weight,
                            decimals=decimals,
                            fl_unweighted=fl_unweighted,
                            fl_diag=fl_diag)
                        esConexo, hayPA = comprobar_pas(g)
                        if esConexo: conexos += 1
                        if hayPA: conArticulacion += 1
                        total += 1
                    prob += 0.2
    print(
        "Total de grafos probados {}\nTotal de grafos conexos {}\nTotal de grafos con PA {}"
        .format(total, conexos, conArticulacion))
    print("Si no hay excepción todo ha ido bien :)")


# VALOR DEL ÁRBOL MÍNIMO, ASUME QUE ES NO DIRIGIDO (HAY RAMAS DUPLICADAS)
def valueOfTree(k):
    out = 0
    for u in k:
        for v in k[u]:
            if u < v:
                for e in k[u][v]:
                    out += k[u][v][e]
    return out


#VALOR DEL ÁRBOL MÍNIMO DE NX
def valueOfTreeOfNX(k):
    out = 0
    for u, v in k.edges():
        for key in k.get_edge_data(u, v):
            out += k.get_edge_data(u, v)[key]['weight']
    return out


# IMPRIME LAS RAMAS CON PESO MÍNIMO
def string_conexiones_grafo_con_peso_min(g):
    out = ''
    for u in g:
        out += ('Desde {} a'.format(u))
        for v in g[u]:
            out += (' {}(w:{})'.format(v, min(g[u][v].values())))
        out += '\n'
    return out


# TEST DE KRUSKAL PARA UN GRAFO CONCRETO
def comprobar_kruskal_g(g):
    ng = multigrafo_a_nx(g)
    k1 = lib.kruskal(g, fl_cc=True)
    k2 = lib.kruskal(g, fl_cc=False)
    if k1 is None:
        if k2 is None:
            if not nx.is_connected(ng):
                return False
            else:
                raise Exception(
                    "Tu librería dice que un grafo conexo no lo es.\nEl grafo es:\n{}"
                    .format(string_conexiones_grafo(g)))
        else:
            raise Exception(
                "Los resultados sobre conexión de caminos cambian si se utiliza o no la compresión de caminos"
            )
    if not nx.is_connected(ng):
        raise Exception(
            "Tu librería dice que un grafo no conexo lo es.\nEl grafo es:\n{}".
            format(string_conexiones_grafo(g)))

    nt = nx.minimum_spanning_tree(ng)
    total1 = valueOfTree(k1)
    total2 = valueOfTree(k2)
    ntotal = valueOfTreeOfNX(nt)
    if total1 != total2:
        raise Exception("Los valores de tus MST con y sin CC no coinciden.")
    if total2 != ntotal:
        raise Exception(
            "El valor de tu MST y el de Networkx no coinciden.\nValor de tu lib: {}.\nValor de NX:{}\nEl grafo es:\n{}\n\nTu solución:\n{}"
            .format(total2, ntotal, string_conexiones_grafo_con_peso_min(g),
                    string_conexiones_grafo_con_peso_min(k2)))
    return True


# TEST COMPLETO DE KRUSKAL
def comprobar_kruskal():
    total = 0
    conexos = 0
    print(
        "Comprobación de Kruskal (solo comprobamos que los pesos de los MST coincidan):\n***"
    )
    for fl_diag in [True, False]:
        for n_nodes in range(5, 35, 5):
            prob = 0.2
            while prob <= 0.8:
                for _ in range(reps):
                    g = lib.rand_weighted_undirected_multigraph(
                        n_nodes,
                        prob=prob,
                        num_max_multiple_edges=num_max_multiple_edges,
                        max_weight=max_weight,
                        decimals=decimals,
                        fl_unweighted=False,
                        fl_diag=fl_diag)
                    esConexo = comprobar_kruskal_g(g)
                    if esConexo: conexos += 1
                    total += 1
                prob += 0.2
    print("Total de grafos probados {}\nTotal de grafos conexos {}".format(
        total, conexos))
    print("Si no hay excepciones todo ha ido bien:)")


# CORREMOS LOS TESTS
comprobar_puntos_de_articulacion()
print("\n\n\n")
comprobar_kruskal()
