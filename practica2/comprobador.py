"""
Librería para comprobar la práctica 2 de DAALG automáticamente.
Si te da algún fallito, o tienes duda, avísame porfa.
Autor: Rodrigo De Pool (rodrigodp05@gmail.com)
"""
from importlib import import_module 
import networkx as nx 
# Importamos tu libería
p = input("Introduce el número de pareja (con cero si es de un dígito): ")
p = "grafos{}".format(p)
try:
    lib = import_module(p)
except:
    print("No  se ha podido importar tu librería ({})".format(p))


num_max_multiple_edges = 10
decimals = 3
max_weight = 50
reps = 5

def multigrafo_a_nx(g):
    edges = []
    s = {u:False for u in g}
    for u in g:
        for v in g[u]:
            if u!=v or (u==v and not s[v]):
                if u==v: s[u] = True
                for num in g[u][v]:
                    edges.append((u,v,g[u][v][num]))
    ng = nx.MultiGraph()
    ng.add_nodes_from([u for u in g])
    ng.add_weighted_edges_from(edges)
    return ng


def  comprobar_pas(g):
    ng = multigrafo_a_nx(g)
    p,o,a= lib.p_o_a_driver(g)
    arts_g = lib.check_pda(p,o,a)
    arts_ng = nx.articulation_points(ng)

    if arts_g is None:
        if not nx.is_connected(ng):
            return False, False
        else:
            raise Exception("Tu librería dice que un grafo conexo no lo es. Grafo:\n{}".format(g))

    if not nx.is_connected(ng):
        raise Exception("Tu librería dice que un grafo no conexo lo es. Grafo:\n{}".format(g))

    a1 = set(list(arts_g)) 
    a2 = set(list(arts_ng))
    if a1 == a2:
        return True, len(arts_g)>0
    else:
        print(a1)
        print(a2)
        print(a1 == a2)
        raise Exception("Tu librería da puntos de articulación distintos a los de NetworkX. El grafo es:\n{}".format(g))
        
def comprobar_puntos_de_articulacion():
    total = 0
    conexos = 0
    conArticulacion = 0
    print("Comprobamos puntos de articulación:\n***")
    for fl_diag in [True, False]:
        for fl_unweighted in [True, False]:
            print("---\nCon los parámetros:\n- num_max_multiple_edges={}\n- decimals={}\n- max_weight={}\n- fl_diag={}\n- fl_unweighted={}".format(num_max_multiple_edges, decimals, max_weight, fl_diag, fl_unweighted))
            for n_nodes in range(5,35, 5):
                prob = 0.2
                while prob<=0.8:
                    for _ in range(reps):
                        g = lib.rand_weighted_undirected_multigraph(n_nodes,
                                        prob=prob,
                                        num_max_multiple_edges=num_max_multiple_edges, 
                                        max_weight=max_weight, 
                                        decimals=decimals, 
                                        fl_unweighted=fl_unweighted, 
                                        fl_diag=fl_diag)
                        esConexo, hayPA = comprobar_pas(g)
                        if esConexo: conexos+=1
                        if hayPA: conArticulacion+=1
                        total+=1
                    prob+=0.2
            print("---")
    print("Total de grafos probados {}\nTotal de grafos conexos {}\nTotal de grafos con PA {}".format(total, conexos, conArticulacion))


def valueOfTree(k):
    out = 0
    for u in k:
        for v in k[u]:
            for e in k[u][v]:
                out += k[u][v][e]
    return out 

def valueOfTreeOfNX(k):
    out = 0
    for u,v in k.edges():
        for val in k.get_edge_data(u,v):
            out += val
    return out
def comprobar_kruskal_g(g):
    ng = multigrafo_a_nx(g)
    k1 = lib.kruskal(g,fl_cc=True)
    k2 = lib.kruskal(g,fl_cc=False)
    if k1 is None:
        if k2 is None:
            if not nx.is_connected(ng):
                return False
            else:
                raise Exception("Tu librería dice que un grafo conexo no lo es.")
        else:
            raise Exception("Los resultados sobre conexión de caminos cambian si se utiliza o no la compresión de caminos")
    if not nx.is_connected(ng):
        raise Exception("Tu librería dice que un grafo no conexo lo es.")
    nt = nx.minimum_spanning_tree(ng)
    total1 = valueOfTree(k1)
    total2 = valueOfTree(k2)
    ntotal = valueOfTreeOfNX(nt)
    if total1!=total2:
        raise Exception("Los valores de tus MST con y sin CC no coinciden.")
    if total2!=ntotal:
        raise Exception("El valor de tu MST y el de Networkx no coinciden.")
    return True
    
def comprobar_kruskal():
    total = 0
    conexos = 0
    print("Comprobación de Kruskal (solo comprobamos que los pesos de los MST coincidan):\n***")
    for fl_diag in [True, False]:
        print("---\nCon los parámetros:\n- num_max_multiple_edges={}\n- decimals={}\n- max_weight={}\n- fl_diag={}".format(num_max_multiple_edges, decimals, max_weight, fl_diag))
        for n_nodes in range(5,35, 5):
            prob = 0.2
            while prob<=0.8:
                for _ in range(reps):
                    g = lib.rand_weighted_undirected_multigraph(n_nodes,
                                    prob=prob,
                                    num_max_multiple_edges=num_max_multiple_edges, 
                                    max_weight=max_weight, 
                                    decimals=decimals, 
                                    fl_unweighted=False, 
                                    fl_diag=fl_diag)
                    esConexo = comprobar_kruskal_g(g)
                    if esConexo: conexos+=1
                    total+=1
                prob+=0.2
        print("---")
    print("Total de grafos probados {}\nTotal de grafos conexos {}\n".format(total, conexos))




    pass



comprobar_kruskal()

print("\n\n\n")
comprobar_puntos_de_articulacion()

