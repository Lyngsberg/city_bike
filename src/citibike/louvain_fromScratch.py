import random
import numpy as np
import scipy.sparse as sp


def directed_to_undirected(graph):
    #som navnet siger, omdanner fra directed til undirected (vægtet)
    # Deprecated as we build a undirected graph another way
    undirected = {u: {} for u in graph}
    for u in graph:
        for v,w in graph[u].items():
            if u != v:
                if v not in undirected:
                    undirected[v] = {}
                undirected[u][v] = undirected[u].get(v, 0) + w
                undirected[v][u] = undirected[v].get(u, 0) + w
    return undirected


def modularity(graph, communities):
    # Her udregner vi modularitet (for en vægtet undirected graf)
    # inside er den totale vægt af edges inde i community c
    # degree er summen af alle nodernes grader i community c

    # her af kommer også at two_m er summen af alle nodernes grader (i hele grafen)
    # v er naboen til u
    two_m = sum(sum(v.values()) for v in graph.values())
    inside, degree = {}, {}
    for u,v in graph.items():
        c = communities[u]
        degree[c] = degree.get(c,0) + sum(v.values())
        inside[c] = inside.get(c, 0) + sum(w for v, w in v.items() if communities[v] == c)
    Q = sum(inside[c] / two_m - (degree[c] / two_m) ** 2 for c in inside)
    return Q


def phase_one(graph, order):
    degree = {u: sum(v.values()) for u,v in graph.items()}
    two_m = sum(degree.values())
    community = {u: u for u in graph}
    totalDegree = dict(degree) 

    changed = True
    while changed:
        changed = False
        for u in order:
            old = community[u]
            k_in = {}
            for v,w in graph[u].items():
                if v != u:
                    k_in[community[v]] = k_in.get(community[v], 0) + w


            totalDegree[old] -= degree[u]
            best = old
            best_gain = k_in.get(old, 0) * two_m - totalDegree[old] * degree[u]
            for c in k_in:
                gain = k_in[c] * two_m - totalDegree[c] * degree[u]
                if gain > best_gain:
                    best = c
                    best_gain = gain

            totalDegree[best] += degree[u]
            community[u] = best
            if best != old:
                changed = True
    return community


def phase_two(graph, community):
    new_graph = {}
    for u in graph:
        cu = community[u]
        if cu not in new_graph:
            new_graph[cu] = {}
        for v in graph[u]:
            cv = community[v]
            new_graph[cu][cv] = new_graph[cu].get(cv, 0) + graph[u][v]
    return new_graph


def louvain(graph, seed=0):
    rng = random.Random(seed)
    node_to_community = {u: u for u in graph}

    while True:
        order = list(graph)
        if seed is not None:
            rng.shuffle(order)
        community = phase_one(graph, order)
        if len(set(community.values())) == len(graph): #Stopper når vi ikke ændrer noget
            break
        node_to_community = {u: community[c] for u,c in node_to_community.items()}
        graph = phase_two(graph, community)

    sizes = {}
    for c in node_to_community.values():
        sizes[c] = sizes.get(c, 0) + 1
    new_label = {c: i for i, c in enumerate(sorted(sizes, key=sizes.get, reverse=True))}

    result = {u: new_label[c] for u,c in node_to_community.items()}
    return result

# The two functions below are needed to compare with the package implementation of Louvain
def from_sparse(A):
    # Converts the scipy matrix (StationGraph.A) to our dict-of-dicts format
    A = sp.csr_matrix(A)
    return {i: dict(zip(A.indices[A.indptr[i]:A.indptr[i+1]].tolist(),
                        A.data[A.indptr[i]:A.indptr[i+1]].tolist()))
            for i in range(A.shape[0])}


def louvain_own(A, seed=0):
    # Same input/output as graph.louvain: scipy matrix in, numpy array of labels out
    labels = louvain(from_sparse(A), seed=seed)
    return np.array([labels[i] for i in range(A.shape[0])])