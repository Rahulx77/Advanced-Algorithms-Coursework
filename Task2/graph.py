"""
graph.py
--------
Graph representation for the city transportation network.

DESIGN CHOICE: Adjacency List (dict-of-dicts)
==============================================
A real transportation network (roads/rail/bus routes connecting suburbs,
depots, interchanges, etc.) is SPARSE: each city/junction typically only
connects to a handful of neighbours, not to every other city.  For a graph
with V vertices and E edges, real road networks satisfy E = O(V), not
O(V^2).

Adjacency list:
    Space:  O(V + E)
    Iterate neighbours of v:  O(deg(v))          -- exactly what Dijkstra,
                                                      Prim and Bellman-Ford's
                                                      per-edge relaxation need
    Check if edge (u,v) exists: O(deg(u)) (or O(1) with the dict-of-dicts
                                             representation used here)

Adjacency matrix (the alternative):
    Space:  O(V^2)               -- wasteful for sparse graphs
    Iterate neighbours of v:  O(V) even though most entries are "no edge"
    Check if edge (u,v) exists: O(1)

For a sparse, city-scale transportation network (thousands of intersections,
each with 2-6 roads), the adjacency list is asymptotically and practically
superior: it uses memory proportional to the number of actual roads, and
every algorithm in this assignment (Dijkstra, Prim, Bellman-Ford) is
naturally expressed as "for each edge out of the current vertex", which the
adjacency list supports directly without wasting time scanning non-existent
edges.  We therefore implement the graph as an adjacency list, using a
dict-of-dicts (`self._adj[u][v] = weight`) which gives O(1) average-case
edge-weight lookup while keeping O(deg(v)) neighbour iteration.

An adjacency matrix would only be preferable if the network were dense
(E close to V^2, e.g. a fully-interconnected small set of hub airports)
or if O(1) worst-case edge-existence queries were the dominant operation.
We include a dense random-graph generator in benchmark.py precisely so we
can measure that trade-off empirically rather than just asserting it.
"""

from collections import defaultdict
import random


class Graph:
    """A weighted directed graph stored as an adjacency list."""

    def __init__(self, directed=True):
        self.directed = directed
        self._adj = defaultdict(dict)   # vertex -> {neighbour: weight}
        self._vertices = set()

    # ---- construction -----------------------------------------------
    def add_vertex(self, v):
        self._vertices.add(v)
        if v not in self._adj:
            self._adj[v] = {}

    def add_edge(self, u, v, weight):
        self.add_vertex(u)
        self.add_vertex(v)
        self._adj[u][v] = weight
        if not self.directed:
            self._adj[v][u] = weight

    # ---- queries -------------------------------------------------------
    def vertices(self):
        return list(self._vertices)

    def neighbours(self, v):
        """Return list of (neighbour, weight) pairs. O(deg(v))."""
        return list(self._adj[v].items())

    def weight(self, u, v):
        return self._adj[u].get(v)

    def edges(self):
        """Return list of (u, v, w) triples."""
        seen = set()
        result = []
        for u in self._adj:
            for v, w in self._adj[u].items():
                if self.directed or (v, u) not in seen:
                    result.append((u, v, w))
                    seen.add((u, v))
        return result

    def num_vertices(self):
        return len(self._vertices)

    def num_edges(self):
        return len(self.edges())

    def as_undirected_copy(self):
        """Return a new undirected Graph with the same vertices/edges
        (used for Prim's MST, which is defined on undirected graphs;
        a transportation network's *road segments* are treated as
        two-way for the purposes of infrastructure-cost MST planning)."""
        g = Graph(directed=False)
        for v in self._vertices:
            g.add_vertex(v)
        for u, v, w in self.edges():
            g.add_edge(u, v, w)
        return g

    def __repr__(self):
        return f"Graph(directed={self.directed}, |V|={self.num_vertices()}, |E|={self.num_edges()})"


# ------------------------------------------------------------------------
# Random graph generators used for benchmarking sparse vs dense behaviour
# ------------------------------------------------------------------------
def random_sparse_graph(n, avg_degree=3, min_w=1, max_w=20, seed=None, allow_negative=False):
    """Generate a connected-ish sparse directed graph: E ~ n * avg_degree."""
    rng = random.Random(seed)
    g = Graph(directed=True)
    nodes = [f"N{i}" for i in range(n)]
    for v in nodes:
        g.add_vertex(v)
    # ensure weak connectivity with a random spanning structure first
    for i in range(1, n):
        j = rng.randint(0, i - 1)
        w = rng.randint(min_w, max_w)
        g.add_edge(nodes[j], nodes[i], w)
    target_edges = n * avg_degree
    while g.num_edges() < target_edges:
        u, v = rng.sample(nodes, 2)
        w = rng.randint(min_w, max_w)
        if allow_negative and rng.random() < 0.1:
            w = -rng.randint(1, max_w // 4 + 1)
        g.add_edge(u, v, w)
    return g


def random_dense_graph(n, density=0.6, min_w=1, max_w=20, seed=None):
    """Generate a dense directed graph: roughly `density` fraction of all
    possible directed edges (n*(n-1)) are present."""
    rng = random.Random(seed)
    g = Graph(directed=True)
    nodes = [f"N{i}" for i in range(n)]
    for v in nodes:
        g.add_vertex(v)
    for i in range(1, n):
        j = rng.randint(0, i - 1)
        g.add_edge(nodes[j], nodes[i], rng.randint(min_w, max_w))
    for u in nodes:
        for v in nodes:
            if u != v and rng.random() < density:
                g.add_edge(u, v, rng.randint(min_w, max_w))
    return g
