"""
algorithms.py
-------------
Implementations of Dijkstra's shortest-path algorithm, Prim's MST algorithm,
and the Bellman-Ford shortest-path algorithm, operating on the Graph class
from graph.py.

Each function optionally records a `trace`: a list of snapshots describing
the algorithm's state after each major step, so that visualisation.py can
render step-by-step diagrams of the shortest-path tree / MST as it is built.
"""

import heapq


INF = float("inf")


# ======================================================================
# Dijkstra's algorithm
# ======================================================================
def dijkstra(graph, source, record_trace=False):
    """
    Single-source shortest paths with non-negative edge weights.

    Data structure: binary min-heap (Python's heapq) as the priority queue.

    Time complexity:  O((V + E) log V)   with a binary heap
    Space complexity: O(V + E)

    Returns (dist, prev, trace) where trace is a list of dicts describing
    each vertex extraction step (empty list if record_trace=False).
    """
    dist = {v: INF for v in graph.vertices()}
    prev = {v: None for v in graph.vertices()}
    dist[source] = 0
    visited = set()
    pq = [(0, source)]
    trace = []

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        step = {"extracted": u, "dist_extracted": d, "relaxed_edges": []}

        for v, w in graph.neighbours(u):
            if v in visited:
                continue
            new_dist = d + w
            if new_dist < dist[v]:
                old = dist[v]
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))
                if record_trace:
                    step["relaxed_edges"].append((u, v, old, new_dist))

        if record_trace:
            step["dist_snapshot"] = dict(dist)
            step["tree_edges"] = [(prev[x], x) for x in graph.vertices() if prev[x] is not None]
            trace.append(step)

    return dist, prev, trace


def dijkstra_path(prev, target):
    """Reconstruct path to target from Dijkstra's predecessor map."""
    path = []
    while target is not None:
        path.append(target)
        target = prev[target]
    return list(reversed(path))


# ======================================================================
# Prim's algorithm (Minimum Spanning Tree, undirected)
# ======================================================================
def prim(graph, source=None, record_trace=False):
    """
    Minimum Spanning Tree via Prim's algorithm.  `graph` is expected to be
    undirected (use Graph.as_undirected_copy() if built from a directed
    transportation network first).

    Data structure: binary min-heap keyed on the cheapest known edge
    connecting the growing tree to each outside vertex ("lazy" Prim).

    Time complexity:  O(E log V)   with a binary heap (lazy version)
    Space complexity: O(V + E)

    Returns (mst_edges, total_weight, trace).
    """
    vertices = graph.vertices()
    if not vertices:
        return [], 0, []
    if source is None:
        source = vertices[0]

    visited = set([source])
    mst_edges = []
    total_weight = 0
    trace = []

    pq = []
    for v, w in graph.neighbours(source):
        heapq.heappush(pq, (w, source, v))

    while pq and len(visited) < len(vertices):
        w, u, v = heapq.heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        total_weight += w

        if record_trace:
            trace.append({
                "added_edge": (u, v, w),
                "tree_vertices": set(visited),
                "tree_edges": list(mst_edges),
                "total_weight": total_weight,
            })

        for nv, nw in graph.neighbours(v):
            if nv not in visited:
                heapq.heappush(pq, (nw, v, nv))

    return mst_edges, total_weight, trace


# ======================================================================
# Bellman-Ford algorithm
# ======================================================================
def bellman_ford(graph, source, record_trace=False):
    """
    Single-source shortest paths, tolerant of negative edge weights.
    Detects negative-weight cycles reachable from the source.

    Time complexity:  O(V * E)   (V-1 relaxation passes over all E edges,
                                   plus one extra pass to detect a
                                   negative cycle)
    Space complexity: O(V + E)

    Returns (dist, prev, has_negative_cycle, negative_cycle_vertices, trace)
    """
    vertices = graph.vertices()
    edges = graph.edges()
    dist = {v: INF for v in vertices}
    prev = {v: None for v in vertices}
    dist[source] = 0
    trace = []

    for i in range(len(vertices) - 1):
        updated = False
        step = {"pass": i + 1, "relaxed_edges": []}
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
                if record_trace:
                    step["relaxed_edges"].append((u, v, w, dist[v]))
        if record_trace:
            step["dist_snapshot"] = dict(dist)
            trace.append(step)
        if not updated:
            break  # early exit: converged before V-1 passes

    # One more pass to detect negative cycles reachable from source
    negative_cycle_vertices = set()
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_negative_cycle = True
            negative_cycle_vertices.add(v)

    # Expand the set: any vertex reachable from a flagged vertex via the
    # predecessor-free forward edges is also considered "affected" /
    # part of/reachable-from the negative cycle for reporting purposes.
    if has_negative_cycle:
        frontier = list(negative_cycle_vertices)
        seen = set(frontier)
        while frontier:
            x = frontier.pop()
            for v, w in graph.neighbours(x):
                if v not in seen:
                    seen.add(v)
                    frontier.append(v)
        negative_cycle_vertices = seen

    return dist, prev, has_negative_cycle, negative_cycle_vertices, trace
