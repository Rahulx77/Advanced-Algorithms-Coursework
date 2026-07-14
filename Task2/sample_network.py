"""
sample_network.py
------------------
A small, illustrative "Nepal transportation network": 9 hubs connected by
directed weighted edges representing travel time in minutes. Used for the
worked examples, step-by-step traces and diagrams in the report.

A second variant, `negative_edge_network()`, adds a negative-weight edge
(e.g. a toll discount / subsidy modelled as negative cost) to demonstrate
Bellman-Ford's advantage over Dijkstra, and `negative_cycle_network()`
adds a genuine negative cycle to demonstrate cycle detection.
"""

from graph import Graph


def city_network():
    g = Graph(directed=True)
    edges = [
        ("Kathmandu", "Lalitpur", 7),
        ("Kathmandu", "Pokhara", 9),
        ("Kathmandu", "Bharatpur", 14),
        ("Lalitpur", "Pokhara", 10),
        ("Lalitpur", "Biratnagar", 15),
        ("Pokhara", "Bharatpur", 2),
        ("Pokhara", "Biratnagar", 11),
        ("Bharatpur", "Biratnagar", 9),
        ("Bharatpur", "Janakpur", 6),
        ("Biratnagar", "Janakpur", 6),
        ("Biratnagar", "Bhaktapur", 9),
        ("Janakpur", "Bhaktapur", 5),
        ("Janakpur", "Nepalgunj", 8),
        ("Bhaktapur", "Nepalgunj", 8),
        ("Bhaktapur", "Tribhuvan", 10),
        ("Nepalgunj", "Tribhuvan", 7),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


def negative_edge_network():
    """Same topology, but one edge has a negative weight (e.g. a transit
    subsidy that makes a route effectively 'pay you' a small amount) --
    something Dijkstra cannot handle correctly, but Bellman-Ford can."""
    g = city_network()
    g.add_edge("Biratnagar", "Janakpur", -9)  # overrides the +6 edge
    return g


def negative_cycle_network():
    """Introduces a genuine negative cycle: Janakpur -> Bhaktapur ->
    Nepalgunj -> Janakpur sums to a negative total, which should be
    reported by Bellman-Ford as unsafe for shortest-path computation."""
    g = city_network()
    g.add_edge("Nepalgunj", "Janakpur", -25)  # closes a negative cycle
    return g


if __name__ == "__main__":
    g = city_network()
    print(g)
    for u, v, w in sorted(g.edges()):
        print(f"  {u:10s} -> {v:10s}  ({w} min)")
