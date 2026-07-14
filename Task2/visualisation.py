"""
visualisation.py
-----------------
Produces step-by-step visualisations of:
  1. Dijkstra's algorithm building the shortest-path tree
  2. Prim's algorithm building the MST
  3. Bellman-Ford's relaxation passes (including negative-cycle detection)

All figures are saved as PNG files into the `figures/` directory.
Layout is computed once with networkx (spring layout, fixed seed) so the
same node positions are reused across every panel for visual consistency.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from graph import Graph
from algorithms import dijkstra, prim, bellman_ford
from sample_network import city_network, negative_edge_network, negative_cycle_network

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def _to_nx(graph: Graph):
    G = nx.DiGraph() if graph.directed else nx.Graph()
    G.add_nodes_from(graph.vertices())
    for u, v, w in graph.edges():
        G.add_edge(u, v, weight=w)
    return G


def _layout(graph: Graph):
    G = _to_nx(graph)
    return G, nx.spring_layout(G, seed=42, k=1.4)


# ------------------------------------------------------------------
# Dijkstra step-by-step
# ------------------------------------------------------------------
def draw_dijkstra_steps(graph, source, trace, filename="dijkstra_steps.png"):
    G, pos = _layout(graph)
    n_steps = len(trace)
    cols = 3
    rows = (n_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    visited_so_far = []
    for i, step in enumerate(trace):
        ax = axes[i]
        visited_so_far.append(step["extracted"])
        tree_edges = set(step["tree_edges"])

        edge_colors = []
        edge_widths = []
        for (u, v) in G.edges():
            if (u, v) in tree_edges or (v, u) in tree_edges:
                edge_colors.append("#d62728")
                edge_widths.append(2.6)
            else:
                edge_colors.append("#c8c8c8")
                edge_widths.append(1.0)

        node_colors = []
        for v in G.nodes():
            if v == step["extracted"]:
                node_colors.append("#d62728")
            elif v in visited_so_far:
                node_colors.append("#ff9896")
            else:
                node_colors.append("#aec7e8")

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths,
                                arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.05")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=650, edgecolors="black")
        labels = {v: f"{v}\n{step['dist_snapshot'].get(v, '∞') if step['dist_snapshot'].get(v) != float('inf') else '∞'}"
                  for v in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)

        ax.set_title(f"Step {i+1}: extract '{step['extracted']}' (dist={step['dist_extracted']})", fontsize=10)
        ax.axis("off")

    for j in range(n_steps, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Dijkstra's Algorithm: Shortest-Path Tree Construction from '{source}'", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Prim step-by-step
# ------------------------------------------------------------------
def draw_prim_steps(graph, source, trace, filename="prim_steps.png"):
    G, pos = _layout(graph)
    n_steps = len(trace)
    cols = 3
    rows = (n_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    for i, step in enumerate(trace):
        ax = axes[i]
        tree_edges = set()
        for (u, v, w) in step["tree_edges"]:
            tree_edges.add((u, v))
            tree_edges.add((v, u))

        edge_colors, edge_widths = [], []
        for (u, v) in G.edges():
            if (u, v) in tree_edges:
                edge_colors.append("#2ca02c")
                edge_widths.append(2.6)
            else:
                edge_colors.append("#c8c8c8")
                edge_widths.append(1.0)

        node_colors = ["#2ca02c" if v in step["tree_vertices"] else "#aec7e8" for v in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=650, edgecolors="black")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)

        u, v, w = step["added_edge"]
        ax.set_title(f"Step {i+1}: add edge {u}-{v} (w={w})\nrunning total = {step['total_weight']}", fontsize=10)
        ax.axis("off")

    for j in range(n_steps, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Prim's Algorithm: Minimum Spanning Tree Construction from '{source}'", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Bellman-Ford step-by-step (relaxation passes)
# ------------------------------------------------------------------
def draw_bellman_ford_steps(graph, source, trace, filename="bellman_ford_steps.png",
                             negative_cycle_vertices=None):
    G, pos = _layout(graph)
    n_steps = len(trace)
    cols = min(3, max(1, n_steps))
    rows = (n_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_steps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, step in enumerate(trace):
        ax = axes[i]
        relaxed = set((u, v) for (u, v, w, nd) in step["relaxed_edges"])

        edge_colors, edge_widths = [], []
        for (u, v) in G.edges():
            if (u, v) in relaxed:
                edge_colors.append("#9467bd")
                edge_widths.append(2.6)
            else:
                edge_colors.append("#c8c8c8")
                edge_widths.append(1.0)

        node_colors = []
        for v in G.nodes():
            if negative_cycle_vertices and v in negative_cycle_vertices:
                node_colors.append("#d62728")
            else:
                node_colors.append("#aec7e8")

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths,
                                arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.05")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=650, edgecolors="black")
        labels = {v: f"{v}\n{step['dist_snapshot'].get(v) if step['dist_snapshot'].get(v) != float('inf') else '∞'}"
                  for v in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)

        ax.set_title(f"Pass {step['pass']}: {len(step['relaxed_edges'])} edge(s) relaxed", fontsize=10)
        ax.axis("off")

    for j in range(n_steps, len(axes)):
        axes[j].axis("off")

    suffix = "  (NEGATIVE CYCLE DETECTED)" if negative_cycle_vertices else ""
    fig.suptitle(f"Bellman-Ford Algorithm: Relaxation Passes from '{source}'{suffix}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def draw_network_overview(graph, filename="network_overview.png", title="City Transportation Network"):
    G, pos = _layout(graph)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888", arrows=True, arrowsize=14,
                            connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#aec7e8", node_size=800, edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    g = city_network()
    (g, title:="Nepal Transportation Network")

    dist, prev, trace = dijkstra(g, "Kathmandu", record_trace=True)
    draw_dijkstra_steps(g, "Kathmandu", trace)

    ug = g.as_undirected_copy()
    mst_edges, total_w, mtrace = prim(ug, source="Kathmandu", record_trace=True)
    draw_prim_steps(ug, "Kathmandu", mtrace)

    gneg = negative_edge_network()
    dist2, prev2, has_cyc, cyc_v, btrace = bellman_ford(gneg, "Kathmandu", record_trace=True)
    draw_bellman_ford_steps(gneg, "Kathmandu", btrace, filename="bellman_ford_negative_edge_steps.png")

    gcyc = negative_cycle_network()
    dist3, prev3, has_cyc3, cyc_v3, btrace3 = bellman_ford(gcyc, "Kathmandu", record_trace=True)
    draw_bellman_ford_steps(gcyc, "Kathmandu", btrace3, filename="bellman_ford_negative_cycle_steps.png",
                             negative_cycle_vertices=cyc_v3)

    print("Figures written to", OUT_DIR)
    print(os.listdir(OUT_DIR))
