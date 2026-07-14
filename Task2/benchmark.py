"""
benchmark.py
------------
Empirically compares the wall-clock runtime of Dijkstra, Prim and
Bellman-Ford across a range of graph sizes, for both sparse and dense
graphs, and reports the observed runtime alongside the Big-O prediction
so that the *constant factor* hidden inside the O(...) notation can be
discussed rather than assumed.
"""

import time
import statistics
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph import random_sparse_graph, random_dense_graph
from algorithms import dijkstra, prim, bellman_ford

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

REPEATS = 3  # repetitions per data point, we report the median


def time_it(fn, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times)


def run_benchmark(sizes, graph_kind, avg_degree=3, density=0.5, seed_base=1000):
    """graph_kind in {'sparse', 'dense'}"""
    rows = []
    for n in sizes:
        seed = seed_base + n
        if graph_kind == "sparse":
            g = random_sparse_graph(n, avg_degree=avg_degree, seed=seed)
        else:
            g = random_dense_graph(n, density=density, seed=seed)
        ug = g.as_undirected_copy()
        source = g.vertices()[0]

        t_dij = time_it(lambda: dijkstra(g, source))
        t_prim = time_it(lambda: prim(ug, source=source))
        t_bf = time_it(lambda: bellman_ford(g, source))

        V, E = g.num_vertices(), g.num_edges()
        rows.append({
            "graph_kind": graph_kind, "V": V, "E": E,
            "dijkstra_s": t_dij, "prim_s": t_prim, "bellman_ford_s": t_bf,
            # theoretical operation counts (not time -- for constant-factor analysis)
            "dijkstra_ops_theory": (V + E) * max(1, (V).bit_length()),
            "prim_ops_theory": E * max(1, (V).bit_length()),
            "bellman_ford_ops_theory": V * E,
        })
        print(f"[{graph_kind:6s}] V={V:5d} E={E:6d}  "
              f"Dijkstra={t_dij*1000:8.3f} ms  Prim={t_prim*1000:8.3f} ms  "
              f"Bellman-Ford={t_bf*1000:8.3f} ms")
    return rows


def save_csv(rows, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_runtime_comparison(sparse_rows, dense_rows, filename="runtime_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, rows, title in [(axes[0], sparse_rows, "Sparse graphs (E ≈ 3V)"),
                              (axes[1], dense_rows, "Dense graphs (E ≈ 0.5·V²)")]:
        Vs = [r["V"] for r in rows]
        ax.plot(Vs, [r["dijkstra_s"] * 1000 for r in rows], "o-", label="Dijkstra (heap)", color="#1f77b4")
        ax.plot(Vs, [r["prim_s"] * 1000 for r in rows], "s-", label="Prim (heap)", color="#2ca02c")
        ax.plot(Vs, [r["bellman_ford_s"] * 1000 for r in rows], "^-", label="Bellman-Ford", color="#9467bd")
        ax.set_xlabel("Number of vertices (V)")
        ax.set_ylabel("Median wall-clock time (ms)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Observed Runtime: Dijkstra vs Prim vs Bellman-Ford", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_constant_factor(sparse_rows, dense_rows, filename="constant_factor.png"):
    """Plot observed_time / theoretical_op_count to visualise the constant
    factor for each algorithm -- demonstrating that Big-O order does not
    by itself predict wall-clock time (Bellman-Ford's larger constant
    hidden inside O(VE) vs Dijkstra's O((V+E)logV) with a heap)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, rows, title in [(axes[0], sparse_rows, "Sparse graphs"),
                              (axes[1], dense_rows, "Dense graphs")]:
        Vs = [r["V"] for r in rows]
        for key, ops_key, label, color in [
            ("dijkstra_s", "dijkstra_ops_theory", "Dijkstra: t / ((V+E)logV)", "#1f77b4"),
            ("prim_s", "prim_ops_theory", "Prim: t / (E logV)", "#2ca02c"),
            ("bellman_ford_s", "bellman_ford_ops_theory", "Bellman-Ford: t / (VE)", "#9467bd"),
        ]:
            ratios = [ (r[key] / r[ops_key]) * 1e9 if r[ops_key] else 0 for r in rows]  # ns per theoretical "op"
            ax.plot(Vs, ratios, "o-", label=label, color=color)
        ax.set_xlabel("Number of vertices (V)")
        ax.set_ylabel("ns per theoretical operation")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Empirical Constant Factor (time normalised by Big-O operation count)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    sizes = [20, 40, 80, 160, 320, 500]
    print("=== Sparse graphs (avg out-degree 3) ===")
    sparse_rows = run_benchmark(sizes, "sparse", avg_degree=3)
    print("\n=== Dense graphs (density 0.5) ===")
    dense_sizes = [10, 20, 40, 80, 120, 160]  # smaller: O(V^2) edges + Bellman-Ford O(V*E) grows fast
    dense_rows = run_benchmark(dense_sizes, "dense", density=0.5)

    save_csv(sparse_rows, "benchmark_sparse.csv")
    save_csv(dense_rows, "benchmark_dense.csv")
    plot_runtime_comparison(sparse_rows, dense_rows)
    plot_constant_factor(sparse_rows, dense_rows)
    print("\nDone. CSVs and figures saved in figures/")
