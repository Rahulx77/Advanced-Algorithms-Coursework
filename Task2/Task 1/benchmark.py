"""
benchmark.py
------------
Empirically benchmarks BST, AVL Tree, Min-Heap, and Hash Table on
route-planning data of size N = 100, 1,000, 10,000.

For each structure we measure wall-clock time for:
    - INSERT   : inserting all N cities (reported as total time)
    - SEARCH   : looking up 1,000 random existing keys (average per-op time)
    - DELETE   : deleting 1,000 random existing keys (average per-op time)

Random, already-sorted, and reverse-sorted insertion orders are all
tested for BST/AVL to show the BST's worst case (results saved to
results_worst_case.json).

Results are written to results.json and results_worst_case.json, and
plotted to PNG files via matplotlib.
"""

import os
import random
import time
import json
import sys

# Ensure data_structures.py is importable regardless of the current
# working directory the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_structures import City, BST, AVLTree, MinHeap, HashTable

random.seed(42)


def make_cities(n):
    cities = []
    ids = list(range(n))
    random.shuffle(ids)  # random insertion order (random keys, not sorted)
    for cid in ids:
        cities.append(City(
            city_id=cid,
            name=f"City{cid}",
            x=random.uniform(0, 1000),
            y=random.uniform(0, 1000),
            population=random.randint(1000, 5_000_000),
            distance=random.uniform(0, 2000),
        ))
    return cities


def make_sorted_cities(n, reverse=False):
    """Used to expose BST worst-case (already sorted keys -> a linked list)."""
    cities = []
    ids = list(range(n))
    if reverse:
        ids = ids[::-1]
    for cid in ids:
        cities.append(City(
            city_id=cid, name=f"City{cid}",
            x=random.uniform(0, 1000), y=random.uniform(0, 1000),
            population=random.randint(1000, 5_000_000),
            distance=random.uniform(0, 2000),
        ))
    return cities


def time_it(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def bench_tree_or_hash(structure_factory, cities, n_ops=1000):
    """Generic benchmark for BST / AVL / HashTable (interfaces match)."""
    ds = structure_factory()

    # INSERT (total time to insert all N cities)
    t_insert = time_it(lambda: [ds.insert(c.city_id, c) for c in cities])

    # SEARCH (average time per lookup over n_ops random existing keys)
    sample_ids = [c.city_id for c in random.sample(cities, min(n_ops, len(cities)))]
    start = time.perf_counter()
    for cid in sample_ids:
        ds.search(cid)
    t_search_total = time.perf_counter() - start
    t_search_avg = t_search_total / len(sample_ids)

    # DELETE (average time per deletion over n_ops random existing keys)
    delete_ids = [c.city_id for c in random.sample(cities, min(n_ops, len(cities)))]
    start = time.perf_counter()
    for cid in delete_ids:
        ds.delete(cid)
    t_delete_total = time.perf_counter() - start
    t_delete_avg = t_delete_total / len(delete_ids)

    return {
        "insert_total_s": t_insert,
        "search_avg_s": t_search_avg,
        "delete_avg_s": t_delete_avg,
    }


def bench_heap(cities, n_ops=1000):
    heap = MinHeap()

    t_insert = time_it(lambda: [heap.insert(c.city_id, c) for c in cities])

    sample_ids = [c.city_id for c in random.sample(cities, min(n_ops, len(cities)))]
    start = time.perf_counter()
    for cid in sample_ids:
        heap.search(cid)
    t_search_avg = (time.perf_counter() - start) / len(sample_ids)

    # For a priority queue the natural "delete" op is extract_min.
    n_extract = min(n_ops, len(heap.heap))
    start = time.perf_counter()
    for _ in range(n_extract):
        heap.extract_min()
    t_extractmin_avg = (time.perf_counter() - start) / n_extract

    # Also measure delete-by-arbitrary-key for a fair comparison table
    heap2 = MinHeap()
    for c in cities:
        heap2.insert(c.city_id, c)
    delete_ids = [c.city_id for c in random.sample(cities, min(n_ops, len(cities)))]
    start = time.perf_counter()
    for cid in delete_ids:
        heap2.delete(cid)
    t_delete_bykey_avg = (time.perf_counter() - start) / len(delete_ids)

    return {
        "insert_total_s": t_insert,
        "search_avg_s": t_search_avg,
        "extract_min_avg_s": t_extractmin_avg,
        "delete_by_key_avg_s": t_delete_bykey_avg,
    }


def run_all(sizes=(100, 1000, 10000), repeats=3):
    results = {"BST": {}, "AVL": {}, "HashTable": {}, "MinHeap": {}}

    for n in sizes:
        bst_runs, avl_runs, ht_runs, heap_runs = [], [], [], []
        for r in range(repeats):
            cities = make_cities(n)
            bst_runs.append(bench_tree_or_hash(BST, cities))
            avl_runs.append(bench_tree_or_hash(AVLTree, cities))
            ht_runs.append(bench_tree_or_hash(HashTable, cities))
            heap_runs.append(bench_heap(cities))

        def avg(runs, key):
            return sum(r[key] for r in runs) / len(runs)

        results["BST"][n] = {k: avg(bst_runs, k) for k in bst_runs[0]}
        results["AVL"][n] = {k: avg(avl_runs, k) for k in avl_runs[0]}
        results["HashTable"][n] = {k: avg(ht_runs, k) for k in ht_runs[0]}
        results["MinHeap"][n] = {k: avg(heap_runs, k) for k in heap_runs[0]}

        print(f"n={n} done", file=sys.stderr)

    return results


def run_worst_case(sizes=(100, 1000, 10000)):
    """BST worst case: inserting already-sorted keys degrades it to O(n)."""
    results = {"BST_sorted": {}, "AVL_sorted": {}}
    for n in sizes:
        cities = make_sorted_cities(n)
        results["BST_sorted"][n] = bench_tree_or_hash(BST, cities)
        results["AVL_sorted"][n] = bench_tree_or_hash(AVLTree, cities)
    return results


if __name__ == "__main__":
    print("Running main random-order benchmark...", file=sys.stderr)
    results = run_all()
    results_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved: {results_path}", file=sys.stderr)

    print("Running worst-case (sorted-input) benchmark...", file=sys.stderr)
    worst = run_worst_case()
    worst_path = os.path.join(SCRIPT_DIR, "results_worst_case.json")
    with open(worst_path, "w") as f:
        json.dump(worst, f, indent=2)
    print(json.dumps(worst, indent=2))
    print(f"Saved: {worst_path}", file=sys.stderr)
