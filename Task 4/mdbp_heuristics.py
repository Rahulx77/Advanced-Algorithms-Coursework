"""
Multi-dimensional Bin Packing (MDBP) -- heuristic implementation.

"""

import random
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class Item:
    idx: int
    demand: Tuple[float, ...]  # one value per resource dimension


@dataclass
class Instance:
    items: List[Item]
    capacity: Tuple[float, ...]
    k: int  # number of dimensions

    @property
    def n(self):
        return len(self.items)


def generate_instance(n_items: int, k: int = 3, seed: int = 0,
                       capacity: float = 1.0) -> Instance:
    """Generate a random MDBP instance.

    Each item's demand in each dimension is drawn uniformly from
    (0.05, 0.8) * capacity, which is a standard way to generate
    non-trivial bin-packing benchmark instances (items that are neither
    too small -- trivial to pack -- nor larger than a bin).
    """
    rng = random.Random(seed)
    items = []
    for i in range(n_items):
        demand = tuple(rng.uniform(0.05, 0.8) * capacity for _ in range(k))
        items.append(Item(idx=i, demand=demand))
    return Instance(items=items, capacity=tuple([capacity] * k), k=k)


# ----------------------------------------------------------------------
# Helper functions shared across heuristics
# ----------------------------------------------------------------------

def bin_load(bin_items: List[Item], k: int) -> Tuple[float, ...]:
    load = [0.0] * k
    for it in bin_items:
        for j in range(k):
            load[j] += it.demand[j]
    return tuple(load)


def fits(bin_items: List[Item], item: Item, capacity: Tuple[float, ...]) -> bool:
    load = bin_load(bin_items, len(capacity))
    return all(load[j] + item.demand[j] <= capacity[j] + 1e-9 for j in range(len(capacity)))


def solution_num_bins(bins: List[List[Item]]) -> int:
    return len([b for b in bins if b])


def solution_waste(bins: List[List[Item]], capacity: Tuple[float, ...]) -> float:
    """Total unused capacity summed across dimensions and non-empty bins.
    Lower waste (for a given bin count) means tighter packing."""
    k = len(capacity)
    total_cap = sum(capacity)
    waste = 0.0
    for b in bins:
        if not b:
            continue
        load = bin_load(b, k)
        waste += total_cap - sum(load)
    return waste


def item_sort_key(item: Item) -> float:
    """Sort key for FFD: sum of normalised demands (a simple 'volume'
    proxy that works reasonably well across multiple dimensions)."""
    return sum(item.demand)


# ----------------------------------------------------------------------
# Heuristic 1: Greedy construction (First-Fit-Decreasing, multi-dim)
# ----------------------------------------------------------------------

def greedy_ffd(instance: Instance) -> List[List[Item]]:
    items_sorted = sorted(instance.items, key=item_sort_key, reverse=True)
    bins: List[List[Item]] = []
    for item in items_sorted:
        placed = False
        for b in bins:
            if fits(b, item, instance.capacity):
                b.append(item)
                placed = True
                break
        if not placed:
            bins.append([item])
    return bins


# ----------------------------------------------------------------------
# Heuristic 2: Local search (hill climbing)
# ----------------------------------------------------------------------

def local_search(instance: Instance, initial_bins: List[List[Item]],
                  time_budget: float = 2.0) -> List[List[Item]]:
    """Hill-climbing local search that iteratively improves a greedy solution by 
    redistributing or swapping items to reduce the number of bins and waste.
    """
    bins = [list(b) for b in initial_bins if b]
    start = time.time()
    improved = True

    while improved and (time.time() - start) < time_budget:
        improved = False

        # --- Move A: try to empty bins, starting from the least loaded ---
        bins.sort(key=lambda b: sum(bin_load(b, instance.k)))
        for i, weak_bin in enumerate(bins):
            others = bins[:i] + bins[i + 1:]
            temp_others = [list(b) for b in others]
            relocated_all = True
            for item in weak_bin:
                placed = False
                for b in temp_others:
                    if fits(b, item, instance.capacity):
                        b.append(item)
                        placed = True
                        break
                if not placed:
                    relocated_all = False
                    break
            if relocated_all and weak_bin:
                bins = temp_others
                improved = True
                break  # restart the pass after a structural change

        if improved:
            continue

        # --- Move B: pairwise swaps that reduce waste (same bin count) ---
        n_bins = len(bins)
        for i in range(n_bins):
            for j in range(i + 1, n_bins):
                bi, bj = bins[i], bins[j]
                swapped = False
                for a_idx, a_item in enumerate(bi):
                    for b_idx, b_item in enumerate(bj):
                        if a_item is b_item:
                            continue
                        bi_wo, bj_wo = bi[:a_idx] + bi[a_idx + 1:], bj[:b_idx] + bj[b_idx + 1:]
                        if fits(bi_wo, b_item, instance.capacity) and fits(bj_wo, a_item, instance.capacity):
                            before = solution_waste([bi, bj], instance.capacity)
                            new_bi = bi_wo + [b_item]
                            new_bj = bj_wo + [a_item]
                            after = solution_waste([new_bi, new_bj], instance.capacity)
                            if after < before - 1e-9:
                                bins[i], bins[j] = new_bi, new_bj
                                improved = True
                                swapped = True
                                break
                    if swapped:
                        break
                if swapped:
                    break
            if swapped:
                break
            if time.time() - start > time_budget:
                break

    return [b for b in bins if b]


# ----------------------------------------------------------------------
# Heuristic 3: Simulated Annealing
# ----------------------------------------------------------------------

def sa_objective(assignment: List[int], instance: Instance, n_bins_upper: int) -> float:
    """Falkenauer-style fill objective: reward fewer, fuller bins.
    Larger is better."""
    k = instance.k
    loads = [[0.0] * k for _ in range(n_bins_upper)]
    used = set()
    for item, b in zip(instance.items, assignment):
        for j in range(k):
            loads[b][j] += item.demand[j]
        used.add(b)
    num_used = len(used)
    if num_used == 0:
        return 0.0
    total = 0.0
    for b in used:
        fill_frac = sum(loads[b][j] / instance.capacity[j] for j in range(k)) / k
        total += fill_frac ** 2
    return total / num_used


def assignment_to_bins(assignment: List[int], instance: Instance) -> List[List[Item]]:
    bin_map = {}
    for item, b in zip(instance.items, assignment):
        bin_map.setdefault(b, []).append(item)
    return list(bin_map.values())


def simulated_annealing(instance: Instance, initial_bins: List[List[Item]],
                         time_budget: float = 2.0,
                         T0: float = 1.0, cooling: float = 0.995,
                         seed: int = 0) -> List[List[Item]]:
    rng = random.Random(seed)
    n = instance.n
    k = instance.k

    # Initial assignment from the greedy/local-search solution
    assignment = [0] * n
    for b_idx, b in enumerate(initial_bins):
        for item in b:
            assignment[item.idx] = b_idx

    n_bins_upper = len(initial_bins) + 5  # allow SA some room to open new bins
    loads = [[0.0] * k for _ in range(n_bins_upper)]
    for item, b in zip(instance.items, assignment):
        for j in range(k):
            loads[b][j] += item.demand[j]

    def current_score():
        return sa_objective(assignment, instance, n_bins_upper)

    best_assignment = list(assignment)
    best_score = current_score()
    score = best_score

    T = T0
    start = time.time()
    while time.time() - start < time_budget and T > 1e-4:
        for _ in range(200):  # moves per temperature step
            i = rng.randrange(n)
            old_b = assignment[i]
            new_b = rng.randrange(n_bins_upper)
            if new_b == old_b:
                continue
            item = instance.items[i]
            if all(loads[new_b][j] + item.demand[j] <= instance.capacity[j] + 1e-9 for j in range(k)):
                for j in range(k):
                    loads[old_b][j] -= item.demand[j]
                    loads[new_b][j] += item.demand[j]
                assignment[i] = new_b
                new_score = current_score()
                delta = new_score - score
                if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-6)):
                    score = new_score
                    if score > best_score:
                        best_score = score
                        best_assignment = list(assignment)
                else:
                    for j in range(k):
                        loads[old_b][j] += item.demand[j]
                        loads[new_b][j] -= item.demand[j]
                    assignment[i] = old_b
        T *= cooling

    return assignment_to_bins(best_assignment, instance)


# ----------------------------------------------------------------------
# Evaluation harness
# ----------------------------------------------------------------------

def evaluate(instance: Instance, time_budget_local: float = 1.5,
             time_budget_sa: float = 1.5, seed: int = 0) -> dict:
    results = {}

    t0 = time.time()
    greedy_bins = greedy_ffd(instance)
    t1 = time.time()
    results['Greedy (FFD)'] = {
        'bins': solution_num_bins(greedy_bins),
        'waste': round(solution_waste(greedy_bins, instance.capacity), 3),
        'time_s': round(t1 - t0, 4),
    }

    t0 = time.time()
    ls_bins = local_search(instance, greedy_bins, time_budget=time_budget_local)
    t1 = time.time()
    results['Local Search'] = {
        'bins': solution_num_bins(ls_bins),
        'waste': round(solution_waste(ls_bins, instance.capacity), 3),
        'time_s': round(t1 - t0, 4),
    }

    t0 = time.time()
    sa_bins = simulated_annealing(instance, greedy_bins, time_budget=time_budget_sa, seed=seed)
    t1 = time.time()
    results['Simulated Annealing'] = {
        'bins': solution_num_bins(sa_bins),
        'waste': round(solution_waste(sa_bins, instance.capacity), 3),
        'time_s': round(t1 - t0, 4),
    }

    return results


def print_table(all_results: dict):
    header = f"{'Instance (n items)':20} {'Heuristic':20} {'#Bins':>7} {'Waste':>10} {'Time (s)':>10}"
    print(header)
    print('-' * len(header))
    for inst_name, res in all_results.items():
        for heuristic_name, metrics in res.items():
            print(f"{inst_name:20} {heuristic_name:20} {metrics['bins']:>7} "
                  f"{metrics['waste']:>10} {metrics['time_s']:>10}")
    print('-' * len(header))


def main():
    sizes = [30, 60, 100]
    k = 3  # CPU, RAM, bandwidth
    all_results = {}

    for n in sizes:
        instance = generate_instance(n_items=n, k=k, seed=42)
        res = evaluate(instance, time_budget_local=1.5, time_budget_sa=1.5, seed=42)
        all_results[f"n={n}"] = res

    print_table(all_results)


if __name__ == "__main__":
    main()