"""
Part B - Greedy Algorithm
Minimum Number of Platforms
-------------------------------------------------------------------
Given arrival and departure times of n trains, find the minimum
number of platforms needed so no two overlapping trains share one.

This file contains:
  1. min_platforms()            -> O(n log n) greedy chronological sweep
  2. min_platforms_brute_force()-> O(n^2) baseline (max overlap by
                                    direct pairwise/point counting),
                                    used only to verify correctness
  3. a __main__ block that runs both on several test cases and checks
     they agree, then prints a worked example with the platform
     assignment made explicit.
"""

import random


# ---------------------------------------------------------------------
# 1. Greedy sweep-line solution -> O(n log n) time, O(n) space
# ---------------------------------------------------------------------
def min_platforms(arrivals, departures):
    """
    arrivals, departures: lists of equal length n; train i has
    arrival[i] and departure[i] (arrival[i] <= departure[i]).
    Returns the minimum number of platforms required.
    """
    n = len(arrivals)
    if n == 0:
        return 0

    arr = sorted(arrivals)
    dep = sorted(departures)

    i = k = 0
    current = max_platforms = 0
    while i < n and k < n:
        if arr[i] <= dep[k]:      # a new train arrives at/before the
            current += 1          # earliest still-pending departure
            max_platforms = max(max_platforms, current)
            i += 1
        else:                      # a platform frees up first
            current -= 1
            k += 1
    return max_platforms


# ---------------------------------------------------------------------
# 2. Brute-force verifier -> O(n^2), only for testing
# ---------------------------------------------------------------------
def min_platforms_brute_force(arrivals, departures):
    """
    For every train's arrival instant, count how many trains (including
    itself) are present at that instant; the answer is the maximum of
    these counts over all n arrival instants (checking at arrival times
    suffices, since the overlap count only changes at event times, and
    is either at, or about to increase at, an arrival instant).
    """
    n = len(arrivals)
    best = 0
    for i in range(n):
        t = arrivals[i]
        count = sum(1 for j in range(n) if arrivals[j] <= t <= departures[j])
        best = max(best, count)
    return best


# ---------------------------------------------------------------------
# 3. Explicit platform assignment (bonus: not just the count)
# ---------------------------------------------------------------------
def assign_platforms(arrivals, departures):
    """
    Returns a list `platform_of[i]` giving the platform number (0-indexed)
    assigned to train i, using a min-heap of platform free-times.
    Demonstrates that the greedy count from min_platforms() is achievable,
    not just a lower bound.
    """
    import heapq

    n = len(arrivals)
    order = sorted(range(n), key=lambda i: arrivals[i])
    free_heap = []          # (free_time, platform_id)
    platform_of = [None] * n
    next_platform_id = 0

    for i in order:
        if free_heap and free_heap[0][0] <= arrivals[i]:
            free_time, pid = heapq.heappop(free_heap)
        else:
            pid = next_platform_id
            next_platform_id += 1
        platform_of[i] = pid
        heapq.heappush(free_heap, (departures[i], pid))

    return platform_of, next_platform_id


# ---------------------------------------------------------------------
# 4. Tests + worked example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    print("Cross-checking greedy sweep against brute force:\n")

    test_cases = [
        ([], []),
        ([900], [910]),
        ([900, 940, 950, 1100, 1500, 1800], [910, 1200, 1120, 1130, 1900, 2000]),
        ([100, 100, 100, 100], [200, 200, 200, 200]),   # all identical -> 4 platforms
        ([100, 200, 300, 400], [150, 250, 350, 450]),   # no overlap -> 1 platform
    ]
    for i, (arr, dep) in enumerate(test_cases, 1):
        greedy = min_platforms(arr, dep)
        brute = min_platforms_brute_force(arr, dep)
        status = "OK" if greedy == brute else "MISMATCH"
        print(f"Test {i}: arrivals={arr}, departures={dep}")
        print(f"  Greedy sweep -> {greedy}")
        print(f"  Brute force  -> {brute}")
        print(f"  Result: {status}\n")
        assert greedy == brute, "Greedy and brute force disagree!"

    # Randomised stress test
    for trial in range(200):
        n = random.randint(0, 12)
        arr = [random.randint(0, 30) for _ in range(n)]
        dep = [a + random.randint(0, 10) for a in arr]
        assert min_platforms(arr, dep) == min_platforms_brute_force(arr, dep), \
            f"Mismatch on trial {trial}: arr={arr}, dep={dep}"
    print("200 randomised stress tests passed. Greedy implementation verified correct.\n")

    # Worked example with explicit platform assignment
    arr = [900, 940, 950, 1100, 1500, 1800]
    dep = [910, 1200, 1120, 1130, 1900, 2000]
    print("Worked example:")
    print("  Arrivals:  ", arr)
    print("  Departures:", dep)
    print("  Minimum platforms required:", min_platforms(arr, dep))
    assignment, num_platforms = assign_platforms(arr, dep)
    print("  Platform assignment (0-indexed):")
    for i, plat in enumerate(assignment):
        print(f"    Train {i} (arr={arr[i]}, dep={dep[i]}) -> platform {plat}")
    print("  Platforms actually used:", num_platforms)
