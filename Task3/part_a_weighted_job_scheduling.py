"""
Part A - Dynamic Programming
Weighted Job Scheduling with Time Windows
-------------------------------------------------------------------
Given n jobs, each with (start, finish, profit), select a subset of
mutually non-overlapping jobs that maximises total profit.

This file contains:
  1. weighted_job_scheduling()  -> the O(n log n) DP solution
  2. brute_force_scheduling()   -> O(2^n) exhaustive search, used only
                                    to verify correctness on small inputs
  3. a __main__ block that runs both on several test cases and checks
     they agree, then prints a worked example.
"""

from bisect import bisect_right
from itertools import combinations


# ---------------------------------------------------------------------
# 1. Dynamic programming solution  ->  O(n log n) time, O(n) space
# ---------------------------------------------------------------------
def weighted_job_scheduling(jobs):
    """
    jobs: list of (start, finish, profit) tuples.
    Returns: (max_profit, chosen_jobs) where chosen_jobs is a list of
             the (start, finish, profit) tuples selected, sorted by start.
    """
    if not jobs:
        return 0, []

    # Step 1: sort by finish time -> O(n log n)
    jobs = sorted(jobs, key=lambda j: j[1])
    n = len(jobs)
    finishes = [j[1] for j in jobs]

    # Step 2: p(j) = latest job index (1-indexed) compatible with job j
    def p(j):
        start = jobs[j - 1][0]
        # rightmost position among finishes[0 .. j-2] that is <= start
        idx = bisect_right(finishes, start, 0, j - 1)
        return idx  # 0 means "no compatible earlier job"

    # Step 3: bottom-up DP table
    OPT = [0] * (n + 1)
    take = [False] * (n + 1)
    for j in range(1, n + 1):
        incl = jobs[j - 1][2] + OPT[p(j)]
        excl = OPT[j - 1]
        if incl > excl:
            OPT[j] = incl
            take[j] = True
        else:
            OPT[j] = excl

    # Step 4: reconstruct the chosen jobs by walking the take[] array back
    chosen = []
    j = n
    while j > 0:
        if take[j]:
            chosen.append(jobs[j - 1])
            j = p(j)
        else:
            j -= 1
    chosen.reverse()
    chosen.sort(key=lambda t: t[0])
    return OPT[n], chosen


# ---------------------------------------------------------------------
# 2. Brute-force verifier -> O(2^n), only for testing on small n
# ---------------------------------------------------------------------
def brute_force_scheduling(jobs):
    def compatible(subset):
        subset = sorted(subset, key=lambda j: j[0])
        return all(subset[i][1] <= subset[i + 1][0] for i in range(len(subset) - 1))

    best_profit, best_subset = 0, []
    n = len(jobs)
    for r in range(n + 1):
        for combo in combinations(jobs, r):
            if compatible(combo):
                profit = sum(j[2] for j in combo)
                if profit > best_profit:
                    best_profit, best_subset = profit, list(combo)
    best_subset.sort(key=lambda t: t[0])
    return best_profit, best_subset


# ---------------------------------------------------------------------
# 3. Tests + worked example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        [],
        [(1, 2, 50)],
        [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4), (5, 8, 11), (7, 9, 2)],
        [(1, 3, 5), (3, 5, 6), (0, 6, 10), (5, 7, 3), (5, 9, 4), (6, 10, 2)],
        [(1, 2, 4), (2, 5, 6), (4, 6, 3), (6, 8, 1), (5, 7, 2), (1, 4, 9)],
    ]

    print("Cross-checking DP against brute force on small test cases:\n")
    for i, jobs in enumerate(test_cases, 1):
        dp_profit, dp_jobs = weighted_job_scheduling(jobs)
        bf_profit, bf_jobs = brute_force_scheduling(jobs)
        status = "OK" if dp_profit == bf_profit else "MISMATCH"
        print(f"Test {i}: jobs={jobs}")
        print(f"  DP     -> profit={dp_profit}, jobs={dp_jobs}")
        print(f"  Brute  -> profit={bf_profit}, jobs={bf_jobs}")
        print(f"  Result: {status}\n")
        assert dp_profit == bf_profit, "DP and brute force disagree!"

    print("All test cases agree. DP implementation verified correct.\n")

    # Worked example for the report
    jobs = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4), (5, 8, 11), (7, 9, 2)]
    profit, chosen = weighted_job_scheduling(jobs)
    print("Worked example:")
    print("  Input jobs (start, finish, profit):", jobs)
    print("  Maximum profit:", profit)
    print("  Jobs chosen:", chosen)
