import os
import json
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "results.json")) as f:
    r = json.load(f)
with open(os.path.join(SCRIPT_DIR, "results_worst_case.json")) as f:
    w = json.load(f)

sizes = [100, 1000, 10000]

# ---------------------------------------------------------------------
# Figure 1: Insert / Search / Delete (average-case, random input) - 3 subplots
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

structures = {
    "BST": r["BST"],
    "AVL": r["AVL"],
    "HashTable": r["HashTable"],
    "MinHeap": r["MinHeap"],
}

metrics = [("insert_total_s", "Total Insert Time (s) -- N items", axes[0]),
           ("search_avg_s", "Average Search Time per Op (s)", axes[1]),
           ("delete_avg_s", "Average Delete Time per Op (s)", axes[2])]

for key, title, ax in metrics:
    for name, data in structures.items():
        ys = []
        for n in sizes:
            entry = data[str(n)]
            if key in entry:
                ys.append(entry[key])
            elif key == "delete_avg_s" and "delete_by_key_avg_s" in entry:
                ys.append(entry["delete_by_key_avg_s"])
            else:
                ys.append(None)
        if all(y is not None for y in ys):
            ax.plot(sizes, ys, marker='o', label=name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Cities (N)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig1_average_case_performance.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: Min-Heap specific -- extract_min vs insert vs search
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
heap = r["MinHeap"]
for key, label in [("insert_total_s", "Insert (total for N)"),
                    ("extract_min_avg_s", "Extract-Min (avg per op)"),
                    ("delete_by_key_avg_s", "Delete-by-key (avg per op)"),
                    ("search_avg_s", "Search-by-key (avg per op)")]:
    ys = [heap[str(n)][key] for n in sizes]
    ax.plot(sizes, ys, marker='o', label=label)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of Cities (N)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Min-Heap: Operation Costs vs N')
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig2_minheap_operations.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Figure 3: BST worst case (sorted input) vs AVL worst case (sorted input)
#           vs BST average case (random input) -- shows O(n) vs O(log n)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Search time
ax = axes[0]
ax.plot(sizes, [r["BST"][str(n)]["search_avg_s"] for n in sizes],
        marker='o', label="BST (random insert order)")
ax.plot(sizes, [w["BST_sorted"][str(n)]["search_avg_s"] for n in sizes],
        marker='s', label="BST (sorted insert order = worst case)")
ax.plot(sizes, [r["AVL"][str(n)]["search_avg_s"] for n in sizes],
        marker='^', label="AVL (random insert order)")
ax.plot(sizes, [w["AVL_sorted"][str(n)]["search_avg_s"] for n in sizes],
        marker='d', label="AVL (sorted insert order)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of Cities (N)')
ax.set_ylabel('Average Search Time (s)')
ax.set_title('BST Worst Case vs AVL: Search Time')
ax.legend(fontsize=8)
ax.grid(True, which="both", ls="--", alpha=0.4)

# Insert time
ax = axes[1]
ax.plot(sizes, [r["BST"][str(n)]["insert_total_s"] for n in sizes],
        marker='o', label="BST (random insert order)")
ax.plot(sizes, [w["BST_sorted"][str(n)]["insert_total_s"] for n in sizes],
        marker='s', label="BST (sorted insert order = worst case)")
ax.plot(sizes, [r["AVL"][str(n)]["insert_total_s"] for n in sizes],
        marker='^', label="AVL (random insert order)")
ax.plot(sizes, [w["AVL_sorted"][str(n)]["insert_total_s"] for n in sizes],
        marker='d', label="AVL (sorted insert order)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of Cities (N)')
ax.set_ylabel('Total Insert Time (s)')
ax.set_title('BST Worst Case vs AVL: Insert Time')
ax.legend(fontsize=8)
ax.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig3_worst_case_comparison.png"), dpi=150)
plt.close()

print(f"Saved figures to: {SCRIPT_DIR}")
print(" - fig1_average_case_performance.png\n - fig2_minheap_operations.png\n - fig3_worst_case_comparison.png")
