"""
data_structures.py
------------------
Implements four data structures for a route-planning application:

    1. Binary Search Tree (BST)          - unbalanced
    2. AVL Tree                          - self-balancing BST
    3. Min-Heap                          - array-based priority queue
    4. Hash Table                        - separate chaining

Each city record stores: city_id (key), name, coordinates (x, y),
population, and distance (distance from a reference depot, used as the
priority key for the heap).

All structures expose a common style of interface:
    insert(key, data)
    search(key)          -> data or None
    delete(key)          -> True/False

so they can be benchmarked identically in benchmark.py.
"""

from dataclasses import dataclass

@dataclass
class City:
    city_id: int
    name: str
    x: float
    y: float
    population: int
    distance: float  # distance from depot -- used as priority in the heap


# ---------------------------------------------------------------------------
# 1. BINARY SEARCH TREE (unbalanced)
# ---------------------------------------------------------------------------
class BSTNode:
    __slots__ = ("key", "data", "left", "right")

    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, key, data):
        """Iterative insert to avoid recursion depth issues on large N."""
        if self.root is None:
            self.root = BSTNode(key, data)
            self.size += 1
            return
        node = self.root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = BSTNode(key, data)
                    self.size += 1
                    return
                node = node.left
            elif key > node.key:
                if node.right is None:
                    node.right = BSTNode(key, data)
                    self.size += 1
                    return
                node = node.right
            else:
                node.data = data  # update existing key
                return

    def search(self, key):
        node = self.root
        while node is not None:
            if key == node.key:
                return node.data
            node = node.left if key < node.key else node.right
        return None

    def delete(self, key):
        """Iterative delete (see note on insert() re: recursion depth)."""
        parent, node = None, self.root
        while node is not None and node.key != key:
            parent = node
            node = node.left if key < node.key else node.right

        if node is None:
            return False  # key not found

        # Case: two children -- swap with in-order successor, then
        # continue the loop to physically delete the successor node
        # (which has at most one child) further down the tree.
        if node.left is not None and node.right is not None:
            succ_parent, succ = node, node.right
            while succ.left is not None:
                succ_parent = succ
                succ = succ.left
            node.key, node.data = succ.key, succ.data
            parent, node = succ_parent, succ  # now delete `succ` instead

        # `node` now has at most one child
        child = node.left if node.left is not None else node.right
        if parent is None:
            self.root = child
        elif parent.left is node:
            parent.left = child
        else:
            parent.right = child

        self.size -= 1
        return True


# ---------------------------------------------------------------------------
# 2. AVL TREE (self-balancing BST)
# ---------------------------------------------------------------------------
class AVLNode:
    __slots__ = ("key", "data", "left", "right", "height")

    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    def __init__(self):
        self.root = None
        self.size = 0

    @staticmethod
    def _h(node):
        return node.height if node else 0

    def _update_height(self, node):
        node.height = 1 + max(self._h(node.left), self._h(node.right))

    def _balance_factor(self, node):
        return self._h(node.left) - self._h(node.right) if node else 0

    def _rotate_right(self, y):
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x):
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node):
        self._update_height(node)
        bf = self._balance_factor(node)
        if bf > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if bf < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def insert(self, key, data):
        self.root = self._insert(self.root, key, data)

    def _insert(self, node, key, data):
        if node is None:
            self.size += 1
            return AVLNode(key, data)
        if key < node.key:
            node.left = self._insert(node.left, key, data)
        elif key > node.key:
            node.right = self._insert(node.right, key, data)
        else:
            node.data = data
            return node
        return self._rebalance(node)

    def search(self, key):
        node = self.root
        while node is not None:
            if key == node.key:
                return node.data
            node = node.left if key < node.key else node.right
        return None

    def delete(self, key):
        self.root, deleted = self._delete(self.root, key)
        if deleted:
            self.size -= 1
        return deleted

    def _delete(self, node, key):
        if node is None:
            return node, False
        if key < node.key:
            node.left, deleted = self._delete(node.left, key)
        elif key > node.key:
            node.right, deleted = self._delete(node.right, key)
        else:
            deleted = True
            if node.left is None:
                return node.right, deleted
            if node.right is None:
                return node.left, deleted
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key, node.data = successor.key, successor.data
            node.right, _ = self._delete(node.right, successor.key)
        if node is None:
            return node, deleted
        return self._rebalance(node), deleted


# ---------------------------------------------------------------------------
# 3. MIN-HEAP (array-based priority queue, keyed on `distance`)
# ---------------------------------------------------------------------------
class MinHeap:
    """
    Binary min-heap stored in a Python list.
    Priority = city.distance (nearest city has highest priority).
    Also keeps a dict {city_id: index} so we can support delete-by-key
    and search-by-key in better than pure O(n) *on average*, although
    the theoretical worst case for search/delete-by-key in a heap is
    still O(n) because the heap array is not ordered by key.
    """

    def __init__(self):
        self.heap = []          # list of City objects, ordered by .distance
        self.pos = {}           # city_id -> index in self.heap

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos[self.heap[i].city_id] = i
        self.pos[self.heap[j].city_id] = j

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i].distance < self.heap[parent].distance:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            left, right, smallest = 2 * i + 1, 2 * i + 2, i
            if left < n and self.heap[left].distance < self.heap[smallest].distance:
                smallest = left
            if right < n and self.heap[right].distance < self.heap[smallest].distance:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def insert(self, key, data: City):
        # key is city_id; data is the City object (priority = data.distance)
        self.heap.append(data)
        i = len(self.heap) - 1
        self.pos[data.city_id] = i
        self._sift_up(i)

    def peek_min(self):
        return self.heap[0] if self.heap else None

    def extract_min(self):
        """Removes and returns the highest-priority (nearest) city."""
        if not self.heap:
            return None
        min_city = self.heap[0]
        last = self.heap.pop()
        del self.pos[min_city.city_id]
        if self.heap:
            self.heap[0] = last
            self.pos[last.city_id] = 0
            self._sift_down(0)
        return min_city

    def search(self, key):
        """O(n) worst case: a heap is only partially ordered."""
        idx = self.pos.get(key)
        return self.heap[idx] if idx is not None else None

    def delete(self, key):
        """Delete an arbitrary city by id (not just the min)."""
        idx = self.pos.get(key)
        if idx is None:
            return False
        last_idx = len(self.heap) - 1
        self._swap(idx, last_idx)
        removed = self.heap.pop()
        del self.pos[removed.city_id]
        if idx < len(self.heap):
            self._sift_down(idx)
            self._sift_up(idx)
        return True


# ---------------------------------------------------------------------------
# 4. HASH TABLE (separate chaining)
# ---------------------------------------------------------------------------
class HashTable:
    def __init__(self, capacity=16, load_factor_limit=0.75):
        self.capacity = capacity
        self.load_factor_limit = load_factor_limit
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _hash(self, key):
        return hash(key) % self.capacity

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old_buckets:
            for key, data in bucket:
                self.insert(key, data)

    def insert(self, key, data):
        if (self.size + 1) / self.capacity > self.load_factor_limit:
            self._resize()
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, data)  # update
                return
        bucket.append((key, data))
        self.size += 1

    def search(self, key):
        idx = self._hash(key)
        for k, data in self.buckets[idx]:
            if k == key:
                return data
        return None

    def delete(self, key):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False
