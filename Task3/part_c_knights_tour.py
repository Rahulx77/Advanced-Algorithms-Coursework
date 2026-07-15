"""
Part C - Backtracking
Knight's Tour (open tour) with Warnsdorff pruning
-------------------------------------------------------------------
Given an n x n board and a starting square, find a sequence of legal
knight moves visiting every square exactly once.

This file contains:
  1. knights_tour_pruned()   -> backtracking + Warnsdorff heuristic
                                 + dead-square forward checking
  2. knights_tour_unpruned() -> plain backtracking, fixed move order,
                                 no heuristic (baseline for comparison)
  3. a call counter injected into both, to measure how many recursive
     calls each makes on the same board -- demonstrating empirically
     how much the pruning reduces the explored search space
  4. a __main__ block that runs both, prints the tour, and reports
     the call counts
"""

MOVES = [(1, 2), (2, 1), (2, -1), (1, -2),
         (-1, -2), (-2, -1), (-2, 1), (-1, 2)]


# ---------------------------------------------------------------------
# 1. Pruned version: Warnsdorff heuristic + dead-square forward checking
# ---------------------------------------------------------------------
def knights_tour_pruned(n, start=(0, 0), count_calls=False):
    board = [[-1] * n for _ in range(n)]
    sr, sc = start
    board[sr][sc] = 0
    calls = [0]

    def on_board(r, c):
        return 0 <= r < n and 0 <= c < n

    def degree(r, c):
        """Number of unvisited squares reachable from (r, c)."""
        d = 0
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if on_board(nr, nc) and board[nr][nc] == -1:
                d += 1
        return d

    def local_degree_treating_current_as_available(r2, c2, moved_r, moved_c):
        """
        Like degree(), but the square the knight currently occupies
        (moved_r, moved_c) is still counted as a valid source for r2,c2
        -- because the *very next* move is allowed to jump straight from
        the current square into r2,c2. Without this the check below
        produces false positives (it would "kill" every neighbour of the
        square you just moved to, since that square was just flipped
        from unvisited to visited).
        """
        d = 0
        for dr, dc in MOVES:
            rr, cc = r2 + dr, c2 + dc
            if not on_board(rr, cc):
                continue
            if (rr, cc) == (moved_r, moved_c) or board[rr][cc] == -1:
                d += 1
        return d

    def creates_dead_square(moved_r, moved_c):
        """
        Forward-checking pruning, done cheaply and correctly: marking
        (moved_r, moved_c) as visited can only reduce the onward-degree
        of squares that are themselves a knight's move away from it (no
        other square's degree is affected by this move). So instead of
        rescanning the whole board -- O(n^2) per candidate, and easy to
        get subtly wrong -- we only re-check those <= 8 local squares.
        If any unvisited neighbour among them has been left with zero
        remaining ways to ever be entered, this branch can never
        complete a full tour, so it is pruned immediately. This is O(1)
        extra work (at most 8x8 degree checks) per candidate move.
        """
        for dr, dc in MOVES:
            r2, c2 = moved_r + dr, moved_c + dc
            if (on_board(r2, c2) and board[r2][c2] == -1
                    and local_degree_treating_current_as_available(r2, c2, moved_r, moved_c) == 0):
                return True
        return False

    def solve(r, c, move_idx):
        calls[0] += 1
        if move_idx == n * n:
            return True

        candidates = []
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if on_board(nr, nc) and board[nr][nc] == -1:
                candidates.append((degree(nr, nc), nr, nc))
        candidates.sort()  # Warnsdorff: try lowest-degree squares first

        for _, nr, nc in candidates:
            board[nr][nc] = move_idx
            dead = creates_dead_square(nr, nc)
            if not dead and solve(nr, nc, move_idx + 1):
                return True
            board[nr][nc] = -1  # backtrack
        return False

    success = solve(sr, sc, 1)
    result = board if success else None
    return (result, calls[0]) if count_calls else result


# ---------------------------------------------------------------------
# 2. Unpruned baseline: fixed move order, no heuristic
# ---------------------------------------------------------------------
def knights_tour_unpruned(n, start=(0, 0), count_calls=False, call_limit=2_000_000):
    board = [[-1] * n for _ in range(n)]
    sr, sc = start
    board[sr][sc] = 0
    calls = [0]

    def on_board(r, c):
        return 0 <= r < n and 0 <= c < n

    def solve(r, c, move_idx):
        calls[0] += 1
        if calls[0] > call_limit:          # safety cap for the report/demo
            raise RuntimeError("call_limit exceeded (unpruned search too slow)")
        if move_idx == n * n:
            return True
        for dr, dc in MOVES:               # fixed order, no ordering heuristic
            nr, nc = r + dr, c + dc
            if on_board(nr, nc) and board[nr][nc] == -1:
                board[nr][nc] = move_idx
                if solve(nr, nc, move_idx + 1):
                    return True
                board[nr][nc] = -1          # backtrack
        return False

    try:
        success = solve(sr, sc, 1)
    except RuntimeError:
        return (None, calls[0]) if count_calls else None
    result = board if success else None
    return (result, calls[0]) if count_calls else result


# ---------------------------------------------------------------------
# 2b. Validator -- confirms a returned board really is a legal tour
# ---------------------------------------------------------------------
def is_valid_tour(board):
    n = len(board)
    seen = {}
    for r in range(n):
        for c in range(n):
            v = board[r][c]
            if v == -1 or v in seen:
                return False
            seen[v] = (r, c)
    if sorted(seen.keys()) != list(range(n * n)):
        return False
    for k in range(n * n - 1):
        r1, c1 = seen[k]
        r2, c2 = seen[k + 1]
        if (abs(r1 - r2), abs(c1 - c2)) not in {(1, 2), (2, 1)}:
            return False
    return True


# ---------------------------------------------------------------------
# 3. Pretty printer
# ---------------------------------------------------------------------
def print_board(board):
    n = len(board)
    width = len(str(n * n - 1))
    for row in board:
        print(" ".join(f"{v:{width}d}" for v in row))


# ---------------------------------------------------------------------
# 4. Demonstration
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("8x8 board, pruned (Warnsdorff) search")
    print("=" * 60)
    board, calls = knights_tour_pruned(8, start=(0, 0), count_calls=True)
    if board:
        print_board(board)
        print(f"\nValid tour? {is_valid_tour(board)}")
    else:
        print("No tour found.")
    print(f"Recursive calls made (pruned):   {calls}")

    print("\n" + "=" * 60)
    print("8x8 board, unpruned baseline search (fixed move order)")
    print("=" * 60)
    board_u, calls_u = knights_tour_unpruned(8, start=(0, 0), count_calls=True)
    if board_u:
        print_board(board_u)
        print(f"\nRecursive calls made (unpruned): {calls_u}")
    else:
        print(f"Unpruned search did not finish within the call cap "
              f"({calls_u} calls made before giving up) -- this is expected:")
        print("it demonstrates the exponential blow-up the pruning avoids.")

    print("\n" + "=" * 60)
    print("Smaller 5x5 board: compare pruned vs unpruned call counts directly")
    print("=" * 60)
    _, calls_p5 = knights_tour_pruned(5, start=(0, 0), count_calls=True)
    _, calls_u5 = knights_tour_unpruned(5, start=(0, 0), count_calls=True)
    print(f"Pruned calls:   {calls_p5}")
    print(f"Unpruned calls: {calls_u5}")
    if calls_u5:
        print(f"Reduction factor: {calls_u5 / max(calls_p5, 1):.1f}x fewer calls with pruning")
