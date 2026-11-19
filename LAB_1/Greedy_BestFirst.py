import heapq
import time

# 8-Puzzle

goal = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)

def get_neighbors(state):
    zero = state.index(0)
    row, col = divmod(zero, 3)

    moves = [(-1, 0), (1, 0),
             (0, -1), (0, 1)]

    for dr, dc in moves:
        nr = row + dr
        nc = col + dc

        if 0 <= nr < 3 and 0 <= nc < 3:
            pos = nr * 3 + nc
            new_state = list(state)
            new_state[zero], new_state[pos] = \
                new_state[pos], new_state[zero]

            yield tuple(new_state)

def misplaced_tiles(state):
    count = 0

    for i in range(9):
        if state[i] != 0 and state[i] != goal[i]:
            count += 1

    return count

def greedy(start, goal, neighbors, heuristic):
    start_time = time.perf_counter()

    pq = [(heuristic(start), start)]
    parent = {start: None}
    visited = {start}
    expanded = 0

    while pq:
        _, current = heapq.heappop(pq)
        expanded += 1

        if current == goal:
            break

        for nxt in neighbors(current):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                heapq.heappush(
                    pq,
                    (heuristic(nxt), nxt)
                )

    if goal not in parent:
        return None, expanded, 0

    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    elapsed = (time.perf_counter() - start_time) * 1000
    return path, expanded, elapsed

# Missionaries and Cannibals

start_mc = (3, 3, "L")
goal_mc = (0, 0, "R")

def valid_state(state):
    m_left, c_left, boat = state
    m_right = 3 - m_left
    c_right = 3 - c_left

    if not (0 <= m_left <= 3 and 0 <= c_left <= 3):
        return False

    if m_left != 0 and m_left < c_left:
        return False

    if m_right != 0 and m_right < c_right:
        return False

    return True

def mc_neighbors(state):
    m_left, c_left, boat = state

    moves = [
        (1, 0),
        (2, 0),
        (0, 1),
        (0, 2),
        (1, 1)
    ]

    for m, c in moves:
        if boat == "L":
            new_state = (
                m_left - m,
                c_left - c,
                "R"
            )
        else:
            new_state = (
                m_left + m,
                c_left + c,
                "L"
            )

        if valid_state(new_state):
            yield new_state

def mc_heuristic(state):
    people_left = state[0] + state[1]
    return (people_left + 1) // 2

if __name__ == "__main__":
    start_8 = (1, 3, 6,
               5, 0, 2,
               4, 7, 8)

    path, expanded, elapsed = greedy(
        start_8, goal, get_neighbors, misplaced_tiles
    )

    print("Greedy Best-First : 8 Puzzle")
    print("Heuristic        : Misplaced Tiles")

    if path:
        print("Solution found")
        print("Solution cost :", len(path) - 1)
        print("Nodes expanded:", expanded)
        print("Time          :", round(elapsed, 4), "ms")
    else:
        print("No solution found")

    path, expanded, elapsed = greedy(
        start_mc, goal_mc, mc_neighbors, mc_heuristic
    )

    print("\nGreedy Best-First : Missionaries and Cannibals")

    if path:
        print("Solution found")
        print("Solution cost :", len(path) - 1)
        print("Nodes expanded:", expanded)
        print("Time          :", round(elapsed, 4), "ms")
    else:
        print("No solution found")