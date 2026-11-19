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

def depth_limited(current, goal, neighbors,
                  limit, depth, path, visited, count):
    count[0] += 1

    if current == goal:
        return path.copy()

    if depth == limit:
        return None

    for nxt in neighbors(current):
        if nxt in visited:
            continue

        visited.add(nxt)
        path.append(nxt)

        result = depth_limited(
            nxt, goal, neighbors, limit,
            depth + 1, path, visited, count
        )

        if result:
            return result

        path.pop()
        visited.remove(nxt)

    return None

def ids(start, goal, neighbors, max_depth):
    start_time = time.perf_counter()
    total_expanded = 0

    for limit in range(max_depth + 1):
        count = [0]

        path = depth_limited(
            start, goal, neighbors, limit,
            0, [start], {start}, count
        )

        total_expanded += count[0]

        if path:
            elapsed = (time.perf_counter() - start_time) * 1000
            return path, total_expanded, elapsed

    elapsed = (time.perf_counter() - start_time) * 1000
    return None, total_expanded, elapsed

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

if __name__ == "__main__":
    start_8 = (1, 3, 6,
               5, 0, 2,
               4, 7, 8)

    path, expanded, elapsed = ids(
        start_8, goal, get_neighbors, 30
    )

    print("IDS : 8 Puzzle")

    if path:
        print("Solution found")
        print("Solution cost :", len(path) - 1)
        print("Nodes expanded:", expanded)
        print("Time          :", round(elapsed, 4), "ms")
    else:
        print("No solution found")

    path, expanded, elapsed = ids(
        start_mc, goal_mc, mc_neighbors, 20
    )

    print("\nIDS : Missionaries and Cannibals")

    if path:
        print("Solution found")
        print("Solution cost :", len(path) - 1)
        print("Nodes expanded:", expanded)
        print("Time          :", round(elapsed, 4), "ms")
    else:
        print("No solution found")