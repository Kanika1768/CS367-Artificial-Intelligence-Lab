activities = {
    "Coding": 3,
    "Exam Preparation": 3,
    "Research": 2,
    "Sleep": 3,
    "Cooking": 2,
    "Eating Outside": 2,
    "Movie": 3,
    "Games": 2
}

utility_A = {
    "Coding": 5,
    "Exam Preparation": 5,
    "Research": 4,
    "Sleep": 1,
    "Cooking": 2,
    "Eating Outside": 1,
    "Movie": 2,
    "Games": 2
}

utility_B = {
    "Coding": 2,
    "Exam Preparation": 1,
    "Research": 1,
    "Sleep": 4,
    "Cooking": 3,
    "Eating Outside": 4,
    "Movie": 5,
    "Games": 5
}


class State:

    def __init__(self, time_remaining, path, player):
        self.time_remaining = time_remaining
        self.path = path
        self.player = player


def calculate_utility(path):

    score_A = 0
    score_B = 0

    for activity in path:
        score_A += utility_A[activity]
        score_B += utility_B[activity]

    return score_A - score_B


def get_legal_actions(state):

    legal_actions = []

    for activity, duration in activities.items():

        if duration <= state.time_remaining:
            legal_actions.append(activity)

    return legal_actions


def is_terminal(state):

    return len(get_legal_actions(state)) == 0


TOTAL_TIME = 8

initial_state = State(
    TOTAL_TIME,
    [],
    "A"
)


print("Available activities:")
print(get_legal_actions(initial_state))

print("\nIs initial state terminal?")
print(is_terminal(initial_state))

nodes_generated = 0
nodes_evaluated = 0

def minimax(state, depth=0):

    global nodes_generated, nodes_evaluated

    # Count this node
    nodes_generated += 1

    # If terminal state, evaluate it
    if is_terminal(state):
        nodes_evaluated += 1
        return calculate_utility(state.path), state.path

    legal_actions = get_legal_actions(state)

    # A is MAX
    if state.player == "A":

        best_value = float("-inf")
        best_path = None

        for action in legal_actions:

            new_state = State(
                state.time_remaining - activities[action],
                state.path + [action],
                "B"
            )

            value, path = minimax(new_state, depth + 1)

            if value > best_value:
                best_value = value
                best_path = path

        return best_value, best_path

    # B is MIN
    else:

        best_value = float("inf")
        best_path = None

        for action in legal_actions:

            new_state = State(
                state.time_remaining - activities[action],
                state.path + [action],
                "A"
            )

            value, path = minimax(new_state, depth + 1)

            if value < best_value:
                best_value = value
                best_path = path

        return best_value, best_path

# -----------------------------------------
# ALPHA-BETA PRUNING
# -----------------------------------------

alpha_beta_generated = 0
alpha_beta_evaluated = 0
alpha_beta_pruned = 0


def alpha_beta(state, alpha, beta, depth=0):

    global alpha_beta_generated
    global alpha_beta_evaluated
    global alpha_beta_pruned

    # Count current node
    alpha_beta_generated += 1

    # If terminal state, evaluate it
    if is_terminal(state):

        alpha_beta_evaluated += 1

        return calculate_utility(state.path), state.path

    legal_actions = get_legal_actions(state)

    # A is MAX
    if state.player == "A":

        best_value = float("-inf")
        best_path = None

        for action in legal_actions:

            new_state = State(
                state.time_remaining - activities[action],
                state.path + [action],
                "B"
            )

            value, path = alpha_beta(
                new_state,
                alpha,
                beta,
                depth + 1
            )

            if value > best_value:
                best_value = value
                best_path = path

            # Update alpha
            alpha = max(alpha, best_value)

            # Pruning condition
            if alpha >= beta:

                alpha_beta_pruned += 1

                print(
                    "Pruned at A:",
                    " -> ".join(new_state.path),
                    "| alpha =", alpha,
                    "| beta =", beta
                )

                break

        return best_value, best_path

    # B is MIN
    else:

        best_value = float("inf")
        best_path = None

        for action in legal_actions:

            new_state = State(
                state.time_remaining - activities[action],
                state.path + [action],
                "A"
            )

            value, path = alpha_beta(
                new_state,
                alpha,
                beta,
                depth + 1
            )

            if value < best_value:
                best_value = value
                best_path = path

            # Update beta
            beta = min(beta, best_value)

            # Pruning condition
            if alpha >= beta:

                alpha_beta_pruned += 1

                print(
                    "Pruned at B:",
                    " -> ".join(new_state.path),
                    "| alpha =", alpha,
                    "| beta =", beta
                )

                break

        return best_value, best_path

value, path = minimax(initial_state)

print("\n--- MINIMAX RESULT ---")
print("Optimal path:", " -> ".join(path))
print("Final utility:", value)
print("Nodes Generated:", nodes_generated)
print("Nodes Evaluated:", nodes_evaluated)
print("Nodes Pruned: 0")

# -----------------------------------------
# ALPHA-BETA RESULT
# -----------------------------------------

alpha_value, alpha_path = alpha_beta(
    initial_state,
    float("-inf"),
    float("inf")
)

print("\n--- ALPHA-BETA RESULT ---")
print("Optimal path:", " -> ".join(alpha_path))
print("Final utility:", alpha_value)
print("Nodes Generated:", alpha_beta_generated)
print("Nodes Evaluated:", alpha_beta_evaluated)
print("Nodes Pruned:", alpha_beta_pruned)

print("\n--- VERIFICATION ---")
print("Same optimal path:", path == alpha_path)
print("Same final utility:", value == alpha_value)