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

def minimax(state, depth=0):

    # If terminal state, return utility and path
    if is_terminal(state):
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