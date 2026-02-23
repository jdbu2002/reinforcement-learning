from .environment import Environment


class ValueIteration:
    def __init__(self, mdp: Environment, discount=0.9, iterations=100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {}

    def run_value_iteration(self):
        for _ in range(self.iterations):
            new_values = {}
            for i in range(self.mdp.nrows):
                for j in range(self.mdp.ncols):
                    state = (i, j)
                    if self.mdp.board[i][j] == self.mdp.PROHIBITED_CELL:
                        new_values[state] = 0
                    else:
                        actions = self.mdp.get_posible_actions(state)
                        if not actions:
                            new_values[state] = 0
                        else:
                            q_values = [
                                self.compute_qvalue_from_values(state, action)
                                for action in actions
                            ]
                            new_values[state] = max(q_values)
            self.values = new_values

    def get_value(self, state):
        return self.values.get(state, 0)

    def compute_qvalue_from_values(self, state, action):
        i, j = state
        action_probabilities = self.mdp.P[i][j]
        if action_probabilities == self.mdp.PROHIBITED_CELL:
            return 0

        if action == "exit":
            reward = action_probabilities[
                0
            ]  # Suponemos que la recompensa está en la primera posición de la lista
            return reward

        action_index = ["up", "down", "left", "right"].index(action)
        action_probability = action_probabilities[action_index]

        if action_probability == 0:
            return 0

        new_state = state
        if action == "up":
            new_state = (max(i - 1, 0), j)
        elif action == "down":
            new_state = (min(i + 1, self.mdp.nrows - 1), j)
        elif action == "left":
            new_state = (i, max(j - 1, 0))
        elif action == "right":
            new_state = (i, min(j + 1, self.mdp.ncols - 1))

        if self.mdp.board[new_state[0]][new_state[1]] == self.mdp.PROHIBITED_CELL:
            new_state = state

        reward = 0
        return (reward + self.discount * self.get_value(new_state)) * action_probability

    def compute_action_from_values(self, state):
        actions = self.mdp.get_posible_actions(state)
        if not actions:
            return None

        best_action = None
        best_q_value = float("-inf")

        for action in actions:
            q_value = self.compute_qvalue_from_values(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action

    def get_action(self, state):
        return self.compute_action_from_values(state)

    def get_qvalue(self, state, action):
        return self.compute_qvalue_from_values(state, action)

    def get_policy(self, state):
        return self.compute_action_from_values(state)
