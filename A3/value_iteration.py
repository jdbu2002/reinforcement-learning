class Environment:
    EMPTY_CELL = " "
    PROHIBITED_CELL = "#"
    INITIAL_CELL = "S"

    def __init__(self, board, P):
        self.board = board
        self.nrows = len(board)
        self.ncols = len(board[0])
        self.initial_state = self.get_initial_state()
        self.current_state = self.initial_state
        self.P = P

    def get_initial_state(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.board[i][j] == self.INITIAL_CELL:
                    return (i, j)
        return None

    def get_current_state(self):
        return self.current_state

    def get_posible_actions(self, state):
        i, j = state
        if self.board[i][j] == self.INITIAL_CELL or self.board[i][j] == self.EMPTY_CELL:
            return ["up", "down", "left", "right"]

        if self.board[i][j] != self.PROHIBITED_CELL:
            return ["exit"]

        return []

    def do_action(self, action):
        i, j = self.current_state
        if action not in self.get_posible_actions(self.current_state):
            return 0, self.current_state

        # Obtener la probabilidad de la acción
        action_probabilities = self.P[i][j]

        if action_probabilities == self.PROHIBITED_CELL:
            print("You should not be here!")
            return 0, self.current_state

        if action == "exit":
            reward = action_probabilities[
                0
            ]  # Suponemos que la recompensa está en la primera posición de la lista
            self.current_state = self.initial_state
            return reward, self.current_state

        action_index = ["up", "down", "left", "right"].index(action)
        action_probability = action_probabilities[action_index]

        if action_probability == 0:
            return 0, self.current_state

        # Determinar el nuevo estado basado en la acción y su probabilidad
        new_state = self.current_state
        if action == "up":
            new_state = (max(i - 1, 0), j)
        elif action == "down":
            new_state = (min(i + 1, self.nrows - 1), j)
        elif action == "left":
            new_state = (i, max(j - 1, 0))
        elif action == "right":
            new_state = (i, min(j + 1, self.ncols - 1))

        # Verificar si el nuevo estado es un obstáculo o no
        if self.board[new_state[0]][new_state[1]] == self.PROHIBITED_CELL:
            new_state = self.current_state

        # Actualizar el estado actual del agente
        self.current_state = new_state

        return 0, new_state

    def reset(self):
        self.current_state = self.initial_state

    def is_terminal(self):
        i, j = self.current_state
        return (
            self.board[i][j] != self.EMPTY_CELL
            and self.board[i][j] != self.INITIAL_CELL
        )


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


# Implementación de PolicyIteration
class PolicyIteration(ValueIteration):
    def __init__(
        self, mdp: Environment, discount=0.9, eval_iterations=20, policy_iterations=100
    ):
        super().__init__(mdp, discount, eval_iterations)
        self.policy_iterations = policy_iterations
        self.values = {}
        self.policy = {}
        # Inicializar política aleatoria válida
        for i in range(self.mdp.nrows):
            for j in range(self.mdp.ncols):
                state = (i, j)
                actions = self.mdp.get_posible_actions(state)
                if actions:
                    self.policy[state] = actions[0]
                else:
                    self.policy[state] = None

    def policy_evaluation(self):
        for _ in range(self.iterations):
            new_values = {}
            for i in range(self.mdp.nrows):
                for j in range(self.mdp.ncols):
                    state = (i, j)
                    if self.mdp.board[i][j] == self.mdp.PROHIBITED_CELL:
                        new_values[state] = 0
                    else:
                        action = self.policy.get(state)
                        if action is None:
                            new_values[state] = 0
                        else:
                            new_values[state] = self.compute_qvalue_from_values(
                                state, action
                            )
            self.values = new_values

    def policy_iteration(self):
        for _ in range(self.policy_iterations):
            # 1. Evaluación de la política
            self.policy_evaluation()

            policy_stable = True

            for i in range(self.mdp.nrows):
                for j in range(self.mdp.ncols):
                    state = (i, j)
                    if self.mdp.board[i][j] == self.mdp.PROHIBITED_CELL:
                        continue
                    actions = self.mdp.get_posible_actions(state)
                    if not actions:
                        continue
                    # Encontrar la mejor acción según los valores actuales
                    best_action = None
                    best_q = float("-inf")
                    for action in actions:
                        q = self.compute_qvalue_from_values(state, action)
                        if q > best_q:
                            best_q = q
                            best_action = action
                    if best_action != self.policy.get(state):
                        policy_stable = False
                        self.policy[state] = best_action

            if policy_stable:
                break

    def get_policy(self, state):
        return self.policy.get(state, None)
