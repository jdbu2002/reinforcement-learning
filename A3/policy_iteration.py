from .value_iteration import ValueIteration
from .environment import Environment

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
