import numpy as np

from environment import Environment

ACTIONS = ["up", "down", "left", "right"]

ACTION_NOISE = {"up": 0.3, "down": 0.2, "left": 0.2, "right": 0.3}


class TDLearning:
    def __init__(
        self,
        env: Environment,
        policy: dict,
        alpha=0.7,
        gamma=0.9,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.policy = dict(policy)
        self.values = {
            (i, j): 0.0
            for i in range(env.nrows)
            for j in range(env.ncols)
            if env.board[i][j] != env.PROHIBITED_CELL
        }

    def _sample_actual_action(self, intended_action):
        noise = ACTION_NOISE[intended_action]
        if np.random.random() < noise:
            return np.random.choice([a for a in ACTIONS if a != intended_action])
        return intended_action

    def _predicted_next_state(self, state, action):
        i, j = state
        candidates = {
            "up": (max(i - 1, 0), j),
            "down": (min(i + 1, self.env.nrows - 1), j),
            "left": (i, max(j - 1, 0)),
            "right": (i, min(j + 1, self.env.ncols - 1)),
        }
        next_s = candidates[action]
        return (
            state
            if self.env.board[next_s[0]][next_s[1]] == self.env.PROHIBITED_CELL
            else next_s
        )

    def _run_episode(self, max_steps):
        self.env.reset()

        i = 0
        while i < max_steps:
            state = self.env.get_current_state()
            intended = self.policy.get(state)
            if intended is None:
                break

            if intended == "exit":
                reward, _ = self.env.do_action("exit")
                self.values[state] = (1 - self.alpha) * self.values[
                    state
                ] + self.alpha * reward
                break

            actual = self._sample_actual_action(intended)
            reward, next_state = self.env.do_action(actual)
            td_target = reward + self.gamma * self.values.get(next_state, 0.0)
            self.values[state] = (1 - self.alpha) * self.values[
                state
            ] + self.alpha * td_target
            i += 1

    def _greedy_policy(self):
        policy = {}
        for i in range(self.env.nrows):
            for j in range(self.env.ncols):
                state = (i, j)
                if self.env.board[i][j] == self.env.PROHIBITED_CELL:
                    continue
                actions = self.env.get_posible_actions(state)
                if len(actions) == 1:
                    policy[state] = actions[0]
                    continue
                best = max(
                    actions,
                    key=lambda a: self.values.get(
                        self._predicted_next_state(state, a), 0.0
                    ),
                )
                policy[state] = best
        return policy

    def run(self, episodes_per_step=500, max_steps=float("inf"), max_iterations=1000):
        stable_count = 0
        iterations = 0
        while stable_count < 3 and iterations < max_iterations:
            i = 0

            while i < episodes_per_step:
                self._run_episode(max_steps=max_steps)
                i += 1

            new_policy = self._greedy_policy()

            iterations += 1
            if new_policy == self.policy:
                stable_count += 1
            else:
                stable_count = 0
                self.policy = new_policy

            if iterations % 10 == 0:
                print(f"Iteración {iterations}, política: {self.policy}")

        self.iterations = iterations
        return iterations
