import numpy as np

from environment import Environment


class MCM:
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(
        self,
        env: Environment,
        discount: float = 0.9,
        epsilon: float = 0.1,
        noise: float = 0.25,
        convergence_threshold: float = 0.01,
        patience: int = 3,
        window: int = 100,
        max_episodes: int = 300_000,
    ):
        self.env = env
        self.discount = discount
        self.epsilon = epsilon
        self.noise = noise
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.window = window
        self.max_episodes = max_episodes

        self.values: dict = {}
        self.policy: dict = {}
        self.returns: dict = {}
        self.episodes_run: int = 0
        self.value_history: list = []

        self._init_state()

    def _init_state(self):
        for i in range(self.env.nrows):
            for j in range(self.env.ncols):
                state = (i, j)
                if self.env.board[i][j] != self.env.PROHIBITED_CELL:
                    self.values[state] = 0.0
                    self.returns[state] = []
                    actions = self.env.get_posible_actions(state)
                    self.policy[state] = actions[0] if actions else None

    def _stochastic_step(self, state, intended_action):
        if np.random.random() < self.noise:
            actual_action = np.random.choice(self.ACTIONS)
        else:
            actual_action = intended_action

        self.env.current_state = state
        reward, new_state = self.env.do_action(actual_action)
        return reward, new_state

    def _epsilon_greedy(self, state):
        actions = self.env.get_posible_actions(state)
        if not actions:
            return None
        if len(actions) == 1:
            return actions[0]
        if np.random.random() < self.epsilon:
            return np.random.choice(actions)
        return self.policy.get(state, np.random.choice(actions))

    def generate_episode(self, max_steps: int = 300):
        self.env.reset()
        trajectory = []

        for _ in range(max_steps):
            state = self.env.get_current_state()

            if self.env.is_terminal():
                action = "exit"
                self.env.current_state = state
                reward, _ = self.env.do_action(action)
                trajectory.append((state, action, reward))
                break

            action = self._epsilon_greedy(state)
            if action is None:
                break

            reward, _ = self._stochastic_step(state, action)
            trajectory.append((state, action, reward))

        return trajectory

    def _update_values(self, episode):
        G = 0.0
        visited = set()
        for state, _, reward in reversed(episode):
            G = reward + self.discount * G
            if state not in visited:
                visited.add(state)
                self.returns[state].append(G)
                self.values[state] = float(np.mean(self.returns[state]))

    def _update_policy(self):
        for i in range(self.env.nrows):
            for j in range(self.env.ncols):
                state = (i, j)
                cell = self.env.board[i][j]
                if cell == self.env.PROHIBITED_CELL:
                    continue
                actions = self.env.get_posible_actions(state)
                if not actions:
                    continue
                if len(actions) == 1:
                    self.policy[state] = actions[0]
                    continue
                best = max(actions, key=lambda a: self._estimate_q(state, a))
                self.policy[state] = best

    def _estimate_q(self, state, action):
        i, j = state
        if action == "exit":
            return self.values.get(state, 0.0)

        p_intended = 1.0 - self.noise
        p_random = self.noise / len(self.ACTIONS)

        total = 0.0
        for act in self.ACTIONS:
            p = p_intended if act == action else 0.0
            p += p_random

            ni, nj = i, j
            if act == "up":
                ni = max(i - 1, 0)
            elif act == "down":
                ni = min(i + 1, self.env.nrows - 1)
            elif act == "left":
                nj = max(j - 1, 0)
            elif act == "right":
                nj = min(j + 1, self.env.ncols - 1)

            ns = (ni, nj)
            if self.env.board[ns[0]][ns[1]] == self.env.PROHIBITED_CELL:
                ns = state

            total += p * self.discount * self.values.get(ns, 0.0)

        return total

    def run_monte_carlo(self):
        stable_count = 0
        prev_policy = {s: None for s in self.values}

        for ep in range(self.max_episodes):
            episode = self.generate_episode()
            self._update_values(episode)
            self._update_policy()
            self.episodes_run = ep + 1

            if (ep + 1) % self.window == 0:
                snapshot = dict(self.values)
                self.value_history.append(snapshot)

                policy_changed = any(
                    self.policy.get(s) != prev_policy.get(s) for s in self.policy
                )
                prev_policy = dict(self.policy)

                max_delta = max(
                    (
                        (np.std(self.returns[s]) / (len(self.returns[s]) ** 0.5))
                        if len(self.returns.get(s, [])) > 1
                        else 1.0
                    )
                    for s in self.returns
                )

                if not policy_changed and max_delta < self.convergence_threshold:
                    stable_count += 1
                    if stable_count >= self.patience:
                        break
                else:
                    stable_count = max(0, stable_count - 1)

    def get_value(self, state):
        return self.values.get(state, 0.0)

    def get_policy(self, state):
        return self.policy.get(state, None)

    def get_qvalue(self, state, action):
        return self._estimate_q(state, action)
