import numpy as np
import matplotlib.pyplot as plt


class SARSA:
    def __init__(self, env, epsilon=0.9, gamma=0.96, alpha=0.81):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = {}
        self.env = env

    def _ensure_state(self, state):
        actions = self.env.get_posible_actions(state)
        if state not in self.Q:
            self.Q[state] = {action: 0.0 for action in actions}
        else:
            for action in actions:
                self.Q[state].setdefault(action, 0.0)

    def choose_action(self, state):
        self._ensure_state(state)
        actions = list(self.Q[state].keys())

        if len(actions) == 1:
            return actions[0]

        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)

        max_q = max(self.Q[state].values())
        best_actions = [a for a, v in self.Q[state].items() if v == max_q]
        return np.random.choice(best_actions)

    def action_function(self, state1, action1, reward, state2, action2):
        self._ensure_state(state1)
        self._ensure_state(state2)
        current = self.Q[state1][action1]
        target = reward + self.gamma * self.Q[state2][action2]
        self.Q[state1][action1] = (1 - self.alpha) * current + self.alpha * target

    def train(
        self,
        episodes=2000,
        max_steps=300,
        epsilon_decay=0.997,
        min_epsilon=0.05,
        step_reward=-1.0,
    ):
        episode_returns = []
        episode_steps = []

        for _ in range(episodes):
            self.env.reset()
            state = self.env.get_current_state()
            action = self.choose_action(state)
            total_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                reward, next_state = self.env.do_action(action)
                steps += 1

                if action != "exit":
                    reward += step_reward

                total_reward += reward
                next_action = self.choose_action(next_state)
                self.action_function(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

                if action == "exit":
                    final_reward, terminal_state = self.env.do_action("exit")
                    total_reward += final_reward
                    self._ensure_state(state)
                    self._ensure_state(terminal_state)
                    current = self.Q[state]["exit"]
                    target = final_reward
                    self.Q[state]["exit"] = (
                        1 - self.alpha
                    ) * current + self.alpha * target
                    steps += 1
                    break

            episode_returns.append(total_reward)
            episode_steps.append(steps)
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)

        return np.array(episode_returns), np.array(episode_steps)

    def greedy_policy(self):
        policy = {}
        for i in range(self.env.nrows):
            for j in range(self.env.ncols):
                if self.env.board[i][j] == self.env.PROHIBITED_CELL:
                    continue
                state = (i, j)
                self._ensure_state(state)
                best_action = max(self.Q[state], key=self.Q[state].get)
                policy[state] = best_action
        return policy

    def value_matrix(self):
        values = np.full((self.env.nrows, self.env.ncols), np.nan)
        for i in range(self.env.nrows):
            for j in range(self.env.ncols):
                if self.env.board[i][j] == self.env.PROHIBITED_CELL:
                    continue
                state = (i, j)
                self._ensure_state(state)
                values[i, j] = max(self.Q[state].values())
        return values

    @staticmethod
    def moving_average(values, window=50):
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")
