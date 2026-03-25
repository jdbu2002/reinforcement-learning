import numpy as np


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def _ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.env.get_possible_actions(state)}
        else:
            for a in self.env.get_possible_actions(state):
                self.Q[state].setdefault(a, 0.0)

    def choose_action(self, state):
        self._ensure_state(state)
        actions = list(self.Q[state].keys())
        if len(actions) == 1:
            return actions[0]
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        max_q = max(self.Q[state].values())
        best = [a for a, v in self.Q[state].items() if v == max_q]
        return np.random.choice(best)

    def _update(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)
        current = self.Q[state][action]
        next_max = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0
        self.Q[state][action] = (1 - self.alpha) * current + self.alpha * (
            reward + self.gamma * next_max
        )

    def get_reward(self, state, action, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)
        return self.Q[state].get(action, 0.0) - max(self.Q[next_state].values())

    def step(self, step_reward, terminal_reward):
        state = self.env.get_current_state()
        action = self.choose_action(state)
        reward, next_state = self.env.do_action(action)
        done = self.env.is_terminal()
        if done and terminal_reward is not None:
            reward = terminal_reward
        else:
            reward += step_reward

        self._update(state, action, reward, next_state)
        state = next_state
        return next_state, reward

    def train(
        self,
        episodes=1000,
        max_steps=500,
        epsilon_decay=0.997,
        min_epsilon=0.05,
        step_reward=-1.0,
        terminal_reward=None,
    ):
        episode_returns = []
        episode_steps = []

        for _ in range(episodes):
            self.env.reset()
            total_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                _, reward = self.step(step_reward, terminal_reward)
                total_reward += reward
                steps += 1

                if self.env.is_terminal():
                    break

            episode_returns.append(total_reward)
            episode_steps.append(steps)
            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)

        return np.array(episode_returns), np.array(episode_steps)

    def greedy_policy(self):
        policy = {}
        for state, actions in self.Q.items():
            if actions:
                policy[state] = max(actions, key=actions.get)
        return policy

    def save_qtable(self, path):
        import json

        serializable = {str(k): v for k, v in self.Q.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load_qtable(self, path):
        import json

        with open(path) as f:
            raw = json.load(f)
        import ast

        self.Q = {ast.literal_eval(k): v for k, v in raw.items()}
