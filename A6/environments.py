import numpy as np
import random


class CliffWalkEnv:
    CLIFF_REWARD = -100
    STEP_REWARD = 0
    GOAL_REWARD = 0

    def __init__(self, rows=5, cols=10, randomize_start=False):
        self.rows = rows
        self.cols = cols
        self.randomize_start = randomize_start
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        self.cliff = [(rows - 1, c) for c in range(1, cols - 1)]
        self.current_state = self.start

    def reset(self):
        if self.randomize_start:
            safe = [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if (r, c) != self.goal and (r, c) not in self.cliff
            ]
            self.current_state = random.choice(safe)
        else:
            self.current_state = self.start

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state=None):
        return ["up", "down", "left", "right"]

    def do_action(self, action):
        r, c = self.current_state
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        dr, dc = moves[action]
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        next_state = (nr, nc)

        if next_state in self.cliff:
            self.current_state = self.start
            return self.CLIFF_REWARD, self.current_state

        self.current_state = next_state
        reward = self.GOAL_REWARD if next_state == self.goal else self.STEP_REWARD
        return reward, self.current_state

    def is_terminal(self):
        return self.current_state == self.goal


class LockedDoorEnv:
    EMPTY = "."
    WALL = "#"
    AGENT = "A"
    KEY = "K"
    BALL = "B"
    DOOR = "D"
    GOAL = "G"

    def __init__(self, key_color="blue", randomize_start=False, random_keys=False):
        self.key_color = key_color
        self.randomize_start = randomize_start
        self.random_keys = random_keys
        self._build_layout()
        self._reset_objects()

    def _build_layout(self):
        self.rows = 7
        self.cols = 13
        self.wall_col = 6
        self.door_row = 3
        self.goal_pos = (3, 11)
        self.fixed_agent_start = (3, 2)
        self.fixed_key_pos = (5, 2)
        self.fixed_ball_pos = (3, 6)

        self.static_walls = set()
        for r in range(self.rows):
            self.static_walls.add((r, 0))
            self.static_walls.add((r, self.cols - 1))
            self.static_walls.add((0, r if r < self.cols else r))
        for c in range(self.cols):
            self.static_walls.add((0, c))
            self.static_walls.add((self.rows - 1, c))
        for r in range(self.rows):
            if r != self.door_row:
                self.static_walls.add((r, self.wall_col))

    def _reset_objects(self):
        if self.randomize_start:
            left_free = [
                (r, c)
                for r in range(1, self.rows - 1)
                for c in range(1, self.wall_col)
                if (r, c) not in self.static_walls
            ]
            self.agent_pos = random.choice(left_free)
        else:
            self.agent_pos = self.fixed_agent_start

        if self.random_keys:
            left_free = [
                (r, c)
                for r in range(1, self.rows - 1)
                for c in range(1, self.wall_col)
                if (r, c) not in self.static_walls and (r, c) != self.agent_pos
            ]
            key_colors = ["blue", "red", "green"]
            self.keys = {random.choice(left_free): random.choice(key_colors)}
        else:
            self.keys = {self.fixed_key_pos: self.key_color}

        self.ball_pos = self.fixed_ball_pos
        self.has_key = None
        self.ball_removed = False
        self.door_open = False

    def reset(self):
        self._reset_objects()

    def get_current_state(self):
        key_tuple = tuple(sorted(self.keys.items())) if self.keys else ()
        return (
            self.agent_pos,
            key_tuple,
            self.ball_pos,
            self.has_key,
            self.ball_removed,
            self.door_open,
        )

    def _left_room_free(self):
        return [
            (r, c)
            for r in range(1, self.rows - 1)
            for c in range(1, self.wall_col)
            if (r, c) not in self.static_walls
            and (r, c) != self.agent_pos
            and (r, c) not in self.keys
        ]

    def get_possible_actions(self, state=None):
        actions = ["up", "down", "left", "right", "pick_up", "open_door"]
        return actions

    def _try_move(self, dr, dc):
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        if (nr, nc) in self.static_walls:
            return
        if (nr, nc) == (self.door_row, self.wall_col) and not self.door_open:
            return
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            self.agent_pos = (nr, nc)

    def do_action(self, action):
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        if action in moves:
            self._try_move(*moves[action])
            return 0, self.get_current_state()

        if action == "pick_up":
            if self.agent_pos == self.ball_pos and not self.ball_removed:
                self.ball_removed = True
                self.ball_pos = None
                return 5, self.get_current_state()
            if self.agent_pos in self.keys:
                color = self.keys.pop(self.agent_pos)
                self.has_key = color
                return 5, self.get_current_state()
            return -1, self.get_current_state()

        if action == "open_door":
            door_adj = (self.door_row, self.wall_col - 1)
            if (
                self.agent_pos == door_adj
                and self.ball_removed
                and self.has_key == self.key_color
                and not self.door_open
            ):
                self.door_open = True
                return 10, self.get_current_state()
            return -1, self.get_current_state()

        return 0, self.get_current_state()

    def is_terminal(self):
        return self.agent_pos == self.goal_pos

    def get_goal_reward(self):
        return 100

    def render(self):
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for r, c in self.static_walls:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                grid[r][c] = "#"
        for r in range(self.rows):
            if r != self.door_row:
                grid[r][self.wall_col] = "|"
        door_char = "D" if not self.door_open else " "
        grid[self.door_row][self.wall_col] = door_char
        for pos, color in self.keys.items():
            grid[pos[0]][pos[1]] = color[0].upper()
        if self.ball_pos:
            grid[self.ball_pos[0]][self.ball_pos[1]] = "B"
        grid[self.goal_pos[0]][self.goal_pos[1]] = "G"
        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()
