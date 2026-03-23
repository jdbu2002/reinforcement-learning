class Environment:
    EMPTY_CELL = " "
    PROHIBITED_CELL = "#"
    INITIAL_CELL = "S"

    def __init__(self, board, P=None):
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

        if self.P is not None:
            action_probabilities = self.P[i][j]

            if action_probabilities == self.PROHIBITED_CELL:
                print("You should not be here!")
                return 0, self.current_state

            if action == "exit":
                reward = action_probabilities[0]
                self.current_state = self.initial_state
                return reward, self.current_state

            action_index = ["up", "down", "left", "right"].index(action)
            action_probability = action_probabilities[action_index]

            if action_probability == 0:
                return 0, self.current_state
        else:
            if action == "exit":
                reward = float(self.board[i][j])
                self.current_state = self.initial_state
                return reward, self.current_state

        new_state = self.current_state
        if action == "up":
            new_state = (max(i - 1, 0), j)
        elif action == "down":
            new_state = (min(i + 1, self.nrows - 1), j)
        elif action == "left":
            new_state = (i, max(j - 1, 0))
        elif action == "right":
            new_state = (i, min(j + 1, self.ncols - 1))

        if self.board[new_state[0]][new_state[1]] == self.PROHIBITED_CELL:
            new_state = self.current_state

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
