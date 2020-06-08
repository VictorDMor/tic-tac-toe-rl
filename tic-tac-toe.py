import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3

def __init__(self, p1, p2):
    self.board = np.zeros({BOARD_ROWS, BOARD_COLS})
    self.p1 = p1
    self.p2 = p2
    self.is_end = False
    self.board_hash = None
    # Player 1 has first play
    self.player_symbol = 1

def get_hash(self):
    ''' This function generates a unique hash that identifies the state for the board '''
    self.board_hash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
    return self.board_hash

def available_positions(self):
    positions = []
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if self.board[i, j] == 0:
                positions.append((i, j)) # According to the author, this parameter needs to be a tuple
    return positions

def update_state(self, position):
    self.board[position] = self.player_symbol
    # Switch to another player
    self.player_symbol = -1 if self.player_symbol == 1 else 1

def check_winner(self)
    # For rows
    for i in range(BOARD_ROWS):
        if sum(self.board[i, :]) == 3
            self.is_end = True
            return 1
        if sum(self.board[i, :]) == -3
            self.is_end = True
            return -1
    # For columns
    for i in range(BOARD_COLS:
        if sum(self.board[:, i]) == 3:
            self.is_end = True
            return 1
        if sum(self.board[:, i]) == -3
            self.is_end = True
            return -1
    # For diagonals
    diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
    diag_sum2 = sum([self.board[i, BOARD_COLS-i-1] for i in range(BOARD_COLS)])
    diag_sum = max(abs(diag_sum1), abs(diag_sum2))
    if diag_sum == 3:
        self.is_end = True
        if diag_sum1 == 3 or diag_sum2 == 3:
            return 1
        else:
            return -1
    # It's a tie and there's no available positions
    if len(self.available_positions()) == 0:
        self.is_end = True
        return 0
    # Else, the game is not over yet
    self.is_end = False
    return None

# Only when game ends
def give_reward(self):
    result = self.winner()
    # Backpropagate the reward!
    if result == 1:
        self.p1.feed_reward(1)
        self.p2.feed_reward(0)
    elif result == -1
        self.p1.feed_reward(0)
        self.p2.feed_reward(1)
    else:
        self.p1.feed_reward(0.1) # Why? (one can try out different reward to see how the agents act)
        self.p2.feed_reward(0.5)