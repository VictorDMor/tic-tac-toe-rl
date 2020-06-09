import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.is_end = False
        self.board_hash = None
        # Player 1 has first play
        self.player_symbol = 1
        self.p1_wins = 0
        self.p2_wins = 0
        self.number_of_ties = 0

    def print_stats(self):
        print("=================================")
        print("{} wins: {}".format(self.p1.name, self.p1_wins))
        print("{} wins: {}".format(self.p2.name, self.p2_wins))
        print("Ties: {}".format(self.number_of_ties))
        print("=================================")

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

    def check_winner(self):
        # For rows
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.is_end = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.is_end = True
                return -1
        # For columns
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.is_end = True
                return 1
            if sum(self.board[:, i]) == -3:
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

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board_hash = None
        self.is_end = False
        self.player_symbol = 1

    # Only when game ends
    def give_reward(self):
        result = self.check_winner()
        # Backpropagate the reward!
        if result == 1:
            self.p1.feed_reward(1)
            self.p2.feed_reward(0)
        elif result == -1:
            self.p1.feed_reward(0)
            self.p2.feed_reward(1)
        else:
            self.p1.feed_reward(0.1) # Why? (one can try out different reward to see how the agents act)
            self.p2.feed_reward(0.5)

    def play(self, rounds=100):
        for i in range(1, rounds+1):
            # self.print_stats()
            # if i % 1 == 0:
            #     print("=================================")
            #     print("             ROUND {}".format(i))
            #     print("=================================")
            while not self.is_end:
                # Player 1
                positions = self.available_positions()
                p1_action = self.p1.choose_action(positions, self.board, self.player_symbol)

                # Take action and update board state
                self.update_state(p1_action)
                board_hash = self.get_hash()
                self.p1.add_state(board_hash)
                # Check board status if the game is over
                win = self.check_winner()
                if win is not None:
                    # The game ended with p1 winning or draw/tie
                    if win == 1:
                        # print("{} wins!".format(self.p1.name))
                        self.p1_wins += 1
                    else:
                        # print("It's a tie!")
                        self.number_of_ties += 1
                    # self.show_board()
                    self.give_reward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    # Player 2
                    positions = self.available_positions()
                    p2_action = self.p2.choose_action(positions, self.board, self.player_symbol)
                    self.update_state(p2_action)
                    board_hash = self.get_hash()
                    self.p2.add_state(board_hash)
                    win = self.check_winner()
                    if win is not None:
                        # The game ended with player 2 winning or draw/tie
                        if win == -1:
                            # print("{} wins!".format(self.p2.name))
                            self.p2_wins += 1
                        else:
                            # print("It's a tie!")
                            self.number_of_ties += 1
                        # self.show_board()
                        self.give_reward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
        self.p1.save_policy()
        self.p2.save_policy()

    def play_with_human(self):
        while not self.is_end:
            # Player 1
            positions = self.available_positions()
            p1_action = self.p1.choose_action(positions, self.board, self.player_symbol)
             
            # Take action and update board state
            self.update_state(p1_action)
            self.show_board()

            win = self.check_winner()
            if win is not None:
                if win == 1:
                    print("{} wins!".format(self.p1.name))
                    self.p1_wins += 1
                else:
                    print("It's a tie!")
                    self.number_of_ties += 1
                self.reset()
                break
            else:
                # Player 2
                positions = self.available_positions()
                p2_action = self.p2.choose_action(positions)

                self.update_state(p2_action)
                self.show_board()
                win = self.check_winner()
                if win is not None:
                    if win == -1:
                        print("{} wins!".format(self.p2.name))
                        self.p2_wins += 1
                    else:
                        print("It's a tie!")
                        self.number_of_ties += 1
                    self.reset()
                    break 

    def show_board(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {} # Maps as state -> value

    # Appends a hash state
    def add_state(self, state):
        self.states.append(state)

    def reset(self):
        self.states = []

    def get_hash(self, board):
        ''' This function generates a unique hash that identifies the state for the board '''
        board_hash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return board_hash

    def choose_action(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # Takes random action because it's lower than the exploration_rate
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_board_hash = self.get_hash(next_board)
                value = 0 if self.states_value.get(next_board_hash) is None else self.states_value.get(next_board_hash)
                #print("Value is: ", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action

    def feed_reward(self, reward):
        for state in reversed(self.states):
            if self.states_value.get(state) is None:
                self.states_value[state] = 0
            self.states_value[state] += self.lr * (self.decay_gamma * reward - self.states_value[state])
            reward = self.states_value[state]

    def save_policy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()
    
    def load_policy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self, name):
        self.name = name
    
    def choose_action(self, positions):
        row = int(input("Input your action row: "))
        col = int(input("Input your action column: "))
        action = (row, col)
        if action in positions:
            return action

    def add_state(self, state):
        pass
    
    def feed_reward(self, reward):
        pass

    def reset(self):
        pass

if __name__ == "__main__":
    player1 = Player("CPU1")
    player2 = Player("CPU2")
    s1 = State(player1, player2)

    print("Training the agents... \n")
    s1.play(50000)
    # s1.print_stats()

    # Play with human
    player1 = Player("CPU", exp_rate=0)
    player1.load_policy("policy_CPU1")

    player2 = HumanPlayer("Victor")
    s2 = State(player1, player2)
    for i in range(10):
        s2.play_with_human()
        s2.print_stats()