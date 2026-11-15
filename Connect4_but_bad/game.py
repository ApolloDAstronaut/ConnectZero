import numpy as np

'''Connect4_but_bad rules and functions, 
    Squares: 0: empty, 1: player 1, -1:player 2'''

# 1. init_board
# 2. place_piece
# 3. get_valid_moves
# 4. is_board_full
# 5. is_win

def init_board():
    return np.zeros((6, 7))

def place_piece(board, player, action):
    board_copy = board.copy()
    row_index = sum(board_copy[:, action] == 0) - 1
    board_copy[row_index, action] = player
    return board_copy

def get_valid_moves(board):
    # return [0, 1, 1, 0, 1, 1, 1]
    return np.array([int(sum(board[:, i] == 0) > 0) for i in range(7)])

def is_board_full(board):
    return np.sum(board == 0) == 0

def is_win(board, player):
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            horizontal = board[row, col:col+4]
            vertical = board[row:row+4, col]
            diagonal = np.diagonal(board[row:4, col:4])

            if horizontal.size == 4 and np.all(horizontal == player):
                return True
            if vertical.size == 4 and np.all(vertical == player):
                return True
            if diagonal.size == 4 and np.all(diagonal == player):
                return True
    return False


