### Encode the rules of Connect4
import numpy as np

'''
Boards are 6x7: 6 rows, 7 columns.

Initially, the board is an array of empty strings.

'X' moves first, --> Player 1 --> 1
'O' moves second, --> Player 2 --> -1

Then they alternate. 
The XO scheme is to make board positions a bit clearer when printed. 

As a rule for function parameters,
board goes first,
then player,
then action.

Narrowing down the scale progressively.
'''

# 1. initialise the board
# 2. what moves are allowed?
# 3. do a move
# 4. is it a win? (for this player)
# 5. is the board full?


def init_board():
    return np.full((6, 7), 0)


def legal_actions(board):
    # [False, False, False, True, False, False, False] <==> can only place a piece in the middle
    allowed_moves = np.full(7, False)
    for i in range(7):
        # check if column has any empty cells left
        allowed_moves[i] = sum(board[:, i] == 0) > 0
    return allowed_moves


# with the given [board], [player] performs [action]
def place_piece(board, player, action):
    # return board state after performing the action
    assert board[0, action] == 0, "Tried to place piece on full column."
    board_copy = board.copy()

    row_index = sum(board[:, action] == 0) - 1 # what is the last empty cell looking from the top?
    board_copy[row_index, action] = player
    return board_copy


# does [player] have 4-in-a-row somewhere on [board]?
def is_win(board, player):
    # Loop through all board cells,
    # check for 4-in-a-row of [player] to the right, bottom, and bottom-right diagonal
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            right = board[row, col:col+4]
            bottom = board[row:row+4, col]
            right_diagonal = np.diag(board[row:row+4, col:col+4])
            left_diagonal = np.diag(board[row:row+4, col:col+4][:, ::-1])

            if right.size == 4 and np.all(right == player):
                return True
            if bottom.size == 4 and np.all(bottom == player):
                return True
            if right_diagonal.size == 4 and np.all(right_diagonal == player):
                return True
            if left_diagonal.size == 4 and np.all(left_diagonal == player):
                return True
    return False


def is_board_full(board):
    # if there are no empty cells left --> board is full
    return np.sum(board == 0) == 0


def is_terminal(board):
    return (is_win(board, 1) or
                is_win(board, -1) or
                is_board_full(board))



