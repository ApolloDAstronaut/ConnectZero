### Encode / Decode representations of board
import numpy as np

'''
Convert between
    - string representation ('X', '', 'O')
    - and number representation (-1, 0, 1)

Strings work well for visualisation,
numbers work better for the neural network. 
'''

# Convert string representation to number
def encode(board: np.ndarray) -> np.ndarray:
    encoded_board = np.zeros_like(board)
    encoded_board[board == 'X'] = 1
    encoded_board[board == 'O'] = -1
    return encoded_board

# Convert number representation to string
def decode(board: np.ndarray) -> np.ndarray:
    decoded_board = np.full_like(board, '', dtype=np.int32)
    decoded_board[board == 1] = 'X'
    decoded_board[board == -1] = 'O'
    return decoded_board