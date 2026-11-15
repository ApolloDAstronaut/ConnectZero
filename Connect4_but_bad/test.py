from connect_net import ConnectNet
from mcts import mcts
import numpy as np

board = np.array(
        [[ 0,-1,-1,-1, 1, 0,-1],
         [ 0, 1,-1, 1, 1, 0, 1],
         [-1, 1,-1, 1, 1, 0,-1],
         [ 1,-1, 1,-1,-1, 0,-1],
         [-1,-1, 1,-1, 1, 1,-1],
         [-1, 1, 1,-1, 1,-1, 1]]
         )
board_1 = board == 1
board_2 = board == -1


tree = mcts(board, 100)
print(tree.value / tree.visit_count)

model = ConnectNet(input_shape=(2, 6, 7), num_filters=5, num_res_block=2)
model.forward((board_1, board_2))


