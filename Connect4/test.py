import numpy as np
import torch

from network import ConnectNet
from rules import init_board


model = ConnectNet(input_shape=(1, 6, 7),
                   num_res_blocks=2,
                   num_actions=7,
                   num_filters=32,
                   policy_conv_channels_out=2,
                   value_conv_channels_out=1,
                   value_lin1_out=8)

board = init_board()
encode_dict = {'':0, 'X':1, 'O':-1}
encoded_board = np.array([[encode_dict[board[i, j]] for j in range(board.shape[1])] for i in range(board.shape[0])])
board = torch.Tensor(encoded_board).unsqueeze(dim=0).unsqueeze(dim=0)

print(board.shape)
p, v = model.forward(board)
print(p)
print(v)