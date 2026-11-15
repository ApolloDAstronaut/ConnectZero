### Code to play against a stupid
### TODO: fix this terrible code
### WARNING: bad spaghetti code

from rules import place_piece, init_board, is_win, is_board_full, legal_actions
import numpy as np
import torch

from network import ConnectNet

encode_dict = {'':0, 'X':1, 'O':-1}
model = ConnectNet(input_shape=(1, 6, 7),
                   num_res_blocks=2,
                   num_actions=7,
                   num_filters=32,
                   policy_conv_channels_out=2,
                   value_conv_channels_out=1,
                   value_lin1_out=8)

board = init_board()
print(board)
current_player = 1
run = True
while run:
    if current_player == -1: # player is bot
        encoded_board = torch.Tensor(np.array(
            [[encode_dict[board[i, j]] for j in range(board.shape[1])] for i in range(board.shape[0])])).unsqueeze(0).unsqueeze(0)
        policy, value = model.forward(encoded_board)
        policy = policy.tolist()[0]
        legal_mask = np.array(legal_actions(board), dtype=bool)
        legal_moves = np.array(list((enumerate(policy))))[legal_mask]
        selected_move = np.argmax(legal_moves[:, 1])
        print(f"Player O selected column {selected_move + 1}!")
        board = place_piece(board, 'O', selected_move)
    else:
        selected_move = int(input("Player X: ")) - 1
        board = place_piece(board, 'X', selected_move)
    print(board)

    if is_win(board, current_player):
        run = False
        print(f"Player {'X' if current_player == 1 else 'O'} has won!")
        print('''HH   HH   OOOOO    OOOOO   RRRRRR    AAAAA   YY   YY
HH   HH  OO   OO  OO   OO  RR   RR  AA   AA   YY YY
HH   HH  OO   OO  OO   OO  RR   RR  AA   AA    YYY
HH   HH  OO   OO  OO   OO  RRRRRR   AAAAAAA     Y
HHHHHHH  OO   OO  OO   OO  RR RR    AA   AA     Y
HH   HH  OO   OO  OO   OO  RR  RR   AA   AA     Y
HH   HH  OO   OO  OO   OO  RR   RR  AA   AA     Y
HH   HH  OO   OO  OO   OO  RR   RR  AA   AA     Y
HH   HH   OO OO    OO OO   RR   RR  AA   AA     Y
HH   HH    OOO      OOO    RR   RR  AA   AA     Y''')
    elif is_board_full(board):
        print("Board full! Tis' a draw. ")
    current_player *= -1