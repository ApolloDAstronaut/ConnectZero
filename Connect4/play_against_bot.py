### Code to play against trained model

from rules import place_piece, init_board, is_win, is_board_full, legal_actions
import numpy as np
import torch

from network import ConnectNet

from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")

MODEL_VERSION = 10

# Load trained model
checkpoint = torch.load(f"./models/model_v{MODEL_VERSION}.pth",
               map_location="cpu",
               weights_only=True)

network_config = checkpoint["network_config"]
model = ConnectNet(input_shape=(2, 6, 7),
                   num_actions=7,
                   **network_config
                   )
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

board = init_board()
print(np.where(board == 1, 'X', np.where(board == -1, 'O', ' ')))

current_player = 1
run = True

while run:
    if current_player == -1:  # Player is bot

        # Encode board from current player's perspective
        encoded_board = torch.tensor(
            np.stack([
                board == current_player,
                board == -current_player
            ]),
            dtype=torch.float32
        ).unsqueeze(0)

        # Forward propagation
        with torch.no_grad():
            policy, value = model(encoded_board)

        # Get raw policy logits
        policy = policy.squeeze(0).numpy()

        # Mask illegal actions
        legal_mask = np.array(legal_actions(board), dtype=bool)
        policy[~legal_mask] = -np.inf

        selected_move = int(np.argmax(policy))

        print(f"Player O selected column {selected_move + 1}!")
        board = place_piece(board, current_player, selected_move)

    else:
        selected_move = int(input("Player X: ")) - 1
        board = place_piece(board, current_player, selected_move)

    # Display using X and O
    print(np.where(board == 1, 'X', np.where(board == -1, 'O', ' ')))

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
        run = False
        print("Board full! Tis' a draw.")

    current_player *= -1
