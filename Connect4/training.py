### Train the models

from rules import init_board, legal_actions, place_piece, is_win, is_board_full, is_terminal
from network import ConnectNet
from encode_decode import encode, decode
from mcts import MCTS, Node
from pipeline import self_play, train_model

import numpy as np
import torch
import torch.nn.functional as F

'''
Simplest code yet! 
Just loops of:
1. data generation via MCTS,
2. model training
'''

# Initialize models
model = ConnectNet(input_shape=(2, 6, 7),
                   num_res_blocks=2,
                   num_actions=7,
                   num_filters=32,
                   policy_conv_channels_out=2,
                   value_conv_channels_out=1,
                   value_lin1_out=8)

torch.save(model.state_dict(), f'./models/model0.pth') # Random weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_iterations = 10
num_games_per_batch = 20
num_epochs = 5

replay_buffer = []
for iteration in range(1, num_iterations + 1):

    # Separate training logs of different models
    print(f"======================= ITERATION {iteration} =======================")

    # Inference mode
    model.eval()

    # Generate multiple games
    for game in range(num_games_per_batch):
        examples = self_play(model)
        replay_buffer.extend(examples)
        print(f"Game No. {game} completed!")

    # Train mode
    model.train()

    # Train on data from multiple games
    for epoch in range(num_epochs):
        np.random.shuffle(replay_buffer)
        train_model(model, optimizer, replay_buffer, logging=True)

    # Save model
    torch.save(model.state_dict(), f'./models/model_v{iteration}.pth')




