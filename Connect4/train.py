### Train the models

from rules import init_board, legal_actions, place_piece, is_win, is_board_full, is_terminal
from network import ConnectNet
from encode_decode import encode, decode
from mcts import MCTS, Node
from pipeline import self_play, train_model

import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
'''
Simplest code yet! 
Just loops of:
1. data generation via MCTS,
2. model training
'''


# Set seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Initialize model
model = ConnectNet(input_shape=(2, 6, 7),
                   num_res_blocks=config.network.num_res_blocks,
                   num_actions=7,
                   num_filters=config.network.num_filters,
                   policy_conv_channels_out=config.network.policy_conv_channels_out,
                   value_conv_channels_out=config.network.value_conv_channels_out,
                   value_lin1_out=config.network.value_lin1_out)

optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)

replay_buffer = []
for iteration in range(1, config.pipeline.num_iterations + 1):

    # Separate training logs of different models
    print(f"======================= ITERATION {iteration} =======================")

    # Inference mode
    model.eval()

    # Generate multiple games
    for game in range(config.self_play.num_games_per_batch):
        examples = self_play(model=model,
                             num_simulations=config.self_play.num_simulations,
                             exploration_rate=config.self_play.exploration_rate,
                             dirichlet_alpha=config.self_play.dirichlet_alpha,
                             dirichlet_epsilon=config.self_play.dirichlet_epsilon)
        replay_buffer.extend(examples)
        print(f"Game No. {game} completed!")

    # Train mode
    model.train()

    # Train on data from multiple games
    for epoch in range(config.optimizer.num_epochs):
        np.random.shuffle(replay_buffer)
        train_model(model=model, optimizer=optimizer, batch_size=config.optimizer.batch_size, data=replay_buffer, lambda_reg=config.optimizer.lambda_reg, logging=True)

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "network_config": OmegaConf.to_container(config.network)
    },f'./models/model_v{iteration}.pth')




