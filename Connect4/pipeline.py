### Training pipeline

from rules import init_board, legal_actions, place_piece, is_win, is_board_full, is_terminal
from network import ConnectNet
from encode_decode import encode, decode
from mcts import MCTS, Node

import numpy as np
import torch
import torch.nn.functional as F

'''
AlphaZero works by completing rounds of 
1. Using MCTS to generate training data, with the previous models version,
2. Training the next models iteration with said data.

It's as simple as that!
I will also be saving each models version and MCTS tree into files,
in order to document progress. 
'''


## Generate target data for the models (s, π, z)
def self_play(model, num_simulations, exploration_rate=0.1, dirichlet_alpha=0.0, dirichlet_epsilon=0.0):
    history = []

    state = init_board() # board state
    current_player = 1

    # Self-play game
    while not is_terminal(state): # a game is short enough for us not to care about limits

        mcts = MCTS(model, exploration_rate=exploration_rate)
        mcts.root = Node(current_player=current_player, state=state, parent=None, action=0, prior=0)

        for simulation in range(num_simulations):
            mcts.simulate(dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon)

        visits = np.array([
            mcts.root.children[a].visit_count
            if a in mcts.root.children else 0
            for a in range(7)
        ], dtype=float)

        pi = visits / visits.sum()

        history.append([state, pi, None, current_player])
        state = place_piece(state, current_player, np.random.choice(7, p=pi))

        current_player = -current_player

    winner = None
    if is_win(state, player=1):
        winner = 1
    elif is_win(state, player=-1):
        winner = -1

    # Propagate result of self-player game
    for example in history:
        player = example[3]

        if winner is None: # Draw
            z = 0
        elif winner == player:
            z = 1
        else:
            z = -1

        example[2] = z

    return history


# Train new models based on training data
def train_model(model, optimizer, batch_size, data, lambda_reg, logging=False):
    # Total losses for evaluation (want to calculate mean later)
    total_policy_kl = 0
    total_value_loss = 0

    # Train
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        # Prepare network input (i.e. booleans for each player)
        s = torch.tensor(np.stack([
            np.stack([state == player, state == -player])
            for state, pi, z, player in batch
        ]), dtype=torch.float32)

        s = s.unsqueeze(0) # to account for BatchNorm expecting a 4D tensor, will use mini-batches later

        # Prepare targets
        pi = torch.tensor(np.stack([
            pi for _, pi, _, _ in batch
             ]), dtype=torch.float32)

        z = torch.tensor([
            z for state, pi, z, player in batch
        ], dtype=torch.float32)

        # Forward
        p, v = model(s)

        # Calculate loss
        policy_ce = -(pi * F.log_softmax(p, dim=-1)).sum()
        policy_kl = F.kl_div(F.log_softmax(p, dim=-1), pi, reduction="sum") # converges to zero as models matches MCTS
        value_loss = F.mse_loss(v.reshape(-1), z.reshape(-1))

        regularization_loss = lambda_reg * torch.nn.utils.get_total_norm(model.parameters())

        loss = policy_ce + value_loss + regularization_loss

        # Training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        total_policy_kl += policy_kl.item() * len(batch) # len(batch) might not match batch_size, e.g. at the end of data
        total_value_loss += value_loss.item() * len(batch)

    if logging:
        print(
            f"Policy KL: {total_policy_kl / len(data):.4f}, "
            f"Value MSE: {total_value_loss / len(data):.4f}, "
            f"Total: {(total_policy_kl + total_value_loss) / len(data):.6f}"
        )

    return model




### TESTING ###
if __name__ == '__main__':
    # TODO: Remove!
    class dummy_model():
        def __init__(self):
            pass


        def forward(self, state):
            return torch.tensor([[1., 0., 0., 0., 0., 0., 0.]]), torch.tensor([[1.]])
    # Initialize models
    model = ConnectNet(input_shape=(2, 6, 7),
                       num_res_blocks=2,
                       num_actions=7,
                       num_filters=32,
                       policy_conv_channels_out=2,
                       value_conv_channels_out=1,
                       value_lin1_out=8)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    data = self_play(model)
    # model2 = train_model(epochs=120, model=model, optimizer=optimizer, data=data)












