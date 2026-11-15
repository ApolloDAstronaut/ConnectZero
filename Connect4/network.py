### Implement the AlphaZero-style neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

'''
A deep neural network f_theta(s) = (p, v).
    - Convolutional block to analyse geometric structures
    - Residual blocks for speed

Takes the board position s, outputs
    - p --> the policy, a vector of move probabilities for each action
    - v --> the value, the expected chance of winning for the current player from this state
    
Architecture:
    - board input (2 channels, one for each player, 1's where their pieces are)
    - convolutional layer + batch norm + ReLU
    - a set amount of residual blocks
    - a policy and a value head
    
    For the sake of simplicity, the initial convolutional layer + residual blocks will maintain the same in/out size,
    a parameter named [num_filters], since internally it's all just convolutions. 
    
    Residual Blocks:
        - 2x (convolutional layer + batch norm + ReLU)
        
    Policy Head:
        - convolutional block + batch norm + ReLU
        - linear layer --> 7 output neurons
        - (Softmax performed during loss calculation for stability)
    
    Value Head:
        - convolutional block + batch norm + ReLU
        - 2 linear layers --> scalar tensor output
        - tanh activation

Function arguments are roughly ordered by their order in forward-propagation and by importance
'''
## HELPER FUNCTIONS ##

# Calculate the output dimensions of a convolutional layer
# Assumes a square kernel
def calc_conv2d_output_dim(h_w, kernel_size=1, stride=1, padding=1):
    """input tuple (h, w), containing the height and width of the input"""
    h_out = math.floor((h_w[0] - kernel_size + 2 * padding) / stride) + 1
    w_out = math.floor((h_w[1] - kernel_size + 2 * padding) / stride) + 1
    return h_out, w_out

## NEURAL NETWORK ##

# Initial convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)



# Intermediate residual block
class ResNetBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(x))
        out += residual
        out = F.relu(out)
        return out


# Policy head
class PolicyHead(nn.Module):
    def __init__(self, in_channels, conv_channels_out, conv_hw_out, num_actions):
        super(PolicyHead, self).__init__()

        h, w = conv_hw_out
        flattened_size = conv_channels_out * h * w

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels_out,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(conv_channels_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, out_features=num_actions)
        )

    def forward(self, x):
        return self.policy_head(x)

# Value head
class ValueHead(nn.Module):
    def __init__(self, in_channels, conv_channels_out,
                 conv_hw_out, lin1_out):
        super(ValueHead, self).__init__()

        h, w = conv_hw_out
        flattened_size = conv_channels_out * h * w

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels_out,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(conv_channels_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, out_features=lin1_out),
            nn.ReLU(),
            nn.Linear(in_features=lin1_out, out_features=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.value_head(x)



# Full neural network
class ConnectNet(nn.Module):
    def __init__(self,
        input_shape,
        num_res_blocks,
        num_actions,
        num_filters,
        policy_conv_channels_out,
        value_conv_channels_out,
        value_lin1_out,
        ):

        super(ConnectNet, self).__init__()
        c, h, w = input_shape

        # Compute size of output of the convolution layers in the value & policy heads
        conv_out_hw = calc_conv2d_output_dim((h, w), 3, 1, 1)
        conv_size_out = conv_out_hw[0] * conv_out_hw[1]

        # Initial convolution block
        self.conv_block = ConvBlock(in_channels=c, out_channels=num_filters)

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Policy head, return policy logits
        self.policy_head = PolicyHead(in_channels=num_filters, conv_channels_out=policy_conv_channels_out,
                                      conv_hw_out=conv_out_hw, num_actions=num_actions)

        # Value head, return estimated value
        self.value_head = ValueHead(in_channels=num_filters, conv_channels_out=value_conv_channels_out,
                                    conv_hw_out=conv_out_hw, lin1_out=value_lin1_out)


    def forward(self, x):
        # Given a board state x, predict the raw logits for the probability distribution of actions,
        # and the evaluated value of the board, from the perspective of the current player

        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)

        # Predict raw logit distribution
        pi_logits = self.policy_head(features)

        # Predict evaluated value from the current player's perspective
        value = self.value_head(features)

        return pi_logits, value

