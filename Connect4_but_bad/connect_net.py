from typing import Any
from torch import nn
import torch
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    # basic residual block
    def __init__(self, num_filters: int):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters,
                      out_channels=num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU()
            )

        # In the 2nd block, ReLU is applied after residual addition
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters,
                      out_channels=num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.BatchNorm2d(num_features=num_filters)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


# given a board state and parameter, output value and policy
class ConnectNet(nn.Module):
    def __init__(self,
                 input_shape = (2, 6, 7),
                 num_filters=128,
                 num_res_block = 2):
        super(ConnectNet, self).__init__()
        c, h, w = input_shape # one channel per player

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(
                num_features=num_filters
            ),
            nn.ReLU()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=32
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6*7*32, 7),
            nn.Softmax()
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=3,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*6*7, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
            )


    def forward(self, s):
        s = self.conv_block(s)
        s = self.res_blocks(s)

        v = self.value_head(s)
        p = self.policy_head(s)
        return v, p






