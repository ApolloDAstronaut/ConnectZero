import numpy as np
from game import *
from view_board import draw_board, render
import math

num_simulations = 100
exploration_rate = 1

board = np.array(
        [[ 0,-1,-1,-1, 1, 0,-1],
         [ 0, 1,-1, 1, 1, 0, 1],
         [-1, 1,-1, 1, 1, 0,-1],
         [ 1,-1, 1,-1,-1, 0,-1],
         [-1,-1, 1,-1, 1, 1,-1],
         [-1, 1, 1,-1, 1,-1, 1]]
         )

# bonus for good moves + exploration
def ucb_score(parent, child):
    prior_score = exploration_rate * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    value_score = child.value
    return prior_score + value_score

# for testing purposes
def dummy_model_predict(board):
    value_head = 0.5
    policy_head = [0.5, 0, 0, 0, 0, 0.5, 0]
    return value_head, policy_head

class Node:
    # initialize node
    def __init__(self, prior, turn, state):
        self.prior = prior
        self.turn = turn # who's turn it is
        self.state = state
        self.children = {}
        self.value = 0
        self.visit_count = 0

    # expand children of node
    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if prob > 0:
                next_state = place_piece(board=self.state, player=self.turn, action=action)
                self.children[action] = Node(prior=prob, turn=-self.turn, state=next_state)

    # select child with the highest ucb score
    def select_child(self):
        max_score = -math.inf
        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > max_score:
                selected_child = child
                selected_action = action
                max_score = score
        return selected_action, selected_child

def mcts(board, num_simulations=100):
    # initialize root
    root = Node(prior=0, turn=1, state=board)

    # expand the root
    value, action_probs = dummy_model_predict(board)
    root.expand(action_probs=action_probs)

    # iterate through simulations
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # select best child until we reach an unexplored node
        while len(node.children) > 0:
            action, node = node.select_child()
            search_path.append(node)

        value = None
        # calculate value once we reach a leaf node
        if is_board_full(board=node.state):
            value = 0
        if is_win(board=node.state, player=1):
            value = 1
        if is_win(board=node.state, player=-1):
            value = -1

        if value is None:
            # if game is not over, get values from network and expand
            value, action_probs = dummy_model_predict(board)
            node.expand(action_probs=action_probs)

        # back up the values
        for node in search_path:
            node.value += value
            node.visit_count += 1
    return root



