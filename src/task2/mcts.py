import os
import argparse
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data
from model import GNN
from utils import *
import math

class Node:
    def __init__(self, state, parent=None, depth=0):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = depth

    def add_child(self, child_state):
        child = Node(child_state, self, self.depth + 1)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def is_fully_expanded(self):
        return len(self.children) == 7

    def best_child(self, c_param=1.4):
        best_child = max(self.children, key=lambda child: child.value / child.visits +
                        c_param * math.sqrt((2 * math.log(self.visits) / child.visits)))

        return best_child


def mcts_search(root_state, model, max_depth=10):
    root = Node(root_state)

    if root.depth >= max_depth:
        return root.best_child(c_param=0).state

    while True:  
        node = root
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            if node.depth >= max_depth: 
                return node.state  
        # Expansion
        if not node.is_fully_expanded():
            next_states = generate_next_states(node.state)
            for state in next_states:
                if state not in [child.state for child in node.children]:
                    next_node = node.add_child(state)
                    break
            node = next_node

        # Simulation
        reward = simulate(node.state, model)

        # Backpropagation
        while node is not None:
            node.update(reward)
            node = node.parent


def g_func(aig_path):
    return eval_aig(aig_path)


def h_func(aig_path, model):
    _data = convert_aig_to_tensor(aig_path)
    x = torch.stack([_data['node_type'], _data['num_inverted_predecessors']], dim=1).float()
    data = Data(x=x, edge_index=_data['edge_index'])
    data.num_nodes = _data['num_nodes']
    data.num_edges = _data['num_edges']
    data.to(device)
    return -model(data)


def simulate(state, model):
    log_path, aig_path = obtain_aig(state)
    eval = h_func(aig_path , model) + g_func(aig_path)
    os.remove(log_path)
    os.remove(aig_path)
    return eval 


def generate_next_states(state):
    next_states = []
    for action in range(7):
        if '_' not in state:
            next_state = state + '_'
        else:
            next_state = state
        
        next_state += str(action)
        next_states.append(next_state)
    return next_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config.yaml', help="path to the config file")
    args, extras = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    device = cfg.device
    model = GNN(cfg.model.hidden_dim).to(device)
    model.load_state_dict(torch.load(cfg.train.checkpoint_path, map_location=device))
    model.eval()

    for filename in os.listdir(INIT_AIG_DIR):
        aig = filename.split('.')[0]
        state = mcts_search(aig, model, 100)

        eval = eval_decision(state)
        print(f'Final decision: {state}, eval: {eval}')

    # filename = 'c2670.aig'
    # aig = filename.split('.')[0]
    # state = mcts_search(aig, model, 100)

    # eval = eval_decision(state)
    # print(f'Final decision: {state}, eval: {eval}')
