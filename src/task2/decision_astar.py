import os
import argparse
from omegaconf import OmegaConf
import heapq
import torch
from torch_geometric.data import Data
from model import GNN
from utils import obtain_aig, eval_aig, eval_decision, convert_aig_to_tensor


def g_func(aig_path):
    return eval_aig(aig_path)


def h_func(aig_path, model):
    _data = convert_aig_to_tensor(aig_path)
    x = torch.stack([_data['node_type'], _data['num_inverted_predecessors']], dim=1).float()
    data = Data(x=x, edge_index=_data['edge_index'])
    data.num_nodes = _data['num_nodes']
    data.num_edges = _data['num_edges']
    data.to(device)
    return model(data)


def astar_search(aig, max_steps, model):
    priority_queue = []
    heapq.heappush(priority_queue, (0, (aig + '_', 0)))

    while priority_queue:
        print(priority_queue)
        _, (state, step) = heapq.heappop(priority_queue)
        print('Processing ' + state)

        if step == max_steps:
            return state

        for action in range(7):
            next_state = state + str(action)
            log_path, aig_path = obtain_aig(next_state)

            f = g_func(aig_path) + h_func(aig_path, model)
            heapq.heappush(priority_queue, (-f.item(), (next_state, step + 1)))

            os.remove(log_path)
            os.remove(aig_path)


synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

libFile = './lib/7nm/7nm.lib'
INIT_AIG_DIR = './InitialAIG/train/'

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
        state = astar_search(aig, 10, model)

        eval = eval_decision(state)
        print(f'Final decision: {state}, eval: {eval}')
