import argparse
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data
import pickle
import os
from model import GNN, train, evaluate
from utils import obtain_aig, convert_aig_to_tensor

if __name__ == '__main__':
    # read and merge configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the config file")
    args, extras = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    device = cfg.device

    # load model and checkpoint
    model = GNN(cfg.model.hidden_dim).to(device)
    model.load_state_dict(torch.load(cfg.train.checkpoint_path))
    model.eval()

    # load test data and evaluate it
    with open(cfg.test.state_path, 'rb') as f:
        file_data = pickle.load(f)
        for i in range(len(file_data['input'])):
            input = file_data['input'][i]
            target = file_data['target'][-1] - file_data['target'][i]

            log_path, aig_path = obtain_aig(input)
            _data = convert_aig_to_tensor(aig_path)

            x = torch.stack([_data['node_type'], _data['num_inverted_predecessors']], dim=1).float()
            y = torch.tensor([target], dtype=torch.float)
            data = Data(x=x, edge_index=_data['edge_index'], y=y)
            data.num_nodes = _data['num_nodes']
            data.num_edges = _data['num_edges']
            data.to(cfg.device)

            os.remove(log_path)
            os.remove(aig_path)

            # output result
            print("Test state: ", input)
            print(f'Predicted and true value: {model(data).item()}, {data.y.item()}')


