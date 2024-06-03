import argparse
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import GNN, train, evaluate
from dataset import generate_data, load_data

if __name__ == '__main__':
    # read and merge configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the config file")
    args, extras = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    device = cfg.device

    # generate and load data
    if cfg.dataset.skip_generate == False:
        print('Generating Data...')
        generate_data(
            cfg.dataset.raw_dir, 
            cfg.dataset.path, 
            cfg.dataset.only_last_state, 
            cfg.dataset.selection_ratio
        )
    else:
        print("Skipped Data Generation")
    print('Loading Data...')
    train_loader, test_loader = load_data(
        cfg.dataset.path, 
        test_ratio=cfg.dataset.test_ratio,
        batch_size=cfg.dataset.batch_size
        )

    # define model, loss and optimizer
    model = GNN(cfg.model.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.MSELoss()

    # train
    best_test_loss = float('inf')
    print('Begin Training...')
    for epoch in range(cfg.train.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{cfg.train.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), cfg.train.checkpoint_path)
