import pickle
import os
from tqdm import tqdm
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import obtain_aig, convert_aig_to_tensor

def generate_data(raw_dir: str, save_path: str):
    """
    Generate dataset for task 1.

    Params:
    raw_dir(`str`): directory of raw data
    save_path(`str`): location to store the processed data, ending with the suffix `.pkl`
    """

    processed_inputs = set()
    all_data = []

    for filename in tqdm(os.listdir(raw_dir)):
        if filename.endswith('.pkl'):
            filepath = os.path.join(raw_dir, filename)
            
            with open(filepath, 'rb') as f:
                file_data = pickle.load(f)
                input_states = file_data['input']
                target_values = file_data['target']

                for input, target in zip(input_states, target_values):
                    if input in processed_inputs:
                        continue
                    
                    try:
                        # generate aig using yosys
                        log_path, aig_path = obtain_aig(input)
                        
                        # generate features from aig
                        data = convert_aig_to_tensor(aig_path)
                        data['input'] = input
                        data['target'] = target
                        all_data.append(data)
                        processed_inputs.add(input)

                        os.remove(log_path)
                        os.remove(aig_path)
                        # print(data)

                    except Exception as e:
                        print(f"Error processing {input}: {e}")

    with open(save_path, 'wb') as f:
        pickle.dump(all_data, f)


def load_data(pkl_path, test_ratio=0.2, batch_size=32):
    """
    Load data from a pkl file and split into training and testing sets.

    Params:
    pkl_path (str): path to the pkl file containing the data.
    test_ratio (float): proportion of the data to include in the test split.
    batch_size (int): batch size for DataLoader.

    Returns:
    train_loader (DataLoader): DataLoader for the training set.
    test_loader (DataLoader): DataLoader for the test set.
    """

    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)

    # convert data_list to PyTorch Geometric Data objects
    data_objects = []
    for item in data_list:
        edge_index = item['edge_index']
        node_type = item['node_type']
        num_inverted_predecessors = item['num_inverted_predecessors']
        target = item['target']

        x = torch.stack([node_type, num_inverted_predecessors], dim=1).float()
        y = torch.tensor([target], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = item['num_nodes']
        data.num_edges = item['num_edges']
        data_objects.append(data)

    num_samples = len(data_objects)
    num_test = int(test_ratio * num_samples)
    num_train = num_samples - num_test

    # split the dataset
    train_data, test_data = random_split(data_objects, [num_train, num_test])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    generate_data('../../dataset/task1/', '../../dataset/task1_small.pkl')