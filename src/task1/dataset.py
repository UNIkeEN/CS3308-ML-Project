import pickle
import os
import random
from tqdm import tqdm
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import obtain_aig, convert_aig_to_tensor

def generate_data(
        raw_dir: str, 
        save_path: str, 
        only_last_state: bool = False,
        ratio: float = 1
    ):
    """
    Generate dataset for task 1.

    Params:
    raw_dir(`str`): directory of raw data
    save_path(`str`): location to store the processed data, ending with the suffix `.pkl`
    only_last_state(`bool`): whether to only use the last state
    ratio(`float`): from 0 to 1, ratio for randomly selecting data to generate the dataset
    """

    processed_inputs = set()

    all_files = [f for f in os.listdir(raw_dir) if f.endswith('.pkl')]
    assert 0 < ratio <= 1
    selected_files = random.sample(all_files, int(len(all_files) * ratio))

    for filename in tqdm(selected_files):
        if filename.endswith('.pkl'):
            filepath = os.path.join(raw_dir, filename)
            
            with open(filepath, 'rb') as f:
                file_data = pickle.load(f)
                input_states = file_data['input']
                target_values = file_data['target']

                _iter = zip(input_states[-1:], target_values[-1:]) if only_last_state else zip(input_states, target_values)

                for input, target in _iter:
                    if input in processed_inputs:
                        continue
                    
                    try:
                        # generate aig using yosys
                        log_path, aig_path = obtain_aig(input)
                        
                        # generate features from aig
                        data = convert_aig_to_tensor(aig_path)
                        data['input'] = input
                        data['target'] = target

                        with open(save_path, 'ab') as f:
                            pickle.dump(data, f)

                        processed_inputs.add(input)
                        os.remove(log_path)
                        os.remove(aig_path)
                        # print(data)

                    except Exception as e:
                        print(f"Error processing {input}: {e}")


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

    data_objects = []
    try:
        with open(pkl_path, 'rb') as f:
            while True:
                try:
                    # convert data to PyTorch Geometric Data objects
                    item = pickle.load(f)
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
                except EOFError:
                    break
    except Exception as e:
        print(f"Error reading data from {pkl_path}: {e}")

    num_samples = len(data_objects)
    num_test = int(test_ratio * num_samples)
    num_train = num_samples - num_test

    # split the dataset
    train_data, test_data = random_split(data_objects, [num_train, num_test])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    generate_data('../../dataset/task1/', '../../dataset/task1_base.pkl')