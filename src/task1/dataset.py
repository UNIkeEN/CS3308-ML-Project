import pickle
import os
from tqdm import tqdm
from utils import obtain_aig, convert_aig_to_tensor

def process_data(raw_dir: str, save_path: str):
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

if __name__ == '__main__':
    process_data('../../dataset/task1/', '../../dataset/task1_processed.pkl')