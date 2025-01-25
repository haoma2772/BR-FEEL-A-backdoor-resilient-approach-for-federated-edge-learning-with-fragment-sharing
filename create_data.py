from utility import load_config
from plugin import dirichlet_distribution
import pickle
import os
from plugin import load_dataset
from utility import load_config

def generate_noniid_distribution(config):
    # Retrieve various parameters for training
    client_number = config['federated_learning']['client_number']
    split_rate = config['federated_learning']['split_rate']

    train_dataset = load_dataset(config, trained=True)
    test_dataset = load_dataset(config, trained=False)
    # Returns the dataset
    dataset_train_list = dirichlet_distribution(train_dataset, config)

    dataset_name = config['dataset']['name']
    Dalpha = config['federated_learning']['dirichlet_rate']

    # Save the generated Dirichlet distribution for later use
    file_path = os.path.join('data', 'distribution', dataset_name)
    tmp = 'dalpha=' + str(Dalpha)
    file_path = os.path.join(file_path, tmp)

    os.makedirs(file_path, exist_ok=True)
    # Save the dataset list as .pkl files
    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test_list.pkl')
    with open(dataset_train_path, 'wb') as f:
        pickle.dump(dataset_train_list, f)

    with open(dataset_test_path, 'wb') as f:
        pickle.dump(test_dataset, f)

if __name__ == '__main__':
    path = 'config.yaml'
    config_list = load_config(path)
    generate_noniid_distribution(config=config_list)
