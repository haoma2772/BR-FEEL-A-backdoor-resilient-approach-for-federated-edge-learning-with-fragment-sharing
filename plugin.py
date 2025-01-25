import torch
import random
import numpy as np
from utility import load_config
from torch.utils.data import random_split, Subset
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from inject_backdoor import create_poisoned_set
import copy
import pickle
import os
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class CustomGTSRBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.training_dir = os.path.join(data_dir, 'Training')
        
        # Load image paths and labels
        for class_id in range(43):
            class_dir = os.path.join(self.training_dir, f'{class_id:05d}')
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.ppm'):
                    image_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(image_path)
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def load_dataset(config, trained=True):
    dataset_name = config['dataset']['name']
    data_dir = 'data'
    resize_transform = transforms.Resize((224, 224))

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            resize_transform,
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='data', train=trained, transform=transform, download=True)

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=trained, transform=transform, download=True)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=trained, transform=transform, download=True)

    elif dataset_name == 'imagenet1k':
        transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    elif dataset_name == 'gtsrb':
        transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if trained:
            train_dataset = torchvision.datasets.GTSRB(root=data_dir, split='train', transform=transform, download=True)
        else:
            train_dataset = torchvision.datasets.GTSRB(root=data_dir, split='test', transform=transform, download=True)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset

def split_data(dataset, train_ratio, random_seed=None):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    if random_seed is not None:
        torch.manual_seed(random_seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def dirichlet_split_noniid(train_labels, alpha, n_clients, n_classes):
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i].append(idcs)
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs

def get_dataset_labels(dataset, config):
    if config['dataset']['name'] == 'gtsrb':
        return [target for _, target in dataset._samples]
    else:
        raise ValueError("Unsupported dataset. Please implement the 'get_labels' method for this dataset.")

def dirichlet_distribution(dataset, config):
    total_samples = len(dataset)
    iid = config['federated_learning']['iid']
    n_clients = config['federated_learning']['client_number']

    if iid:
        samples_per_client = total_samples // n_clients
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        subsets = [indices[i * samples_per_client:(i + 1) * samples_per_client] for i in range(n_clients)]
    else:
        DIRICHLET_ALPHA = config['federated_learning']['dirichlet_rate']
        if config['dataset']['name'] == 'cifar10':
            train_labels = np.array(dataset.targets)
            num_cls = 10
        elif config['dataset']['name'] == 'gtsrb':
            train_labels = np.array(get_dataset_labels(dataset, config))
            num_cls = 43
        elif config['dataset']['name'] == 'cifar100':
            train_labels = np.array(dataset.targets)
            num_cls = 100
        elif config['dataset']['name'] == 'mnist':
            train_labels = np.array(dataset.targets)
            num_cls = 10
        else:
            raise ValueError("Unsupported dataset")

        subsets = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=n_clients, n_classes=num_cls)

    local_dataset = [Subset(dataset, subset_indices) for subset_indices in subsets]
    return local_dataset

def get_dataset(backdoor_index, config):
    dataset_name = config['dataset']['name']
    Dalpha = config['federated_learning']['dirichlet_rate']
    poisoned_rate = config['poisoned_paras']['poisoned_rate']
    num_client = config['federated_learning']['client_number']
    cover_rate = config['poisoned_paras']['cover_rate']

    file_path = os.path.join('data', 'distribution', dataset_name, f'dalpha={Dalpha}')
    os.makedirs(file_path, exist_ok=True)
    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test_list.pkl')

    with open(dataset_train_path, 'rb') as f:
        dataset_train_list = pickle.load(f)
    with open(dataset_test_path, 'rb') as f:
        dataset_test_list = pickle.load(f)

    train_dataset_list = []
    for client_id in range(num_client):
        if client_id in backdoor_index:
            poisoned_train_dataset = create_poisoned_set(
                dataset=copy.deepcopy(dataset_train_list[client_id]),
                poison_rate=poisoned_rate,
                cover_rate=cover_rate,
                config_list=config
            )
            train_dataset_list.append(poisoned_train_dataset)
        else:
            train_dataset_list.append(copy.deepcopy(dataset_train_list[client_id]))

    return train_dataset_list, copy.deepcopy(dataset_test_list)
