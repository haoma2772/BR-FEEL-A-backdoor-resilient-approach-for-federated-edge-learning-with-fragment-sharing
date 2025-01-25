import torchvision
from torchvision import transforms
from utility import load_config

def download_dataset(config_list):
    dataset_name = config_list['dataset']['name']
    save_path = config_list['dataset']['save_path']
    # Define a mapping from dataset names to torchvision dataset classes
    dataset_mapping = {
        'mnist': torchvision.datasets.MNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'imagenet1k': torchvision.datasets.ImageNet,
    }

    # Check if the input dataset name is valid
    if dataset_name not in dataset_mapping:
        print(f"Dataset '{dataset_name}' is not supported. Please choose from 'mnist', 'cifar10', 'cifar100', or 'imagenet1k'.")
        return

    # Use torchvision-provided transforms for preprocessing (modify as needed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create a dataset object and download the data
    dataset_class = dataset_mapping[dataset_name]
    dataset = dataset_class(root=save_path, train=True, download=True, transform=transform)

    print(f"Successfully downloaded and saved the '{dataset_name}' dataset to path '{save_path}'.")

if __name__ == "__main__":
    path = 'config.yaml'
    config = load_config(path)
    download_dataset(config)
