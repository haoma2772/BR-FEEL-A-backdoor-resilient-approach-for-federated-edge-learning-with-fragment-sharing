import yaml
import torchvision.transforms as transforms
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import random_split
import os,base_config


def load_config(config_file):
    """
    Load configuration settings from a YAML file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(config):
    """
    Set random seed for reproducibility.
    """
    seed = config['general']['random_seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_poison_transform(config_list, poison_type, dataset_name, target_class, source_class=1, 
                         cover_classes=[5, 7], is_normalized_input=False, trigger_transform=None, 
                         alpha=0.2, trigger_name=None):
    """
    Generate a poison transformation based on the specified dataset and poisoning method.

    Args:
        config_list: Configuration dictionary.
        poison_type: The type of poisoning (e.g., 'badnet', 'blend').
        dataset_name: Name of the dataset (e.g., 'cifar10', 'gtsrb').
        target_class: The target class for poisoning.
        source_class: Source class for poisoning (used in some methods like 'TaCT').
        cover_classes: List of additional classes to cover (used in specific methods).
        is_normalized_input: Whether the input is normalized.
        trigger_transform: Predefined transformation for the trigger.
        alpha: Blending ratio for methods like 'blend'.
        trigger_name: Name of the trigger file.

    Returns:
        Poison transformation function.
    """
    # Default trigger name handling
    if trigger_name is None:
        if dataset_name != 'imagenette':
            trigger_name = base_config.trigger_default[dataset_name][poison_type]
        else:
            if poison_type == 'badnet':
                trigger_name = 'badnet_high_res.png'
            else:
                raise NotImplementedError(f'{poison_type} is not implemented for imagenette.')

    # Dataset-specific transformations and sizes
    if dataset_name in ['mnist', 'gtsrb', 'cifar10', 'cifar100']:
        img_size = 224
    elif dataset_name in ['imagenette', 'imagenet']:
        img_size = 224
    else:
        raise NotImplementedError(f'Undefined dataset: {dataset_name}')

    resize_transform = transforms.Resize(img_size, antialias=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        resize_transform,
    ])

    poison_transform = None
    trigger = None
    trigger_mask = None

    if poison_type in ['basic', 'badnet', 'blend', 'clean_label', 'refool',
                       'adaptive_blend', 'adaptive_patch', 'adaptive_k_way',
                       'SIG', 'TaCT', 'WaNet', 'SleeperAgent', 'none',
                       'badnet_all_to_all', 'trojan', 'SRA', 'bpp']:
        if trigger_transform is None:
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
            ])

        # Load trigger and trigger mask
        if trigger_name != "none":
            trigger_path = os.path.join(base_config.triggers_dir, trigger_name)
            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)

            trigger_mask_path = os.path.join(base_config.triggers_dir, f'mask_{trigger_name}')
            if os.path.exists(trigger_mask_path):
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = trigger_transform(trigger_mask)
            else:
                trigger_mask = (trigger[0] > 0).float()

        # Define poison transformations based on method
        if poison_type == 'badnet':
            from poison_tool_box import badnet
            poison_transform = badnet.poison_transform(
                img_size=img_size, trigger_mark=trigger, trigger_mask=trigger_mask, target_class=target_class
            )
        elif poison_type == 'blend':
            from poison_tool_box import blend
            poison_transform = blend.poison_transform(
                img_size=img_size, trigger=trigger, target_class=target_class, alpha=alpha
            )
        elif poison_type == 'WaNet':
            s = 0.5
            k = 4
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = torch.nn.functional.interpolate(ins, size=img_size, mode="bicubic", align_corners=True)
            identity_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, steps=img_size), 
                                                       torch.linspace(-1, 1, steps=img_size)), 2)[None, ...]
            from poison_tool_box import WaNet
            poison_transform = WaNet.poison_transform(
                img_size=img_size, identity_grid=identity_grid, noise_grid=noise_grid, s=s, k=k,
                target_class=target_class
            )
        elif poison_type == 'SIG':
            from poison_tool_box import SIG
            poison_transform = SIG.poison_transform(
                img_size=img_size, target_class=target_class, delta=30 / 255, f=6,
                has_normalized=is_normalized_input, cuda_devices=config_list['general']['cuda_number']
            )
        else:
            raise NotImplementedError(f'{poison_type} is not defined.')

        return poison_transform

    elif poison_type == 'dynamic':
        # Dynamic poisoning (requires pre-trained generator)
        normalizer = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
        denormalizer = transforms.Compose([transforms.Normalize(mean=(-0.5 / 0.5), std=(1.0 / 0.5)), ])

        if dataset_name == 'cifar10':
            channel_init = 32
            steps = 3
            input_channel = 3
            ckpt_path = './models/all2one_cifar10_ckpt.pth.tar'
        elif dataset_name == 'gtsrb':
            channel_init = 32
            steps = 3
            input_channel = 3
            ckpt_path = './models/all2one_gtsrb_ckpt.pth.tar'
        else:
            raise Exception("Invalid dataset for dynamic poisoning.")

        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                'Download pre-trained generator from https://github.com/VinAIResearch/input-aware-backdoor-attack-release'
            )

        from poison_tool_box import dynamic
        poison_transform = dynamic.poison_transform(
            ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
            input_channel=input_channel, normalizer=normalizer, denormalizer=denormalizer, 
            target_class=target_class, has_normalized=is_normalized_input
        )
        return poison_transform

    else:
        raise NotImplementedError(f'Undefined poison type: {poison_type}')
