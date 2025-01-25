import torch, torchvision
from torchvision import transforms
import os


triggers_dir = './triggers' # default triggers directory

# default target class (without loss of generality)
# the type source_class and cover_classes should be list
# source_class = [0,1,2,3,4,5,6,7,8,9]          #||| default source class for TaCT
# cover_classes = [0,1,2,3,4,5,6,7,8,9]      #||| default cover classes for TaCT
source_class = [1,2,5,9]          #||| default source class for TaCT
cover_classes = [6,7]      #||| default cover classes for TaCT
poison_seed = 0
record_poison_seed = True
record_model_arch = False


target_class = {
    'cifar10' : 5,
    'gtsrb' : 2,
    # 'gtsrb' : 12, # BadEncoder
    'imagenette': 0,
    'imagenet' : 0,
    'cifar100':5,
    'mnist':1,
}



trigger_default = {
    'mnist':{
        'badnet' : 'badnet_patch_32.png',
    },
    
    'cifar10': {
        'none' : 'none',
        'adaptive':'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
        'bpp': 'none',
        'WB': 'none',
    },
    'gtsrb': {
        'none' : 'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
    },
    'imagenet': {
        'none': 'none',
        'badnet': 'badnet_patch_256.png',
        'blend' : 'hellokitty_224.png',
        'trojan' : 'trojan_watermark_224.png',
        'SRA': 'phoenix_corner_256.png',
    },
    'cifar100': {
        'none' : 'none',
        'adaptive':'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
        'bpp': 'none',
        'WB': 'none',
    },
}




# adapitve-patch triggers for different datasets
adaptive_patch_train_trigger_names = {
    'cifar10': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'gtsrb': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'cifar100': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
}

adaptive_patch_train_trigger_alphas = {
    'cifar10': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'gtsrb': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'cifar100': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
}

adaptive_patch_test_trigger_names = {
    'cifar10': [
        'phoenix_corner2_32.png',
        'badnet_patch4_32.png',
    ],
    'gtsrb': [
        'firefox_corner_32.png',
        'trojan_square_32.png',
    ],
    'cifar100': [
        'phoenix_corner2_32.png',
        'badnet_patch4_32.png',
    ],
}

adaptive_patch_test_trigger_alphas = {
    'cifar10': [
        1,
        1,
    ],
    'gtsrb': [
        1,
        1,
    ],
        'cifar100': [
        1,
        1,
    ],
}

