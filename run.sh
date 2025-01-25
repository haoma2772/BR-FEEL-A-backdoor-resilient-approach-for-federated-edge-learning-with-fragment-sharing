#!/bin/bash

# Source the conda.sh script to enable conda commands
source /root/miniconda3/etc/profile.d/conda.sh

# activate vir envs
conda activate pytorch

# run Python script and pass the config path
# python mask_main.py --config_path=./config/resnet34_config.yaml
python mask_main.py --config_path=./config/mobilenetv2_config.yaml
# python mask_main.py --config_path=./config/mlp_config.yaml

# close vir env
conda deactivate
