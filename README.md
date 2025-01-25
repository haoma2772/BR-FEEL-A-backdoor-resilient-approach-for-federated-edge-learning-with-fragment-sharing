# BR-FEEL: A Backdoor Resilient Approach for Federated Edge Learning with Fragment-Sharing

## Abstract

In resource-constrained Federated Edge Learning (FEEL) systems, fragment-sharing enables clients to cooperatively train giant models with billions of parameters. Unlike traditional federated learning, where the entire local model is trained and shared, fragment-sharing allows clients to train and share only selected parameter fragments based on their storage, computational, and networking capabilities.

However, this selective sharing introduces new challenges:
- Backdoor attacks hidden in fragments are harder to detect when the full model is not shared.
- The security of the overall FEEL system is significantly compromised.

To address these challenges, we propose **BR-FEEL**:
1. **Backdoor-Resilient Twin Model**: Each benign client integrates fragments from others into a twin model to identify and handle malicious parameters.
2. **Knowledge Distillation**: Clean knowledge from the twin model is transferred back to the local model through a carefully designed distillation process.

Our experiments on CIFAR-10 and GTSRB datasets, using MobileNetV2 and ResNet-34, show that **BR-FEEL** reduces attack success rates by over 90% compared to other baselines under various attack methods.

---

## Repository Structure

```plaintext
├── config/                      # YAML configuration files for different experiments
│   ├── mobilenetv2_config.yaml  # Configuration for MobileNetV2-based experiments
│   ├── resnet34_config.yaml     # Configuration for ResNet34-based experiments
├── data/                        # Dataset storage
├── defense/                     # Defense strategies implementations
├── mask_backdoor_res/           # Results for experiments with masking backdoors
├── models/                      # Pretrained models and model architectures
├── poison_tool_box/             # Tools for generating poisoned datasets
├── poisoned_set/                # Directory for poisoned datasets
├── rebuttal/                    # Results for rebuttal experiments
├── triggers/                    # Trigger images for backdoor attacks
├── VisualTools/                 # Visualization tools for experiment results
├── wandb/                       # Logs generated using Weights and Biases
├── base_config.py               # Base configurations for the experiments
├── create_data.py               # Script for creating datasets
├── CustomDataset.py             # Custom dataset loader
├── download_data.py             # Script for downloading required datasets
├── inject_backdoor.py           # Code for injecting backdoor attacks
├── mask_main.py                 # Main script for running backdoor masking experiments
├── mask_utils.py                # Utility functions for masking-based defenses
├── model.py                     # Model definitions
├── motivation.py                # Script for running motivation experiments
├── plugin.py                    # Additional plugin utilities
├── requirement.txt              # Dependencies and required Python packages
├── run.sh                       # Shell script for running experiments
├── test_model.py                # Model testing and evaluation script
└── utility.py                   # General utility functions


## dun and install
use this command for the needed package: pip install -r requirements.txt
1. Ensure the run.sh script has executable permissions: chmod +x run.sh
2. Execute the script: ./run.sh



Setup and Installation
To run BR-FEEL experiments, follow these steps:

Install Required Packages
Use the following command to install the necessary Python dependencies:
pip install -r requirements.txt
Ensure Script Permissions
Make the run.sh script executable:
chmod +x run.sh
Run the Experiments
Execute the run.sh script to start the training:./run.sh
This script will automatically:

Activate the Conda environment.
Run the main experiment script (mask_main.py) with the specified configuration.
Deactivate the environment after execution.

Running with Different Configurations
The configurations for different models and datasets are stored in the config/ directory. Modify the run.sh script to specify the desired configuration file. For example:

# To use MobileNetV2
python mask_main.py --config_path=./config/mobilenetv2_config.yaml

# To use ResNet-34
python mask_main.py --config_path=./config/resnet34_config.yaml









If you use BR-FEEL in your research, please cite our paper:

bibtex
Copy
Edit
@article{br_feel,
  title={BR-FEEL: A Backdoor Resilient Approach for Federated Edge Learning with Fragment-Sharing},
  author={Your Name et al.},
  journal={Your Journal},
  year={2023}
}


Acknowledgements
This repository was developed as part of the research project on enhancing the resilience of federated edge learning against backdoor attacks. We thank the contributors and the community for their support.


