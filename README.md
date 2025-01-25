BR-FEEL: A Backdoor-Resilient Approach for Federated Edge Learning with Fragment-Sharing

---

Abstract

In Federated Edge Learning (FEEL) systems, where resources are often constrained, fragment-sharing enables clients to collaboratively train large models with billions of parameters. Unlike traditional federated learning, where the entire local model is trained and shared, fragment-sharing allows clients to train and share only specific parameter fragments based on their storage, computational, and networking capabilities.

Challenges:
1. Backdoor attacks hidden in fragments are harder to detect since the full model is not shared.
2. The security of the overall FEEL system is significantly compromised.

Proposed Solution: BR-FEEL
1. Backdoor-Resilient Twin Model:
   Each benign client integrates fragments from others into a twin model to identify and handle malicious parameters.
2. Knowledge Distillation:
   Clean knowledge from the twin model is transferred back to the local model through a carefully designed distillation process.

Results:
Our experiments on the CIFAR-10 and GTSRB datasets, using MobileNetV2 and ResNet-34, demonstrate that BR-FEEL reduces attack success rates by over 90% compared to other baselines under various attack methods.

---

Repository Structure

Folders:
- config/
  - YAML configuration files for different experiments
  - mobilenetv2_config.yaml - Configuration for MobileNetV2-based experiments
  - resnet34_config.yaml - Configuration for ResNet34-based experiments
- data/
  - Dataset storage
- defense/
  - Defense strategies implementations
- mask_backdoor_res/
  - Results for experiments with masking backdoors
- models/
  - Pretrained models and model architectures
- poison_tool_box/
  - Tools for generating poisoned datasets
- poisoned_set/
  - Directory for poisoned datasets
- rebuttal/
  - Results for rebuttal experiments
- triggers/
  - Trigger images for backdoor attacks
- VisualTools/
  - Visualization tools for experiment results
- wandb/
  - Logs generated using Weights and Biases

Scripts:
- base_config.py - Base configurations for the experiments
- create_data.py - Script for creating datasets
- CustomDataset.py - Custom dataset loader
- download_data.py - Script for downloading required datasets
- inject_backdoor.py - Code for injecting backdoor attacks
- mask_main.py - Main script for running backdoor masking experiments
- mask_utils.py - Utility functions for masking-based defenses
- model.py - Model definitions
- motivation.py - Script for running motivation experiments
- plugin.py - Additional plugin utilities
- requirement.txt - Dependencies and required Python packages
- run.sh - Shell script for running experiments
- test_model.py - Model testing and evaluation script
- utility.py - General utility functions

---

Setup and Installation

1. Install Required Packages
   Run the following command to install the necessary Python dependencies:
   pip install -r requirements.txt

2. Ensure Script Permissions
   Make the run.sh script executable:
   chmod +x run.sh

3. Run the Experiments
   Execute the run.sh script to start the training:
   ./run.sh
   This script will automatically:
   - Activate the Conda environment.
   - Run the main experiment script (mask_main.py) with the specified configuration.
   - Deactivate the environment after execution.

4. Run with Different Configurations
   Modify the run.sh script to specify the desired configuration file. Examples:
   - To use MobileNetV2:
     python mask_main.py --config_path=./config/mobilenetv2_config.yaml
   - To use ResNet-34:
     python mask_main.py --config_path=./config/resnet34_config.yaml

---

Citation

If you use BR-FEEL in your research, please cite our paper:

@article{br_feel,
  title={BR-FEEL: A Backdoor Resilient Approach for Federated Edge Learning with Fragment-Sharing},
  author={Your Name et al.},
  journal={Your Journal},
  year={2023}
}

---

Acknowledgements

This repository was developed as part of the research project on enhancing the resilience of federated edge learning against backdoor attacks. We thank the contributors and the community for their support.
