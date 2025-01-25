# BR-FEEL: A Backdoor-Resilient Approach for Federated Edge Learning with Fragment-Sharing

---

## Abstract

In **Federated Edge Learning (FEEL)** systems, where resources are often constrained, **fragment-sharing** enables clients to collaboratively train large models with billions of parameters. Unlike traditional federated learning, where the entire local model is trained and shared, fragment-sharing allows clients to train and share only specific parameter fragments based on their **storage, computational, and networking capabilities**.

### Challenges:
1. **Backdoor attacks** hidden in fragments are harder to detect since the full model is not shared.  
2. The security of the overall FEEL system is significantly compromised.

### Proposed Solution: BR-FEEL
1. **Backdoor-Resilient Twin Model:**  
   Each benign client integrates fragments from others into a twin model to identify and handle malicious parameters.  
2. **Knowledge Distillation:**  
   Clean knowledge from the twin model is transferred back to the local model through a carefully designed distillation process.

### Results:
Our experiments on the **CIFAR-10** and **GTSRB** datasets, using **MobileNetV2** and **ResNet-34**, demonstrate that **BR-FEEL** reduces attack success rates by **over 90%** compared to other baselines under various attack methods.

---

## Repository Structure

### Folders:
- **config/**  
  - YAML configuration files for different experiments  
  - `mobilenetv2_config.yaml` - Configuration for MobileNetV2-based experiments  
  - `resnet34_config.yaml` - Configuration for ResNet34-based experiments  
- **data/**  
  - Dataset storage  
- **defense/**  
  - Defense strategies implementations  
- **mask_backdoor_res/**  
  - Results for experiments with masking backdoors  
- **models/**  
  - Pretrained models and model architectures  
- **poison_tool_box/**  
  - Tools for generating poisoned datasets  
- **poisoned_set/**  
  - Directory for poisoned datasets  
- **rebuttal/**  
  - Results for rebuttal experiments  
- **triggers/**  
  - Trigger images for backdoor attacks  
- **VisualTools/**  
  - Visualization tools for experiment results  
- **wandb/**  
  - Logs generated using Weights and Biases  

### Scripts:
- `base_config.py` - Base configurations for the experiments  
- `create_data.py` - Script for creating datasets  
- `CustomDataset.py` - Custom dataset loader  
- `download_data.py` - Script for downloading required datasets  
- `inject_backdoor.py` - Code for injecting backdoor attacks  
- `mask_main.py` - Main script for running backdoor masking experiments  
- `mask_utils.py` - Utility functions for masking-based defenses  
- `model.py` - Model definitions  
- `motivation.py` - Script for running motivation experiments  
- `plugin.py` - Additional plugin utilities  
- `requirement.txt` - Dependencies and required Python packages  
- `run.sh` - Shell script for running experiments  
- `test_model.py` - Model testing and evaluation script  
- `utility.py` - General utility functions  

---

## Setup and Installation

1. **Install Required Packages**  
   Run the following command to install the necessary Python dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
2. **Ensure Script Permissions**
   Make the run.sh script executable:
   ```bash
   chmod +x run.sh
   ```
3. **Generate the Dataset**  
   Use the following command to generate the required datasets:  
   ```bash
   python download_data.py
   ```
4. **Run the Experiments**  

   - **Using `run.sh` Script**  
     To start the training process, execute the `run.sh` script:  
     ```bash
     ./run.sh
     ```  
     For a simple run, this script can be used directly. To customize parameters, modify the script content to adjust the inner shell configurations as needed.

   - **Using Python Directly**  
     Alternatively, you can directly run the experiments using Python with the desired configuration file:  
     - To use **MobileNetV2**:  
       ```bash
       python mask_main.py --config_path=./config/mobilenetv2_config.yaml
       ```  
     - To use **ResNet-34**:  
       ```bash
       python mask_main.py --config_path=./config/resnet34_config.yaml
       ```
       
5. **Real-Time Result Monitoring with Weights and Biases (WandB)**
   This project leverages WandB for real-time experiment tracking and visualization
   ```bash
      wandb login
   ```
## Citation
   If you use BR-FEEL in your research, please cite our paper:
      
      ```bibtex
      @article{QI2024103258,
      title = {BR-FEEL: A backdoor resilient approach for federated edge learning with fragment-sharing},
      journal = {Journal of Systems Architecture},
      volume = {155},
      pages = {103258},
      year = {2024},
      issn = {1383-7621},
      doi = {https://doi.org/10.1016/j.sysarc.2024.103258},
      url = {https://www.sciencedirect.com/science/article/pii/S1383762124001954},
      author = {Senmao Qi and Hao Ma and Yifei Zou and Yuan Yuan and Peng Li and Dongxiao Yu},
      keywords = {Federated edge learning, Fragment-sharing, Backdoor defense, Knowledge distillation},
      }
      ```

## Acknowledgements
      ```css
      This repository was developed as part of a research project focused on enhancing the resilience of federated edge learning systems against backdoor attacks. We extend our gratitude to all contributors and the community for their invaluable support and feedback.
      ```
