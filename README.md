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
├── config/                # YAML configuration files for experiments
├── data/                  # Directory for storing datasets
├── models/                # Pretrained models and model architectures
├── src/                   # Source code for BR-FEEL
│   ├── main.py            # Main training script
│   ├── fragment_sharing.py # Implementation of fragment-sharing
│   ├── knowledge_distillation.py # Knowledge distillation process
│   └── utils/             # Utility functions
├── results/               # Directory to save experimental results
└── README.md              # This file
