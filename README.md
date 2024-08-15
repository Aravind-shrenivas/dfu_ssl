# Multi-Scale Attention Network for Diabetic Foot Ulcer Segmentation using Self-Supervised Learning

## Overview

This project focuses on detecting diabetic foot ulcers using a self-supervised learning approach combined with attention mechanisms. The model utilizes a pretrained Vision Transformer (ViT) from DinoV2 and multi-scale DenseNet for feature extraction, followed by an Attention UNet for segmentation.

## Scripts

- **`main.py`**: Trains the self-supervised model using unlabelled images to extract meaningful features.
- **`fine_tune.py`**: Fine-tunes the pre-trained model for segmentation tasks using labeled data.
- **`inference.py`**: Runs inference on new images to generate ulcer segmentation masks.

## Results

- **Self-Supervised Learning**: MSE Loss = **0.01**
- **Segmentation (Attention UNet)**:
  - BCEWithLogits Loss = **0.14**
  - Dice Coefficient Loss = **0.3054**

## Segmentation Results

![200025_output](https://github.com/user-attachments/assets/49a60625-f26b-4a6b-898d-62baf2233ece)

![200019_output](https://github.com/user-attachments/assets/02f22cc5-cf19-4da0-bd1a-66d29ffabc8b)


## Instructions

1. **Replace Paths**: Update the paths in the scripts with your own paths.
2. **Run the Scripts**: Execute the scripts (`main.py`, `fine_tune.py`, `inference.py`) in sequence for training, fine-tuning, and inference.

## Authors

- **Aravind shrenivas Murali** - Graduate Student, University of Arizona, aravindshrenivas@arizona.edu
- **Marino Alejandro Chuquilin Fernandez** - Undergraduate Student, University of Arizona, marinocf@arizona.edu
- **Dr. Eung-Joo Lee** - Assistant Professor, University of Arizona, eungjoolee@arizona.edu

## License

This project is licensed under the MIT License.
