# OCR Engine

## Overview
This repository contains a PyTorch-based model designed to classify characters from the TMNIST Glyphs dataset. The model processes grayscale 28x28 images and predicts the corresponding character label. The training pipeline leverages convolutional neural networks (CNNs) to learn effective feature representations.


## Dependencies
Ensure you have the following installed:
- Python 3.11.7

Install dependencies with:
```bash
pip install requirements.txt
```

## Model Architecture
The model is based on a convolutional neural network (CNN) with the following structure:
1. Two convolutional layers
2. Fully connected layers for classification
3. Softmax activation for multi-class classification

## Training Process
- **Data Splitting**: The dataset is divided into training and validation sets.
- **Training Loop**:
  - Forward pass through the model
  - Compute loss using cross-entropy
  - Backpropagation and weight update using Adam optimizer
- **Evaluation**:
  - Computes accuracy and loss on validation data
  - Logs training progress per epoch

## Future Improvements

- Implement data augmentation for better generalization.
- Optimize hyperparameters (learning rate, batch size, etc.).
- Explore deeper CNN architectures for improved accuracy.

## Acknowledgments

This project utilizes TMNIST glyph data, available for download [**here**](https://www.kaggle.com/datasets/nimishmagre/tmnist-glyphs-1812-characters). This dataset contains over 500,000 images and is part of the Warhol.ai Computational Creativity and Congnitive Type projects.
