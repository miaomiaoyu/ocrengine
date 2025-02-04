# GlyphNet Model

## Overview
This repository contains a PyTorch-based model designed to classify characters from the TMNIST dataset. The model processes grayscale 28x28 images and predicts the corresponding character label. The training pipeline leverages convolutional neural networks (CNNs) to learn effective feature representations.

## Project Structure
```
.
├── src
│   ├── models
│   │   ├── train_model.py  # Training pipeline
│   ├── data
│   │   ├── dataset.py  # Data loading utilities
│   ├── model_architecture.py  # CNN Model definition
│   ├── utils.py  # Helper functions (if applicable)
├── README.md  # Project documentation
```

## Dependencies
Ensure you have the following installed:
- Python 3.x
- PyTorch
- NumPy
- Scikit-learn

Install dependencies with:
```bash
pip install torch numpy scikit-learn
```

## Training the Model
To train the TMNIST character recognition model, call the `train_model` function in `src/models/train_model.py`:
```python
from src.models.train_model import train_model
import numpy as np

# Example data
X = np.random.rand(1000, 28, 28)  # Replace with actual TMNIST images
y = np.random.randint(0, 10, 1000)  # Replace with actual labels

# Training parameters
TRAIN_SIZE = 0.8
BATCH_SIZE = 32
SHUFFLE = True
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Train the model
model, performance = train_model(X, y, TRAIN_SIZE, BATCH_SIZE, SHUFFLE, NUM_EPOCHS, LEARNING_RATE)
```

## Model Architecture
The model is based on a convolutional neural network (CNN) with the following structure:
1. Two convolutional layers (`CNN2L`)
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

## Performance Tracking
The training function returns a dictionary containing training and validation loss/accuracy per epoch:
```python
{
    "train": [[epoch, train_accuracy, train_loss], ...],
    "val": [[epoch, val_accuracy, val_loss], ...]
}
```

## Model Saving & Deployment
To save the trained model:
```python
torch.save(model.state_dict(), "tmnist_model.pth")
```
To load for inference:
```python
from src.model_architecture import CNN2L
import torch

model = CNN2L(num_classes=10)
model.load_state_dict(torch.load("tmnist_model.pth"))
model.eval()
```

## Future Improvements

- Implement data augmentation for better generalization.
- Optimize hyperparameters (learning rate, batch size, etc.).
- Explore deeper CNN architectures for improved accuracy.

## Acknowledgments

This project utilizes TMNIST glyph data for character recognition research and experimentation.
