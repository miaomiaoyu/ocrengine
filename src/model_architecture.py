# src.model_architecture
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


class CNN2L(nn.Module):
    def __init__(self, num_classes):
        super(CNN2L, self).__init__()
        """Simple Convolutional Neural Network with 2 layers."""

        # 1 input channel (grayscale), 32 output channels.
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 32 input channel (grayscale), 64 output channels.
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Drop out layer, helps prevent overfitting during training by
        # randomly setting some of the input units to 0 during the forward pass.
        self.drop_out = nn.Dropout()

        # Fully-connected layer 1: takes in 7 * 7 * 64 input features and produces
        # 1000 output features.
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)

        # Fully-connected layer 2: takes in 1000 input features and produces
        # num_classes output features.
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
