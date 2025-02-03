# src.models.train_model
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data.dataset import ImageDataset
from src.model_architecture import CNN2L


def train_model(X, y, TRAIN_SIZE, BATCH_SIZE, SHUFFLE, NUM_EPOCHS, LEARNING_RATE):
    # Split data into train/test sets
    x_train, x_val, y_train, y_val = train_test_split(X, y, train_size=TRAIN_SIZE)

    # Package data into torch's DataLoaders
    train_loader = DataLoader(
        ImageDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )

    val_loader = DataLoader(
        ImageDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )

    # Establish the number of unique classes for initializing ConvNet in Trainer
    num_classes = len(np.unique(y))

    # Init ConvNet
    trainer = ModelTrainer(num_classes)

    # Train ConvNet on loaders and model params
    model, model_performance = trainer.fit(
        train_loader,
        val_loader,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )

    return model, model_performance


class ModelTrainer:
    def __init__(self, num_classes):
        super().__init__()
        self.model = CNN2L(num_classes=num_classes)
        print(f"CNN2L initialized in ModelTrainer with {num_classes} classes.\n")
        print("*\n" * 3)

    def fit(self, train_loader, val_loader, batch_size, num_epochs, learning_rate):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_eval, val_eval = [], []

        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            train_loss, correct, total = 0, 0, 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % batch_size == 0:
                    batch_accuracy = 100 * correct / total
                else:
                    batch_accuracy = None

            average_loss = train_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            train_eval.append([epoch + 1, train_accuracy, average_loss])

            # Validation
            model.eval()  # Set the model to validation mode
            val_loss, correct, total = 0, 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = 100 * correct / total
                val_eval.append([epoch + 1, val_accuracy, val_loss])

                self.print_training_status(
                    epoch,
                    num_epochs,
                    train_loss,
                    train_accuracy,
                    batch_accuracy,
                    val_loss,
                    val_accuracy,
                )

        model_performance = {"train": train_eval, "val": val_eval}

        self.model = model  # Save the trained model.

        return model, model_performance

    @staticmethod
    def print_training_status(
        epoch,
        num_epochs,
        loss,
        accuracy,
        batch_accuracy=None,
        val_loss=None,
        val_accuracy=None,
    ):
        message = (
            f"Epoch: {epoch + 1} / {num_epochs} | "
            f"Training Loss: {loss:.4f} | "
            f"Training Accuracy: {accuracy:.2f}%"
        )

        if val_loss is not None and val_accuracy is not None:
            message += (
                f" | Validation Loss: {val_loss:.4f} | "
                f"Validation Accuracy: {val_accuracy:.2f}%"
            )

        if batch_accuracy is not None:
            message += f" *** Batch Accuracy: {batch_accuracy:.4f}"

        print(message)
