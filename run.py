#!/usr/bin/python3
import os
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.train import OCRTrainer
from src.dataset import OCRDataset
from src.io import (
    load_and_preprocess_tmnist_data,
    export_ocr_model,
)
import warnings

warnings.filterwarnings("ignore")


def get_configs():
    return {
        "batch_size": 128,
        "num_epochs": 11,
        "lr": 0.001,
        "train_size": 0.7,
    }


def cls():
    os.system("cls" if os.name == "nt" else "clear")


def train_model(X, y, train_size, batch_size, num_epochs, lr):
    # Split data into train/test sets
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, train_size=train_size
    )

    # Package data into torch's DataLoaders
    train_loader = DataLoader(
        OCRDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        OCRDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )

    # Establish the number of unique classes for initializing ConvNet in Trainer
    num_classes = len(np.unique(y))

    # Init ConvNet
    trainer = OCRTrainer(num_classes)

    # Train ConvNet on loaders and model params
    model, model_performance = trainer.fit(
        train_loader,
        val_loader,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
    )

    return model, model_performance


def main():
    # Create a name for the model
    model_id = f"{datetime.now().strftime('%y-%m-%d-%H')}"
    model_dir = os.path.join("models", "testing", model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Also create a symlink to the latest model
    latest_dir = os.path.join("models", "testing", "latest")
    if os.path.exists(latest_dir):
        if os.path.islink(latest_dir):
            os.unlink(latest_dir)
        else:
            import shutil

            shutil.rmtree(latest_dir)

    # Model training and validation
    configs = get_configs()
    target_classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "-",
        ".",
        ",",
        "<",
    ]

    num_classes = len(target_classes)

    # Model classes
    model_classes = dict(zip(target_classes, range(num_classes)))

    # Clear terminal screen
    cls()

    # Load data
    X, y = load_and_preprocess_tmnist_data(model_classes=model_classes)

    # Train CNN model
    model, model_performance = train_model(X, y, **configs)

    # Export the model with metadata
    export_ocr_model(model, model_classes, model_performance, model_dir)
    print("\n   >> Training complete. Model and metadata saved.")

    # Create symbolic link to the latest model
    os.symlink(model_dir, latest_dir)
    print(f"    >> Created symbolic link to latest model {latest_dir}")


if __name__ == "__main__":
    main()
