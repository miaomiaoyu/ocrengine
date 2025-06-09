import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Define default locations to search for data file
DEFAULT_DATA_FILENAME = "TMNISTGlyphs.csv.gz"


def find_data_file(filename=DEFAULT_DATA_FILENAME):
    """Search for the data file in common locations"""
    for file_loc in [
        "data/",  # Relative to current working directory
        "../data/",  # Up one level
        os.path.dirname(__file__),  # Same directory as this module
        os.path.join(
            os.path.dirname(__file__), "data"
        ),  # data/ subdirectory of module
    ]:
        data_file = os.path.join(file_loc, filename)

        if os.path.exists(data_file):
            return data_file

        raise FileNotFoundError(f"Could not find {filename}.")


def load_and_preprocess_tmnist_data(data_file=None, model_classes=None):
    """
    Load and preprocess TMNIST data.

    Parameters:
    -----------
    data_file : str, optional
        Path to the data file. If None, will search in standard locations.
    model_classes : dict, optional
        Dictionary mapping characters to class indices.
        If None, will use all unique classes in the dataset.
    """
    # Find the data file if not provided
    if data_file is None:
        data_file = find_data_file()

    # Load the data
    print(f"Loading data from: {data_file}")

    # Load in TMNIST Glyphs dataset
    data = pd.read_csv(data_file, compression="gzip", index_col=0)

    # Filter rows by the keys in `model_classes`.
    training_data = data[data["label"].isin(model_classes.keys())].copy()

    # Filter columns and leave label and image columns required for training.
    training_data = training_data.iloc[:, 2:].reset_index(drop=True)

    # Assign data (X) and label (y)
    X = training_data.iloc[:, 1:].values
    y_labels = training_data.iloc[:, 0]

    y = np.array([
        model_classes[y_label] for y_label in y_labels
    ])  # Encoded labels.

    return X, y


def export_ocr_model(model, model_classes, model_performance, export_dir):
    """Export model with organized structure and metadata"""
    # Create directory structure
    weights_dir = os.path.join(export_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Save the complete model
    torch.save(model, os.path.join(weights_dir, "model.pt"))

    # Save just the weights for more flexible loading
    torch.save(model.state_dict(), os.path.join(weights_dir, "weights.pt"))

    # Create label decoder dictionary (inverse of model_classes)
    label_decoder = {str(idx): char for char, idx in model_classes.items()}

    # Create metadata with performance metrics and label decoder
    latest_val_metrics = (
        model_performance["val"][-1] if model_performance["val"] else None
    )
    best_val_acc = (
        max([x[1] for x in model_performance["val"]])
        if model_performance["val"]
        else 0
    )

    metadata = {
        "name": "OCR-Engine",
        "version": datetime.now().strftime("%y.%m.%d"),
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_size": [28, 28],
        "num_classes": len(model_classes),
        "classes": list(model_classes.keys()),
        "label_decoder": label_decoder,
        "performance": {
            "best_accuracy": float(best_val_acc),
            "final_epoch": {
                "train_acc": float(model_performance["train"][-1][1])
                if model_performance["train"]
                else None,
                "val_acc": float(latest_val_metrics[1])
                if latest_val_metrics
                else None,
                "train_loss": float(model_performance["train"][-1][2])
                if model_performance["train"]
                else None,
                "val_loss": float(latest_val_metrics[2])
                if latest_val_metrics
                else None,
            },
        },
    }

    # Save metadata as JSON for better interoperability
    with open(os.path.join(export_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save raw performance metrics in PyTorch format for further analysis if needed
    torch.save(
        model_performance, os.path.join(export_dir, "performance_metrics.pt")
    )

    print(f"Model exported to {export_dir}")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
