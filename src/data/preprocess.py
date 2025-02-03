# src.data.preprocess
import pandas as pd
import numpy as np


def load_and_preprocess_data(data_file: str, model_classes: dict):
    # Load in TMNIST Glyphs dataset
    data = pd.read_csv(data_file, compression="gzip", index_col=0)

    # Filter rows by the MODEL_CLASSES specified
    training_data = data[data["label"].isin(model_classes.keys())].copy()

    # Filter columns and leave label and image columns required for training.
    training_data = training_data.iloc[:, 2:].reset_index(drop=True)

    # Assign data (X) and label (y)
    X = training_data.iloc[:, 1:].values
    y_labels = training_data.iloc[:, 0]

    y = np.array([model_classes[y_label] for y_label in y_labels])  # Encoded labels.

    return X, y
