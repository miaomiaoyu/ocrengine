import os
import json
import torch
from datetime import datetime
from src.train_model import train_model
from src.data.preprocess import load_and_preprocess_data
import warnings

warnings.filterwarnings("ignore")

training_labels = [
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

# Training dataset
data_file = "data/source/TMNIST_Glyphs.csv.gz"
params_file = ".config/params.json"
output_dir = "output"


def cls():
    os.system("cls" if os.name == "nt" else "clear")


def main(params):
    # Clear terminal screen
    cls()

    # Create a name for the model
    model_id = f"{datetime.now().strftime('%y%m%d%H%M')}"
    output_path = os.path.join(output_dir, model_id)
    os.makedirs(output_path, exist_ok=True)
    print(f"\033[4mModel ID: {model_id}\033[0m\n")

    # Model classes
    model_classes = dict(
        zip(training_labels, [int(i) for i in range(len(training_labels))])
    )

    # Load data
    X, y = load_and_preprocess_data(data_file, model_classes)

    # Train CNN model
    model, model_performance = train_model(X, y, **params)

    # Assume 'model' is your trained PyTorch model
    torch.save(model, os.path.join(output_path, "model.pt"))

    # Save the model weights
    torch.save(model.state_dict(), os.path.join(output_path, "weights.pt"))

    # Save the classes the model was trained on with character as keys and the corresponding class output by the model as value.
    torch.save(
        {
            "model_classes": model_classes,
            "model_performance": model_performance,
        },
        os.path.join(output_path, "model_metrics.pt"),
    )


if __name__ == "__main__":
    with open(params_file, "r") as f:
        params = json.load(f)
    main(params)
