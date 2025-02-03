import torch
from model_architecture import CNN2L  # Import the model class from model.py


def load_model(weights_path, num_classes=None):
    """
    Load the model architecture and its weights.

    Args:
        path (str): Path to the saved model weights (.pt or .pth file).

    Returns:
        model (torch.nn.Module): The loaded model with weights, set to evaluation mode.
    """

    # Initialize the model
    model = CNN2L(num_classes=num_classes)

    model.load_state_dict(torch.load(weights_path))

    # Set the model to evaluation mode
    model.eval()

    return model
