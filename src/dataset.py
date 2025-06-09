import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    """
    A memory-efficient PyTorch Dataset for processing and transforming TMNIST char images.

    This class supports on-demand processing, configurable transformations, and
    separate augmentation pipelines for training and evaluation.

    Attributes:
        image_size (Tuple[int, int]): The shape of each processed image as (height, width).
        train (bool): Whether this dataset is for training (enables augmentation).
        X (numpy.ndarray): The original image data.
        y (numpy.ndarray): The corresponding labels.
        transform (torchvision.transforms.Compose): Base transformations applied to all images.
        train_transform (torchvision.transforms.Compose): Additional augmentations for training.
        cache_processed (bool): Whether to cache processed images in memory.
        cached_images (dict): Dictionary storing processed images if caching is enabled.

    Args:
        X (numpy.ndarray): The original image data as a numpy array.
        y (numpy.ndarray): The corresponding labels as a numpy array.
        image_size (Tuple[int, int], optional): The desired image shape. Defaults to (28, 28).
        train (bool, optional): Whether this dataset is for training. Defaults to True.
        transform (Optional[callable], optional): Custom transform pipeline. Defaults to None.
        use_augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
        cache_processed (bool, optional): Whether to cache processed images. Defaults to False.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        image_size: Tuple[int, int] = (28, 28),
        train: bool = True,
        transform: Optional[callable] = None,
        use_augmentation: bool = False,
        cache_processed: bool = False,
    ):
        super(OCRDataset, self).__init__()

        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same length. Got {X.shape[0]} vs {y.shape[0]}"
            )

        self.image_size = image_size
        self.train = train
        self.X = X
        self.y = y
        self.cache_processed = cache_processed
        self.cached_images = {} if cache_processed else None

        # Use provided transform or create default
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Pad(padding=5),
                T.Resize(self.image_size),
                T.Normalize((0.5,), (0.5,)),
            ])

        # Add augmentation for training if requested
        self.use_augmentation = use_augmentation
        if use_augmentation and train:
            self.train_transform = T.Compose([
                T.RandomRotation(10),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                # Add more augmentations as needed
            ])
        else:
            self.train_transform = None

        logger.info(
            f"Created {self.__class__.__name__} with {len(self)} samples"
            f" (train={train}, augmentation={use_augmentation})"
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single image-label pair by index."""
        # Check if this image is already cached
        if self.cache_processed and index in self.cached_images:
            return self.cached_images[index]

        # Process the image
        image = self.X[index]

        # Convert to numpy array if needed (handles different input types)
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        # Reshape to proper format for transformation
        try:
            image = image.reshape(*self.image_size, 1).astype("float32")
        except ValueError as e:
            raise ValueError(
                f"Failed to reshape image at index {index} to {self.image_size}: {e}"
            )

        # Apply base transforms
        image = self.transform(image)

        # Apply training augmentations if applicable
        if self.train and self.train_transform:
            image = self.train_transform(image)

        # Get corresponding label
        label = torch.tensor(self.y[index])

        # Cache if requested
        if self.cache_processed:
            self.cached_images[index] = (image, label)

        return image, label

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.X)

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Returns:
            torch.Tensor: Weights for each class, inversely proportional to frequency.
        """
        class_counts = np.bincount(self.y)
        n_samples = len(self.y)
        n_classes = len(class_counts)

        # Calculate weights (inversely proportional to class frequency)
        weights = n_samples / (n_classes * class_counts)
        return torch.FloatTensor(weights)
