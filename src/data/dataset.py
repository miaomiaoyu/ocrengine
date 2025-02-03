import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):
    """
    ImageDataset is a specialized PyTorch Dataset for processing and transforming TMNIST glyph images.

    This class loads TMNIST image data, applies a sequence of transformations for data augmentation, and structures the image set for input into a machine learning model.

    Attributes:
        image_shape (Tuple[int, int]): The shape of each processed image as (height, width).
        transforms (torchvision.transforms.Compose): Composes several torchvision transformations that are applied to the images in sequence.
        images (List[torch.Tensor]): The list of transformed images.
        labels (List[int]): A corresponding list of labels for each of the images.

    Args:
        X (numpy.ndarray): The original image data as a numpy array.
        y (numpy.ndarray): The corresponding labels as a numpy array.
        image_size (Tuple[int, int], optional): The desired image shape as (height, width) for each transformed image. Defaults to (28, 28).
    """

    def __init__(self, X, y, image_size=(28, 28)):
        super(ImageDataset, self).__init__()

        self.image_size = image_size
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Pad(padding=5),
                T.Resize(self.image_size),
                T.Normalize((0.5,), (0.5,)),
            ]
        )

        images, labels = [], []

        for i in range(len(X)):
            image = X[i]
            image = np.asarray(image).reshape(*self.image_size, 1).astype("float32")
            image = self.transform(image)
            label = torch.tensor(y[i])
            images.append(image)
            labels.append(label)

        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
