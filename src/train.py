# src.train
import torch
import torch.nn as nn
from src.model import OCRModel


class OCRTrainer:
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = OCRModel(num_classes=num_classes)
        self.optimizer = torch.optim.Adam
        self.criterion = nn.CrossEntropyLoss

        print(
            f"{self.__class__.__name__} initialized with {num_classes} classes.\n"
        )
        print("*\n" * 3)

    def fit(self, train_loader, val_loader, batch_size, num_epochs, lr):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.cnn.to(device)
        criterion = self.criterion()
        optimizer = self.optimizer(model.parameters(), lr=lr)

        train_eval, val_eval = [], []

        # Print header once before training begins
        print(
            f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^12} | {'Val Acc':^10}"
        )
        print("-" * 60)

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
                    batch_acc = 100 * correct / total
                else:
                    batch_acc = None

            avg_loss = train_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_eval.append([epoch + 1, train_acc, avg_loss])

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
                val_acc = 100 * (correct / total)
                val_eval.append([epoch + 1, val_acc, val_loss])

                # After validation
                self.print_training_status(
                    epoch + 1,  # 1-based epoch indexing
                    num_epochs,
                    avg_loss,
                    train_acc=train_acc,
                    batch_acc=batch_acc,  # Only show batch acc during training if needed
                    val_loss=val_loss,
                    val_acc=val_acc,
                )

        model_performance = {"train": train_eval, "val": val_eval}
        self.model = model  # Save the trained model.
        return model, model_performance

    @staticmethod
    def print_training_status(
        epoch,
        num_epochs,
        avg_loss,
        train_acc,
        batch_acc=None,
        val_loss=None,
        val_acc=None,
    ):
        """Print training status in a clean, aligned format."""
        # Convert accuracy from 0-100 to 0-1 range if needed
        train_acc_fmt = train_acc / 100 if train_acc > 1 else train_acc
        val_acc_fmt = val_acc / 100 if val_acc and val_acc > 1 else val_acc

        # Create the formatted status line
        status = (
            f"{epoch:2d}/{num_epochs:<2d} | "
            f"{avg_loss:<12.4f} | "
            f"{train_acc_fmt:9.2%} | "
            f"{val_loss:<12.4f} | "
            f"{val_acc_fmt:9.2%}"
        )

        # Add batch accuracy if available
        if batch_acc is not None:
            batch_acc_fmt = batch_acc / 100 if batch_acc > 1 else batch_acc
            status += f" | Batch: {batch_acc_fmt:.2%}"

        print(status)
