import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

# Load MNIST dataset (for constructing our custom dataset)
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((7, 7))  # Resize each digit to 7x7
])
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)


# Custom dataset to form 224x224 images with 7x7 MNIST tiles
class MNISTPatchDataset(Dataset):
    def __init__(self, mnist_data, grid_size=7, image_size=224):
        self.mnist_data = mnist_data
        self.grid_size = grid_size  # 7x7 grid
        self.image_size = image_size
        self.tile_size = image_size // grid_size  # 224/7 = 32
        self.num_tiles = grid_size * grid_size  # 49 patches per image

    def __len__(self):
        return len(self.mnist_data) // self.num_tiles  # Enough to form full images

    def __getitem__(self, idx):
        indices = np.random.choice(len(self.mnist_data), self.num_tiles, replace=True)
        tiles = []
        labels = []

        for i in indices:
            digit_img, digit_label = self.mnist_data[i]
            resized_tile = transforms.Resize((self.tile_size, self.tile_size))(digit_img)  # 7x7 → 32x32
            tiles.append(resized_tile)
            labels.append(digit_label)

        # Create 7x7 grid of 32x32 MNIST tiles
        image = torch.cat(
            [torch.cat(tiles[i:i + self.grid_size], dim=2) for i in range(0, self.num_tiles, self.grid_size)], dim=1)
        label = torch.tensor(labels, dtype=torch.long)  # Shape: [49]

        return image.repeat(3, 1, 1), label  # Convert grayscale to 3-channel RGB


# Create datasets and data loaders
batch_size = 64
train_dataset = MNISTPatchDataset(mnist_train)
test_dataset = MNISTPatchDataset(mnist_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained ConvNeXT-Tiny model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

# Modify classification layer to predict 49 digits (one for each tile)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 49 * 10)

# Wrap the model for correct output shape
class TileClassifier(nn.Module):
    def __init__(self, model):
        super(TileClassifier, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=-1)  # Softmax over 10 classes per tile

    def forward(self, x):
        x = self.model(x)  # Output shape: [batch, 490]
        x = x.view(x.shape[0], 49, 10)  # Reshape to [batch, 49, 10]
        return x  # Return raw logits (CrossEntropyLoss will apply softmax)

model = TileClassifier(model).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training function
def train(model, train_loader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)  # labels: [batch, 49]

            optimizer.zero_grad()
            outputs = model(images)  # Shape: [batch_size, 49, 10]

            loss = criterion(outputs.view(-1, 10), labels.view(-1))  # Reshape for loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Compute accuracy
            preds = outputs.argmax(dim=2)  # Get predicted labels (batch, 49)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            loop.set_postfix(loss=epoch_loss / (total / 49), acc=100 * correct / total)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {100 * correct / total:.2f}%")


# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Shape: [batch, 49, 10]
            loss = criterion(outputs.view(-1, 10), labels.view(-1))  # Compute loss
            total_loss += loss.item()

            preds = outputs.argmax(dim=2)  # Predicted class per tile
            correct += (preds == labels).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    print(f"Test Loss: {total_loss:.4f} - Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Train and evaluate
evaluate(model, test_loader, criterion, device)
train(model, train_loader, optimizer, criterion, device, epochs=5)
evaluate(model, test_loader, criterion, device)
