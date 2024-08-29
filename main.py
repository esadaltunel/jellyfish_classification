import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.functional import relu

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = self.pool(relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor for the fully connected layer
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_mean_std():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    dataset = datasets.ImageFolder(root='archive', transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in data_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

mean, std = get_mean_std()


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),       # Random horizontal flip with 50% probability
    transforms.RandomRotation(degrees=30),        # Random rotation within a range of 30 degrees
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop and resize, focusing on different parts of the image
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation within 10% of the image size
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # More aggressive color jittering
    transforms.RandomGrayscale(p=0.1),            # Randomly convert images to grayscale with a 10% chance
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
    transforms.RandomRotation(degrees=15),   # Random rotation within a range of 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random changes in brightness, contrast, saturation, and hue
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = datasets.ImageFolder(root='archive', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(root='test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


n_class = len(train_dataset.classes)
model = SimpleCNN(num_classes=n_class)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_loss = []
train_accuracy = []

test_loss = []
test_accuracy = []

print("Training/Testing started...")
EPOCHS = 30
for epoch in range(EPOCHS):
    train_running_loss = 0.0
    train_correct_predictions = 0
    train_total_predictions = 0
    
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        train_correct_predictions += (predicted == labels).sum().item()
        train_total_predictions += labels.size(0)
    
    accuracy = train_correct_predictions / train_total_predictions * 100
    train_accuracy.append(accuracy)
    train_loss.append(train_running_loss)

    model.eval()
    with torch.no_grad():
        test_running_loss = 0.0
        test_correct_predictions = 0
        test_total_predictions = 0

        for inputs, labels in test_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            test_correct_predictions += (predicted == labels).sum().item()
            test_total_predictions += labels.size(0)
            
        test_loss.append(test_running_loss)
        accuracy = test_correct_predictions / test_total_predictions * 100
        test_accuracy.append(accuracy)
        if epoch % (EPOCHS/10) == 0:
            print(f"Epoch: {epoch} Train Loss: {train_running_loss} Train Accuracy: {accuracy:.2f}%") 
            print(f"Epoch: {epoch} Test Loss: {test_running_loss} Test Accuracy: {accuracy:.2f}%\n")

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(range(EPOCHS), train_loss, label='Train Loss')
axs[0].plot(range(EPOCHS), test_loss, label='Test Loss')
axs[0].set_title('Loss')
axs[0].set_xlabel('EPOCHS')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(range(EPOCHS), train_accuracy, label='Train Accuracy')
axs[1].plot(range(EPOCHS), test_accuracy, label='Test Accuracy')
axs[1].set_title('Accuracy')
axs[1].set_xlabel('EPOCHS')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

torch.save(model, f"{datetime.date.today()}_model.pth")


