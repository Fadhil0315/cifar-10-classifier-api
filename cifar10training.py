
pip3 install torch torchvision torchaudio


# %%
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Transform: Convert to tensor + normalize
transform = transforms.Compose([
    transforms.ToTensor(),   # [0,255] → [0,1]
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])




# Train dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

# Test dataset
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

from torch.utils.data import random_split

train_size = int(0.9 * len(trainset))
val_size = len(trainset) - train_size

train_dataset, val_dataset = random_split(
    trainset, [train_size, val_size]
)


# DataLoaders
trainloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    
    pin_memory=False
)

valloader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    
    pin_memory=False
)


testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

classes = (
    'plane','car','bird','cat','deer',
    'dog','frog','horse','ship','truck'
)


# %%
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)


    def forward(self, x):

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))


        x = x.view(x.size(0), -1)   # flatten

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# %%
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

model.load_state_dict(torch.load("best_model.pth",map_location=device))

print("Loaded base model")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,      # total epochs
    eta_min=1e-5   # minimum LR
)



# %%
def validate(model, loader, criterion):

    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total

    return val_loss / len(loader), acc


# %%
best_val_loss = float("inf")
patience = 7
counter = 0


# %%


model.to(device)

epochs = 40   # max (won't reach usually)

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(trainloader)

    # Validation
    val_loss, val_acc = validate(model, valloader, criterion)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.2f}%")

    # ----- EARLY STOPPING LOGIC -----

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), "best_model.pth")

        print("Saved best model ✅")

    else:

        counter += 1
        print(f"No improvement ({counter}/{patience})")

        if counter >= patience:

            print("Early stopping triggered ⛔")
            break

scheduler.step()




# %%
correct = 0
total = 0

model.eval()

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", 100*correct/total, "%")


# %%
model.load_state_dict(torch.load("best_model.pth"))

print("Best model restored ✅")




