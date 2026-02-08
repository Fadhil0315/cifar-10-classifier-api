import torch
import torch.nn as nn
import torchvision.transforms as transforms

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io


# -----------------------------
# App
# -----------------------------
app = FastAPI()


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# CNN (MUST MATCH TRAINING)
# -----------------------------
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# -----------------------------
# Load Model ONCE
# -----------------------------
model = CNN().to(device)

model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()


# -----------------------------
# CIFAR Classes
# -----------------------------
classes = (
    'plane','car','bird','cat','deer',
    'dog','frog','horse','ship','truck'
)


# -----------------------------
# Transform (TEST ONLY)
# -----------------------------
transform = transforms.Compose([

    transforms.Resize((32,32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])


# -----------------------------
# Predict Function
# -----------------------------
def predict_image(image: Image.Image):

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, 1)

    return classes[pred.item()], conf.item()


# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = predict_image(image)

    return {
        "prediction": label,
        "confidence": round(confidence, 3)
    }
