import torch
import torch.nn as nn
import torchvision.transforms as transforms

from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image
import io
import uuid


# -----------------------------
# Session History Store
# -----------------------------
user_history = {}
MAX_HISTORY = 5


# -----------------------------
# App
# -----------------------------
app = FastAPI()


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# CNN (MATCH TRAINING)
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
# Load Model
# -----------------------------
model = CNN().to(device)

model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()


# -----------------------------
# Classes
# -----------------------------
classes = (
    'plane','car','bird','cat','deer',
    'dog','frog','horse','ship','truck'
)


# -----------------------------
# Transform
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
# Prediction Logic
# -----------------------------
def predict_image(image):

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        out = model(image)

        probs = torch.softmax(out, dim=1)

        conf, pred = torch.max(probs, 1)

    return classes[pred.item()], conf.item()


# -----------------------------
# Health Check (HF Safe)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Homepage (UI)
# -----------------------------
@app.api_route("/", methods=["GET"], response_class=HTMLResponse)
def home(request: Request, response: Response):

    # Get / create session id
    session_id = request.cookies.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie("session_id", session_id)

    history = user_history.get(session_id, [])


    history_html = ""

    for item in reversed(history):

        history_html += f"""
        <div class="hist-item">
            <b>{item['label']}</b> ({item['conf']*100:.1f}%)
        </div>
        """


    return f"""
    <html>
    <head>
        <title>CIFAR-10 Classifier</title>

        <meta name="viewport" content="width=device-width, initial-scale=1">

        <style>

            body {{
                font-family: Arial;
                background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                color: white;
                text-align: center;
                margin: 0;
                padding: 0;
            }}

            .container {{
                max-width: 520px;
                margin: 50px auto;
                background: #111;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0px 0px 25px rgba(0,255,200,0.3);
            }}

            h1 {{ color: #00ffcc; }}

            p {{ color: #ccc; }}

            input {{
                margin: 15px 0;
                padding: 10px;
                width: 100%;
            }}

            button {{
                background: #00ffcc;
                border: none;
                padding: 12px;
                font-size: 16px;
                cursor: pointer;
                border-radius: 6px;
                width: 100%;
            }}

            button:hover {{
                background: #00e6b8;
            }}

            #preview {{
                width: 100%;
                height: 200px;
                object-fit: contain;
                border: 2px dashed #555;
                margin-bottom: 10px;
                display: none;
            }}

            .history {{
                margin-top: 20px;
                text-align: left;
            }}

            .hist-item {{
                background: #1b1b1b;
                padding: 8px;
                margin: 5px 0;
                border-radius: 5px;
                font-size: 14px;
            }}

            .footer {{
                margin-top: 15px;
                font-size: 12px;
                color: #888;
            }}

        </style>

    </head>


    <body>

        <div class="container">

            <h1>CIFAR-10 Classifier</h1>

            <p>Upload an image to classify using CNN</p>

            <img id="preview">

            <form action="/predict_ui" method="post" enctype="multipart/form-data">

                <input type="file" name="file" accept="image/*"
                       onchange="loadPreview(event)" required>

                <button type="submit">Predict</button>

            </form>


            <div class="history">

                <h3>Recent Predictions</h3>

                {history_html}

            </div>


            <div class="footer">
                Built with PyTorch + FastAPI
            </div>

        </div>


        <script>

            function loadPreview(event){{
                const img = document.getElementById("preview");

                img.src = URL.createObjectURL(event.target.files[0]);

                img.style.display = "block";
            }}

        </script>


    </body>
    </html>
    """


# -----------------------------
# Redirect GET
# -----------------------------
@app.get("/predict_ui")
def predict_ui_get():
    return RedirectResponse("/")


# -----------------------------
# UI Prediction
# -----------------------------
@app.post("/predict_ui", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    response: Response,
    file: UploadFile = File(...)
):

    # Session id
    session_id = request.cookies.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie("session_id", session_id)

    history = user_history.get(session_id, [])


    # Read image
    data = await file.read()

    image = Image.open(io.BytesIO(data)).convert("RGB")

    label, conf = predict_image(image)


    # Save history
    history.append({
        "label": label,
        "conf": conf
    })

    if len(history) > MAX_HISTORY:
        history.pop(0)

    user_history[session_id] = history


    percent = int(conf * 100)


    return f"""
    <html>

    <head>

        <title>Result</title>

        <meta name="viewport" content="width=device-width, initial-scale=1">

        <style>

            body {{
                font-family: Arial;
                background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                color: white;
                text-align: center;
                margin: 0;
                padding: 0;
            }}

            .box {{
                max-width: 480px;
                margin: 60px auto;
                background: #111;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0px 0px 25px rgba(0,255,200,0.3);
            }}

            h1 {{ color: #00ffcc; }}

            .result {{
                font-size: 30px;
                margin: 15px 0;
            }}

            .conf-text {{
                color: #aaa;
                margin-bottom: 10px;
            }}

            .bar-bg {{
                width: 100%;
                background: #333;
                border-radius: 6px;
                margin-bottom: 15px;
            }}

            .bar {{
                height: 22px;
                width: {percent}%;
                background: #00ffcc;
                border-radius: 6px;
                transition: width 0.8s;
            }}

            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background: #00ffcc;
                color: black;
                text-decoration: none;
                border-radius: 5px;
            }}

            a:hover {{
                background: #00e6b8;
            }}

        </style>

    </head>


    <body>

        <div class="box">

            <h1>Prediction Result</h1>

            <div class="result">
                {label.upper()}
            </div>

            <div class="conf-text">
                Confidence: {conf:.2f}
            </div>


            <div class="bar-bg">
                <div class="bar"></div>
            </div>


            <a href="/">Try Another</a>

        </div>

    </body>
    </html>
    """
