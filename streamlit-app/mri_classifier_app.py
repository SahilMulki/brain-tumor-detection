import base64
import io
import os
import re

import cv2
import gdown
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from torchvision import transforms
from PIL import Image

# GROUP'S MODELS
# ===============

# Angel CNN architecture
class Angel_CNN(nn.Module):
    def __init__(self):
        super(Angel_CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # cut image size by half

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 256x256
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # 128x128
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 64x64
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # 32x32
        self.bn4 = nn.BatchNorm2d(num_features=128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # 16x16
        self.bn5 = nn.BatchNorm2d(num_features=256)

        self.dropout = nn.Dropout()
        self.full_conn1 = nn.Linear(in_features=256 * 8 * 8, out_features=512)  # 8x8
        self.full_conn2 = nn.Linear(in_features=512, out_features=4)  # 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.full_conn1(x))
        x = self.dropout(x)
        x = self.full_conn2(x)
        return x

# Sahil CNN architecture
class Sahil_CNN(nn.Module):
    def __init__(self, num_classes=4, in_channels=1, img_size=256):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.conv = nn.Sequential(
            block(in_channels, 32),   # 256 -> 128
            block(32, 64),            # 128 -> 64
            block(64, 128),           # 64  -> 32
            block(128, 128),          # 32  -> 16
        )

        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, img_size, img_size)).flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Yi CNN architecture
class Yi_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)  # 64 -> 16
        )

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# Izzie CNN architecture
class Izzie_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Izzie_CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # cut image size by half

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # input size: 256x256
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # input size: 128x128
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # input size: 64x64
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # input size: 32x32
        self.bn4 = nn.BatchNorm2d(num_features=128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # input size: 16x16
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.spatial_dropout5 = nn.Dropout2d(0.1)   # light regularization, only on the deepest/widest block

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 256 x 8 x 8 -> 256 x 1 x 1

        self.dropout = nn.Dropout(0.5)
        self.full_conn1 = nn.Linear(in_features=256, out_features=256)
        self.full_conn2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # output size: 128x128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # output size: 64x64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # output size: 32x32
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # output size: 16x16
        x = self.spatial_dropout5(self.pool(F.relu(self.bn5(self.conv5(x)))))  # output size: 8x8

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)  # (batch, 256)

        x = F.relu(self.full_conn1(x))
        x = self.dropout(x)
        x = self.full_conn2(x)
        return x


# GROUP'S PREPROCESSING PIPELINES
# ================================

# Sahil margin cropping helper function via OpenCV
def crop_brain_region(pil_img):
    gray = np.array(pil_img.convert("L"))
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return pil_img.crop((x, y, x + w, y + h))

# force 3-channel input for the RGB pipelines (grayscale / RGBA uploads would
# otherwise break Normalize's 3-mean/3-std tensors)
def to_rgb(pil_img):
    return pil_img.convert("RGB")

# Angel preprocessing pipeline
def angel_preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

# Sahil preprocessing pipeline
def sahil_preprocess(image):
    transform = transforms.Compose([
        transforms.Lambda(crop_brain_region),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

# Yi preprocessing pipeline
def yi_preprocess(image):
    transform = transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# Hamet Pipeline (TensorFlow: RGB, 3 Channels, Scaled 0.0 - 1.0)
# mirrors training: image_dataset_from_directory loads RGB at 256x256 (bilinear), then x/255
def hamet_preprocess(image):
    img = image.convert("RGB").resize((256, 256), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

# Izzie preprocessing pipeline
def izzie_preprocess(image):
    transform = transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Shape: 1 x 3 x 256 x 256
    return transform(image).unsqueeze(0)


# model registry and metadata config — this drives the whole UI (registry rows,
# header, preprocess pipeline, panel state)
MODEL_CONFIG = {
    "Angel Model": {
        "file_id": "17ajFrT4D4p--d4DVeFDK2pZXqCeeFUIm",
        "file_name": "brain_mri_cnn.pth",
        "model_name": Angel_CNN,
        "preprocess_fn": angel_preprocess,
        "framework": "PyTorch",
        "accuracy": "~93%",
        "description": "5-layer convolutional network with the highest recall across tumor categories.",
        "preprocess_steps": ["Resize 256×256", "Grayscale ×1", "Normalize μ 0.5 σ 0.5"],
    },
    "Sahil Model": {
        "file_id": "1VzAvU5BIMeZHHxI1b2boRVCG7Nok6xkt",
        "file_name": "best_cnn.pt",
        "model_name": Sahil_CNN,
        "preprocess_fn": sahil_preprocess,
        "framework": "PyTorch",
        "accuracy": "~85%",
        "description": "Convolutional network with contour margin cropping for high recall across tumor types.",
        "preprocess_steps": ["Contour crop", "Grayscale ×1", "Resize 256×256", "Normalize μ 0.5 σ 0.5"],
    },
    "Yi Model": {
        "file_id": "1erH01ontmINe8Il-Ap3_In_gVEoBsphD",
        "file_name": "yi_cnn_ver3_result.pth",
        "model_name": Yi_CNN,
        "preprocess_fn": yi_preprocess,
        "framework": "PyTorch",
        "accuracy": "~90%",
        "description": "3-channel convolutional network with adaptive average pooling.",
        "preprocess_steps": ["RGB ×3", "Resize 256×256", "Normalize μ 0.5 σ 0.5"],
    },
    "Hamet Model": {
        "file_id": "1t33a801MgyRScx0NKvSwoTqcvJ-bRg9r",
        "file_name": "CNNTest_v2.0.keras",
        "model_name": None,  # Keras loads the full model, no architecture class needed
        "preprocess_fn": hamet_preprocess,
        "framework": "TensorFlow",
        "accuracy": "~90%",
        "description": "4-block convolutional network (16-32-64-128 filters) with progressive dropout, "
                       "trained with Adam and early stopping.",
        "preprocess_steps": ["RGB ×3", "Resize 256×256 bilinear", "Scale 0–1"],
    },
    "Izzie Model": {
        "file_id": "1iDdZ5XLqM5o5xBOJ1Y2FOe7_leHKL_qg",
        "file_name": "izzie_cnn_weights.pth",
        "model_name": Izzie_CNN,
        "preprocess_fn": izzie_preprocess,
        "framework": "PyTorch",
        "accuracy": "~90%",
        "description": "Convolutional network with spatial dropout and adaptive pooling.",
        "preprocess_steps": ["RGB ×3", "Resize 256×256", "Normalize μ 0.5 σ 0.5"],
    },
    "Mike Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "preprocess_fn": None,
        "framework": "—",
        "accuracy": "—",
        "description": "Awaiting deployment configuration.",
        "preprocess_steps": ["Awaiting configuration"],
    }
}

# a model is deployed once its Drive file ID is filled in
for _cfg in MODEL_CONFIG.values():
    _cfg["active"] = bool(_cfg["file_id"])

MODEL_NAMES = list(MODEL_CONFIG.keys())
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

TEAM = [
    ("Angel Velasquez", "https://www.linkedin.com/in/angel-velasquez-419338320/", "https://github.com/avelasquez7711-bit"),
    ("Sahil Mulki", "https://www.linkedin.com/in/smulki/", "https://github.com/SahilMulki"),
    ("Yi Zou", "https://www.linkedin.com/in/yi-zou-222898296/", "https://github.com/yizou275"),
    ("Hamet Coulibaly", "https://www.linkedin.com/in/hametc/", "https://github.com/hamet-c"),
    ("Michael Abraham", "https://www.linkedin.com/in/michael-abraham-a9b120214/", "https://github.com/mikeiioo"),
    ("Izzie Lee", "https://www.linkedin.com/in/izzie-lee/", "https://github.com/jiwonizzielee"),
]

# download trained model weights from Google Drive; loads model weights
@st.cache_resource
def load_model(file_id, file_name, model_name):
    if not os.path.exists(file_name):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)

    # TensorFlow model Loading
    if file_name.endswith(('.keras', '.h5')):
        model = tf.keras.models.load_model(file_name)
        return model, "tensorflow"

    # PyTorch model Loading
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model_name()
    model.load_state_dict(torch.load(file_name, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# run forward pass thru model for inference
def run_inference(model, device, original_image, preprocess_fn, classes):
    # apply the selected model's custom pre-processing
    input_data = preprocess_fn(original_image)

    # TensorFlow inference
    if device == "tensorflow":
        probabilities = model.predict(input_data, verbose=0)[0]
        predicted_idx = int(np.argmax(probabilities))
        confidence_pct = float(probabilities[predicted_idx]) * 100
        predicted_class = classes[predicted_idx]
        return predicted_class, confidence_pct, probabilities

    # PyTorch inference
    input_tensor = input_data.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = classes[predicted_idx.item()]
        confidence_pct = confidence.item() * 100

    return predicted_class, confidence_pct, probabilities.cpu().numpy()


# INK THEME — design tokens resolved for the dark "Ink" redesign
# ==============================================================

def html(markup):
    """Collapse authored indentation so Streamlit's markdown parser doesn't
    treat indented lines as code blocks. Line breaks between tags close up;
    line breaks inside a sentence keep their word space."""
    markup = re.sub(r">\s*\n\s*<", "><", markup)
    return re.sub(r"\s*\n\s*", " ", markup).strip()


def write_html(markup):
    st.markdown(html(markup), unsafe_allow_html=True)


THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;700&family=Barlow+Condensed:wght@400;600&display=swap');

:root{
  --ind-bg:#2b2b2d;
  --ind-sidebar:#333335;
  --ind-panel:#373739;
  --ind-ink:#f5f5f8;
  --ind-acc:#94bce3;
  --ind-acc-base:#5980a6;
  --ind-hair:rgba(245,245,248,.16);
  --ind-rule:rgba(245,245,248,.08);
  --ind-track:rgba(245,245,248,.12);
  --ind-font-body:"Barlow",system-ui,-apple-system,sans-serif;
  --ind-font-head:"Barlow Condensed",system-ui,-apple-system,sans-serif;
}

/* — ground — */
html, body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--ind-bg); }
.stApp{
  font-family:var(--ind-font-body); font-size:15px; line-height:1.55;
  font-weight:400; color:var(--ind-ink);
}
.stApp p, .stApp div, .stApp span, .stApp label, .stApp li, .stApp a,
.stApp button, .stApp input, .stApp textarea, .stApp small,
.stApp strong, .stApp em, .stApp code{ font-family:var(--ind-font-body); }
/* keep Streamlit's icon fonts intact */
[data-testid="stIconMaterial"], .material-icons, .material-icons-outlined,
[class*="material-symbols"]{ font-family:'Material Symbols Rounded','Material Icons' !important; }

.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6{
  font-family:var(--ind-font-head); font-weight:600;
  line-height:1.12; letter-spacing:-.015em; color:var(--ind-ink);
}
.stApp a{ color:var(--ind-acc); text-decoration:none; text-underline-offset:3px; }
.stApp a:hover{ text-decoration:underline; }
.stApp :focus-visible{ outline:2px solid var(--ind-acc) !important; outline-offset:2px !important; box-shadow:none !important; }
::selection{ background:rgba(148,188,227,.3); }

/* — chrome: no top bar, no decoration, keep the toolbar reachable — */
[data-testid="stHeader"]{ background:transparent; height:0; min-height:0; }
[data-testid="stDecoration"], [data-testid="stAppDeployButton"]{ display:none; }
[data-testid="stToolbar"]{ right:.5rem; top:.25rem; }
footer{ display:none; }

/* Streamlit pulls every markdown block up by 1rem and lets the vertical
   block's gap do the spacing — zero it so the gaps below are the real ones */
[data-testid="stMarkdown"] [data-testid="stMarkdownContainer"]{ margin-bottom:0; }

/* — main column: fluid, 27px / 64px padding — */
[data-testid="stMainBlockContainer"], .stMainBlockContainer, .block-container{
  max-width:none; padding:27px 64px 48px;
}
[data-testid="stMain"] div[data-testid="stVerticalBlock"]{ gap:10.2px; }

/* — sidebar: 300px, #333335, hairline edge — */
[data-testid="stSidebar"]{
  width:300px !important; min-width:300px !important; max-width:300px !important;
  background:var(--ind-sidebar); border-right:1px solid var(--ind-hair);
}
[data-testid="stSidebar"] > div:first-child{ width:300px; background:var(--ind-sidebar); }
[data-testid="stSidebarHeader"]{ padding:.5rem .75rem 0; height:auto; min-height:0; }
[data-testid="stSidebarUserContent"]{ padding:20px 20px 27px; }
/* top-level sidebar stack: ~27px rhythm, tall enough to pin the disclaimer */
[data-testid="stSidebarUserContent"] > div > div[data-testid="stVerticalBlock"]{
  gap:27.2px; min-height:calc(100vh - 100px);
}
/* groups inside it (registry, team) keep the tighter space-2 rhythm */
[data-testid="stSidebarUserContent"] div[data-testid="stVerticalBlock"]
  div[data-testid="stVerticalBlock"]{ gap:6.8px; }
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > *:has(.ind-disclaimer){ margin-top:auto; }
[data-testid="stSidebarResizeHandle"]{ display:none; }

/* — shared type — */
.ind-label{
  font-size:11px; text-transform:uppercase; letter-spacing:.08em;
  line-height:1.2; color:rgba(245,245,248,.6);
}
.ind-label-spaced{ margin-top:13.6px; }
/* .stApp prefix outranks Streamlit's own heading sizes */
.stApp .ind-h2{ font-family:var(--ind-font-head); font-weight:600; font-size:32px;
  line-height:1.12; letter-spacing:-.015em; margin:0 0 6.8px; padding:0; color:var(--ind-ink); }
.stApp .ind-h3{ font-family:var(--ind-font-head); font-weight:600; font-size:25px;
  line-height:1.12; letter-spacing:-.015em; margin:0 0 2px; padding:0; color:var(--ind-ink); }

/* — sidebar: brand — */
.ind-brand{ font-family:var(--ind-font-head); font-weight:600; font-size:20px;
  line-height:1.05; margin:0 0 3.4px; color:var(--ind-ink); }
.ind-brand-sub{ font-size:12.5px; line-height:1.4; color:rgba(245,245,248,.55); }

/* — sidebar: model registry (st.radio restyled as selectable rows) — */
[data-testid="stSidebar"] div[role="radiogroup"]{
  display:flex; flex-direction:column; gap:2px;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label{
  position:relative; display:flex; align-items:center; width:100%;
  margin:0; padding:8px 10px; cursor:pointer;
  background:transparent; border:1px solid transparent; border-radius:4px;
}
/* BaseWeb nests the row content two divs deep — unwrap it so the label
   markup becomes the row's flex children */
[data-testid="stSidebar"] div[role="radiogroup"] > label > div,
[data-testid="stSidebar"] div[role="radiogroup"] > label > div > div{ display:contents; }
/* drop the native radio mark: the block immediately before the label content
   (Streamlit already keeps the real <input> visually hidden but focusable) */
[data-testid="stSidebar"] div[role="radiogroup"] > label
  div:has(+ div[data-testid="stMarkdownContainer"]){ display:none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label
  div[data-testid="stMarkdownContainer"]{ flex:1 1 auto; min-width:0; }
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p{
  display:flex; align-items:center; gap:10px; margin:0;
  font-size:13.5px; line-height:1.3; color:var(--ind-ink);
}
/* 1st inline-code slot = two digit index, 2nd = accuracy (pushed right) */
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p code{
  background:none; padding:0; border:0; white-space:nowrap;
}
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p code:nth-of-type(1){
  flex:none; width:20px; font-family:var(--ind-font-head); font-size:12px;
  letter-spacing:.06em; color:rgba(245,245,248,.45);
}
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p code:nth-of-type(2){
  flex:none; margin-left:auto; font-size:12px; font-variant-numeric:tabular-nums;
  color:rgba(245,245,248,.5);
}
/* status dot: bold ● = deployed, italic ○ = awaiting weights */
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p strong,
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p em{
  flex:none; font-size:9px; line-height:1; font-weight:400; font-style:normal;
}
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p strong{ color:var(--ind-acc); }
[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p em{ color:rgba(245,245,248,.4); }

[data-testid="stSidebar"] div[role="radiogroup"] > label:not([data-selected="true"]):hover{
  background:rgba(245,245,248,.06);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"],
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
  background:rgba(116,157,196,.22); border-color:rgba(148,188,227,.45);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"]
  [data-testid="stMarkdownContainer"] p code:nth-of-type(1),
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)
  [data-testid="stMarkdownContainer"] p code:nth-of-type(1){ color:var(--ind-acc); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:focus-visible){
  outline:2px solid var(--ind-acc); outline-offset:2px;
}

.ind-legend{ display:flex; align-items:center; gap:6.8px; margin-top:6.8px;
  font-size:11.5px; color:rgba(245,245,248,.5); }
.ind-dot{ display:inline-block; width:7px; height:7px; border-radius:50%; }
.ind-dot-on{ background:var(--ind-acc); }
.ind-dot-off{ border:1px solid rgba(245,245,248,.4); }

/* — sidebar: team + disclaimer — */
.ind-team-row{ display:flex; justify-content:space-between; align-items:baseline;
  gap:10.2px; font-size:12.5px; padding:2.5px 0; color:var(--ind-ink); }
.ind-team-links{ font-size:11.5px; white-space:nowrap; }
.ind-team-links a{ color:var(--ind-acc); text-decoration:none; }
.ind-team-links a:hover{ text-decoration:underline; }
.ind-disclaimer{ font-size:11.5px; line-height:1.5; color:rgba(245,245,248,.5); }

/* — main: header — */
.ind-header{ margin-bottom:17px; }  /* + the 10.2px block gap = space-8 */
.ind-kicker{ font-size:11px; text-transform:uppercase; letter-spacing:.08em;
  line-height:1.2; margin:0 0 6.8px; color:var(--ind-acc); }
.ind-desc{ font-size:15px; line-height:1.55; max-width:560px;
  margin:0 0 10.2px; color:rgba(245,245,248,.75); }
.ind-tags{ display:flex; gap:6.8px; flex-wrap:wrap; }
.ind-tag{ display:inline-flex; align-items:center; font-size:11px; letter-spacing:.02em;
  padding:4px 10px; border-radius:3px; background:rgba(245,245,248,.08);
  color:rgba(245,245,248,.75); }
.ind-tag-accent{ background:rgba(148,188,227,.16); color:var(--ind-acc); }

/* — main: two-column grid, 380px / 1fr, 64px gap — */
[data-testid="stMain"] div[data-testid="stHorizontalBlock"]{
  gap:64px; flex-wrap:nowrap; align-items:flex-start;
}
[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child{
  flex:0 0 380px; width:380px; min-width:380px;
}
[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child{
  flex:1 1 auto; width:auto; min-width:0;
}
@media (max-width:1080px){
  [data-testid="stMainBlockContainer"], .stMainBlockContainer, .block-container{ padding:27px 27px 48px; }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"]{ flex-wrap:wrap; gap:40px; }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child{
    flex:1 1 100%; width:100%; min-width:0;
  }
}

/* — input scan: the file uploader IS the square drop frame — */
[data-testid="stFileUploader"] > label{ display:none; }
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"]{
  position:relative; aspect-ratio:1/1; width:100%; padding:0;
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px;
  background:var(--ind-panel); border:1px solid var(--ind-rule); border-radius:4px;
  overflow:hidden;
}
[data-testid="stFileUploaderDropzone"]::before, [data-testid="stFileUploadDropzone"]::before{
  content:"Drop an MRI scan"; font-size:13px; color:rgba(245,245,248,.45);
}
[data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploadDropzoneInstructions"]{ display:none; }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFileList"]{ display:none; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploadDropzone"] button{
  position:relative; z-index:2;
  font-family:var(--ind-font-head); font-weight:600; font-size:13px; line-height:1.2;
  padding:7px 16px; border-radius:4px;
  background:transparent; color:var(--ind-ink);
  border:1px solid var(--ind-hair); box-shadow:none;
}
[data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stFileUploadDropzone"] button:hover{
  background:rgba(245,245,248,.07); color:var(--ind-ink); border-color:var(--ind-hair);
}
.ind-caption{ font-size:12px; line-height:1.4; color:rgba(245,245,248,.5); }

/* — input scan: preprocess pipeline — */
.ind-step{ display:flex; gap:10.2px; align-items:baseline; padding:6.8px 0;
  border-bottom:1px solid var(--ind-rule); font-size:13px; color:var(--ind-ink); }
.ind-step-n{ font-family:var(--ind-font-head); font-size:12px;
  letter-spacing:.06em; color:rgba(245,245,248,.45); }

/* — primary button — */
[data-testid="stMain"] div[data-testid="stElementContainer"]:has(.stButton){ margin-top:10.2px; }
[data-testid="stMain"] .stButton > button{
  background:var(--ind-acc-base); color:#f2f2f3; border:1px solid var(--ind-acc-base);
  border-radius:4px; padding:10px 22px; box-shadow:none;
  font-family:var(--ind-font-head); font-weight:600; font-size:14px; line-height:1.2;
}
[data-testid="stMain"] .stButton > button:hover:not(:disabled){
  background:var(--ind-acc); border-color:var(--ind-acc); color:#26262a;
}
[data-testid="stMain"] .stButton > button:active:not(:disabled){
  background:var(--ind-acc); border-color:var(--ind-acc); color:#26262a;
}
[data-testid="stMain"] .stButton > button:disabled{
  opacity:.45; cursor:not-allowed; background:var(--ind-acc-base); color:#f2f2f3;
}

/* — diagnostic result panel — */
.ind-panel{
  display:flex; flex-direction:column; gap:20.4px; padding:27.2px;
  background:var(--ind-panel); border:1px solid var(--ind-rule); border-radius:4px;
}
.ind-panel p{ margin:0; max-width:420px; font-size:15px; line-height:1.55;
  color:rgba(245,245,248,.6); }
.ind-panel-top{ display:flex; justify-content:space-between; align-items:center; gap:10.2px; }
.ind-hero{ display:flex; gap:20.4px; align-items:baseline; flex-wrap:wrap; }
.ind-figure{ font-family:var(--ind-font-head); font-weight:600; font-size:68px;
  line-height:.9; letter-spacing:-.01em; color:var(--ind-ink); }
.ind-figure span{ font-size:26px; }
.ind-sub{ font-size:13px; line-height:1.4; color:rgba(245,245,248,.55); }
.ind-probs{ display:flex; flex-direction:column; gap:10.2px;
  padding-top:20.4px; border-top:1px solid var(--ind-hair); }
.ind-bar-row{ display:grid; grid-template-columns:110px 1fr 60px; gap:13.6px; align-items:center; }
.ind-bar-name{ font-size:13px; color:var(--ind-ink); }
.ind-bar-name-empty{ font-size:13px; color:rgba(245,245,248,.45); }
.ind-bar-track{ position:relative; height:5px; border-radius:999px;
  background:var(--ind-track); overflow:hidden; }
.ind-bar-fill{ position:absolute; inset:0 auto 0 0; background:rgba(148,188,227,.4);
  transition:width .45s ease; animation:ind-bar-grow .45s ease; }
.ind-bar-fill.is-top{ background:var(--ind-acc); }
@keyframes ind-bar-grow{ from{ width:0; } }
.ind-bar-pct{ text-align:right; font-size:12.5px; font-variant-numeric:tabular-nums;
  color:rgba(245,245,248,.55); }
.ind-bar-pct.is-top{ color:var(--ind-ink); font-weight:700; }
.ind-bar-pct-empty{ text-align:right; font-size:12.5px; color:rgba(245,245,248,.35); }
.ind-foot{ font-size:12px; line-height:1.4; color:rgba(245,245,248,.5); }

/* — spinner / toasts inherit the palette — */
.stSpinner > div{ color:rgba(245,245,248,.6); font-size:13px; }
"""

# CSS applied only while a scan is loaded: the frame becomes the scan itself.
# The dropzone <section> keeps its own click handler, so the bare frame still
# opens the file browser — no chip or button needs to sit on top of the image.
SCAN_CSS = """
[data-testid="stFileUploaderDropzone"]::before, [data-testid="stFileUploadDropzone"]::before{
  display:none;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stFileChips"]{ display:none; }
[data-testid="stFileUploaderDropzone"]::after, [data-testid="stFileUploadDropzone"]::after{
  content:""; position:absolute; inset:0; pointer-events:none;
  background-image:url("%s"); background-size:cover; background-position:center;
}
"""


# UI HELPERS
# ==========

def registry_label(index, name, config):
    """Row markup for the registry radio: `01` name `~93%` ● / ○.

    The two inline-code slots and the bold/italic dot are the hooks the
    injected CSS lays out (index left, accuracy right, status dot last).
    """
    dot = "**●**" if config["active"] else "*○*"
    return f"`{index + 1:02d}` {name} `{config['accuracy']}` {dot}"


def pipeline_html(steps):
    rows = "".join(
        f'<div class="ind-step"><span class="ind-step-n">{i + 1:02d}</span><span>{step}</span></div>'
        for i, step in enumerate(steps)
    )
    return f'<div class="ind-pipeline">{rows}</div>'


def bar_rows_html(probabilities, top_index):
    rows = []
    for i, name in enumerate(CLASSES):
        pct = float(probabilities[i]) * 100
        top = " is-top" if i == top_index else ""
        rows.append(
            f'<div class="ind-bar-row">'
            f'<div class="ind-bar-name">{name}</div>'
            f'<div class="ind-bar-track"><div class="ind-bar-fill{top}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="ind-bar-pct{top}">{pct:.1f}%</div>'
            f'</div>'
        )
    return "".join(rows)


def empty_bar_rows_html():
    return "".join(
        f'<div class="ind-bar-row">'
        f'<div class="ind-bar-name-empty">{name}</div>'
        f'<div class="ind-bar-track"></div>'
        f'<div class="ind-bar-pct-empty">—</div>'
        f'</div>'
        for name in CLASSES
    )


def results_panel_html(model_name, prediction, confidence, probabilities):
    top_index = int(np.argmax(probabilities))
    status = "No tumor" if prediction == "No Tumor" else "Tumor detected"
    return f"""
      <div class="ind-panel">
        <div class="ind-panel-top">
          <span class="ind-label">Prediction</span>
          <span class="ind-tag ind-tag-accent">{status}</span>
        </div>
        <div class="ind-hero">
          <div class="ind-figure">{confidence:.1f}<span>%</span></div>
          <div>
            <div class="ind-h3">{prediction}</div>
            <div class="ind-sub">Softmax confidence · {model_name}</div>
          </div>
        </div>
        <div class="ind-probs">
          <div class="ind-label">Class probabilities</div>
          {bar_rows_html(probabilities, top_index)}
        </div>
        <div class="ind-foot">Softmax over 4 classes · academic demonstration, not a clinical diagnosis.</div>
      </div>
    """


def awaiting_panel_html():
    return f"""
      <div class="ind-panel">
        <p>Add a scan and run inference — the predicted class and confidence breakdown will appear here.</p>
        <div class="ind-probs" style="border-top:none;padding-top:0">
          {empty_bar_rows_html()}
        </div>
      </div>
    """


def pending_panel_html(model_name):
    return f"""
      <div class="ind-panel" style="align-items:flex-start;gap:13.6px">
        <span class="ind-tag">Awaiting weights</span>
        <p>{model_name} activates once its Google Drive file ID is configured in MODEL_CONFIG.
        The registry entry, preprocessing pipeline and description are already wired.</p>
      </div>
    """


def error_panel_html(message):
    return f"""
      <div class="ind-panel" style="align-items:flex-start;gap:13.6px">
        <span class="ind-tag">Inference unavailable</span>
        <p>{message}</p>
      </div>
    """


def scan_signature(uploaded):
    """Identify the current upload so a result is only shown for the scan and
    model it was produced from."""
    if uploaded is None:
        return None
    return getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"


# PAGE
# ====

st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# the uploader's value is already in session state when the script reruns, so the
# frame's background image can be styled before the widget itself is drawn
scan_file = st.session_state.get("scan_upload")
scan_bytes = scan_file.getvalue() if scan_file is not None else None

page_css = THEME_CSS
if scan_bytes:
    mime = getattr(scan_file, "type", None) or "image/png"
    data_uri = f"data:{mime};base64,{base64.b64encode(scan_bytes).decode('ascii')}"
    page_css += SCAN_CSS % data_uri
st.markdown(f"<style>{page_css}</style>", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None

# SIDEBAR
# =======
with st.sidebar:
    write_html("""
      <div>
        <div class="ind-brand">Brain MRI<br>Tumor Classifier</div>
        <div class="ind-brand-sub">Multi-architecture CNN registry · academic demo</div>
      </div>
    """)

    with st.container():
        write_html('<div class="ind-label">Model registry</div>')
        selected_model_name = st.radio(
            "Model registry",
            MODEL_NAMES,
            index=0,
            key="selected_model",
            label_visibility="collapsed",
            format_func=lambda name: registry_label(
                MODEL_NAMES.index(name), name, MODEL_CONFIG[name]
            ),
        )
        write_html("""
          <div class="ind-legend">
            <span class="ind-dot ind-dot-on"></span><span>deployed</span>
            <span style="margin-left:6.8px" class="ind-dot ind-dot-off"></span><span>awaiting weights</span>
          </div>
        """)

    with st.container():
        write_html('<div class="ind-label">Project team</div>')
        write_html(
            "<div>"
            + "".join(
                f'<div class="ind-team-row"><span>{name}</span>'
                f'<span class="ind-team-links">'
                f'<a href="{linkedin}" target="_blank" rel="noopener">LinkedIn</a> · '
                f'<a href="{github}" target="_blank" rel="noopener">GitHub</a>'
                f'</span></div>'
                for name, linkedin, github in TEAM
            )
            + "</div>"
        )

    write_html(
        '<div class="ind-disclaimer">Academic demonstration only — '
        'not intended for medical diagnosis.</div>'
    )

selected_config = MODEL_CONFIG[selected_model_name]
selected_index = MODEL_NAMES.index(selected_model_name)

# MAIN — header
# =============
accuracy_tag = (
    "accuracy pending"
    if selected_config["accuracy"] == "—"
    else f"{selected_config['accuracy']} test accuracy"
)
write_html(f"""
  <div class="ind-header">
    <div class="ind-kicker">Model {selected_index + 1:02d} · {selected_config['framework']}</div>
    <h2 class="ind-h2">{selected_model_name}</h2>
    <p class="ind-desc">{selected_config['description']}</p>
    <div class="ind-tags">
      <span class="ind-tag">{selected_config['framework']}</span>
      <span class="ind-tag">{accuracy_tag}</span>
    </div>
  </div>
""")

# MAIN — input scan / diagnostic result
# =====================================
scan_column, result_column = st.columns(2)

with scan_column:
    write_html('<div class="ind-label">Input scan</div>')
    st.file_uploader(
        "Input scan",
        type=["jpg", "jpeg", "png"],
        key="scan_upload",
        label_visibility="collapsed",
    )
    write_html(
        '<div class="ind-caption">Drag a scan onto the frame or click to browse · JPG / PNG</div>'
    )

    write_html('<div class="ind-label ind-label-spaced">Preprocess pipeline</div>')
    write_html(pipeline_html(selected_config["preprocess_steps"]))

    if selected_config["active"]:
        run_clicked = st.button(
            "Run inference",
            type="primary",
            disabled=scan_bytes is None,
        )

        if run_clicked and scan_bytes is not None:
            with st.spinner("Loading weights and running inference…"):
                try:
                    model, device = load_model(
                        file_id=selected_config["file_id"],
                        file_name=selected_config["file_name"],
                        model_name=selected_config["model_name"],
                    )
                    original_image = Image.open(io.BytesIO(scan_bytes))
                    prediction, confidence, probabilities = run_inference(
                        model,
                        device,
                        original_image,
                        selected_config["preprocess_fn"],
                        CLASSES,
                    )
                    st.session_state.result = {
                        "model": selected_model_name,
                        "signature": scan_signature(scan_file),
                        "prediction": prediction,
                        "confidence": confidence,
                        "probabilities": np.asarray(probabilities, dtype=float),
                        "error": None,
                    }
                except Exception as exc:  # surfaced in the result panel, not a red box
                    st.session_state.result = {
                        "model": selected_model_name,
                        "signature": scan_signature(scan_file),
                        "error": f"Could not load {selected_model_name}: {exc}",
                    }

with result_column:
    write_html('<div class="ind-label">Diagnostic result</div>')

    result = st.session_state.result
    fresh = (
        result is not None
        and result["model"] == selected_model_name
        and result["signature"] == scan_signature(scan_file)
    )

    if not selected_config["active"]:
        write_html(pending_panel_html(selected_model_name))
    elif fresh and result.get("error"):
        write_html(error_panel_html(result["error"]))
    elif fresh:
        write_html(
            results_panel_html(
                selected_model_name,
                result["prediction"],
                result["confidence"],
                result["probabilities"],
            )
        )
    else:
        write_html(awaiting_panel_html())
