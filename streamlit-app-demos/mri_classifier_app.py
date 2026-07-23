import os
import cv2
import gdown
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# default visual transform for UI rendering
visual_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1)
])


# model registry and metadata config
MODEL_CONFIG = {
    "Angel Model": {
        "file_id": "17ajFrT4D4p--d4DVeFDK2pZXqCeeFUIm",
        "file_name": "brain_mri_cnn.pth",
        "model_name": Angel_CNN,
        "preprocess_fn": angel_preprocess,
        "description": "Best used for general baseline feature extraction and balanced accuracy across all 4 tumor categories.",
    },
    "Sahil Model": {
        "file_id": "1VzAvU5BIMeZHHxI1b2boRVCG7Nok6xkt",
        "file_name": "best_cnn.pt",
        "model_name": Sahil_CNN,
        "preprocess_fn": sahil_preprocess,
        "description": "Optimized with OpenCV margin cropping for high recall across tumor types.",
    },
    "Yi Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "preprocess_fn": angel_preprocess,
        "description": "Awaiting deployment configurations...",
    },
    "Hamet Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "preprocess_fn": angel_preprocess,
        "description": "Awaiting deployment configurations...",
    },
    "Izzie Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "preprocess_fn": angel_preprocess,
        "description": "Awaiting deployment configurations...",
    },
    "Mike Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "preprocess_fn": angel_preprocess,
        "description": "Awaiting deployment configurations...",
    }
}

# download trained model weights from Google Drive; loads model weights
@st.cache_resource
def load_model(file_id: str, file_name: str, model_name: nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if not os.path.exists(file_name):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)
    
    model = model_name()
    model.load_state_dict(torch.load(file_name, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# run forward pass thru model for inference
def run_inference(model, device, original_image, preprocess_fn, classes):
    # Dynamically apply the selected model's custom pre-processing
    input_tensor = preprocess_fn(original_image).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = classes[predicted_idx.item()]
        confidence_pct = confidence.item() * 100
        
    return predicted_class, confidence_pct, probabilities


# setup Streamlit UI
st.set_page_config(page_title="Brain MRI Tumor Classifier", layout="centered")

# setup sidebar UI
with st.sidebar:
    st.title("Brain MRI Tumor Classifier")
    st.caption("AI Powered Classification | PyTorch Deployment")
    st.divider()
    
    st.subheader("⚙️ Model Settings")
    selected_model_name = st.selectbox(
        "Choose Model Architecture:",
        list(MODEL_CONFIG.keys())
    )
    
    st.divider()
    st.subheader("Project Team!")
    st.markdown("""
    * **Angel Velasquez** - [*LinkedIn*](https://linkedin.com) | [*GitHub*](https://github.com)
    * **Sahil Mulki** - [*LinkedIn*](https://linkedin.com) | [*GitHub*](https://github.com)
    * **Yi Zou** - [*LinkedIn*](https://linkedin.com) | [*GitHub*](https://github.com)
    * **Hamet Coulibaly** - [*LinkedIn*](https://linkedin.com) | [*GitHub*](https://github.com)
    * **Michael Abraham** - [*LinkedIn*](https://linkedin.com) | [*GitHub*](https://github.com)
    * **Izzie Lee** - [*LinkedIn*](https://linkedin.com) | [*GitHub*](https://github.com)
    * **Ying Lin Zhao** - [*LinkedIn*](https://linkedin.com) | [*GitHub*](https://github.com)
    """)
    st.divider()
    st.info("⚠️ **Disclaimer:** This tool is an academic demonstration and is **NOT** intended for official medical diagnosis.")

selected_config = MODEL_CONFIG[selected_model_name]

# setup main page layout
st.title("Brain MRI Tumor Classifier")
st.header(f"**{selected_model_name}**")
st.write(f"**Model Overview:** {selected_config['description']}")
st.markdown("--")
st.write("Upload a brain MRI scan (JPG, JPEG, or PNG) below. The model will preprocess it, run inference, and predict the clinical diagnosis.")
st.bottom.info("⚠️ **Disclaimer:** This tool is an academic demonstration and is **NOT** intended for official medical diagnosis.")

uploaded_file = st.file_uploader("Upload an MRI Scan...", type=["jpg", "jpeg", "png"])
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    display_image = visual_transform(original_image)  
    
    st.write("##### Uploaded Scan (Processed Preview)")
    st.image(display_image, width=300)

    # Validate if model class & file ID are configured
    if selected_config["model_name"] is not None and selected_config["file_id"] != "":
        try:
            if not os.path.exists(selected_config["file_name"]):
                st.toast(f"Downloading {selected_config['file_name']} from Google Drive...")
                
            model, device = load_model(
                file_id=selected_config["file_id"],
                file_name=selected_config["file_name"],
                model_name=selected_config["model_name"]
            )

            st.toast("*Running image inference...*")
            pred, conf, probs = run_inference(
                model, 
                device, 
                original_image, 
                selected_config["preprocess_fn"], 
                classes
            )
            
            st.divider()
            st.subheader("Diagnostic Results")
            
            if pred == "No Tumor":
                st.success(f"**Prediction: {pred}** (Confidence: {conf:.2f}%)")
            else:
                st.warning(f"**Tumor Detected: {pred}** (Confidence: {conf:.2f}%)")

            st.markdown("--")
            st.subheader("Classification Confidence Breakdown:")
            for idx, name in enumerate(classes):
                prob = probs[idx].item() * 100
                st.progress(prob / 100, text=f"{name}: {prob:.2f}%")
                
        except Exception as e:
            st.error(f"Could not load selected model: {e}")
    else:
        st.info(f"**{selected_model_name}** will be activated once Google Drive File ID and architecture are configured.")