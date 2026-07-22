import os
import gdown
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# copy same cnn model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # cut image size by half

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # input size: 256x256
        self.bn1 = nn.BatchNorm2d(num_features=16) # match out_channels

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # input size: 128x128
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # input size: 64x64
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # input size: 32x32
        self.bn4 = nn.BatchNorm2d(num_features=128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # input size: 16x16
        self.bn5 = nn.BatchNorm2d(num_features=256)

        self.dropout = nn.Dropout()
        self.full_conn1 = nn.Linear(in_features=256 * 8 * 8, out_features=512)  # input size: 8x8
        self.full_conn2 = nn.Linear(in_features=512, out_features=4)  # output size: 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # output size: 128x128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # output size: 64x64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # output size: 32x32
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # output size: 16x16
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # output size: 8x8

        x = torch.flatten(x, start_dim=1)  # flatten the tensor for fully connected layer
        x = F.relu(self.full_conn1(x))
        x = self.dropout(x)
        x = self.full_conn2(x)
        return x

# model registry and metadata config
MODEL_CONFIG = {
    "Angel Model": {
        "file_id": "17ajFrT4D4p--d4DVeFDK2pZXqCeeFUIm",
        "file_name": "brain_mri_cnn.pth",
        "model_name": SimpleCNN,
        "description": "Best used for general baseline feature extraction and balanced accuracy across all 4 tumor categories.",
    },
    "Sahil Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "description": "...",
    },
    "Yi Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "description": "...",
    },
    "Hamet Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "description": "...",
    },
    "Izzie Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "description": "...",
    },
    "Mike Model": {
        "file_id": "",
        "file_name": "",
        "model_name": None,
        "description": "...",
    }
}

# downloads a .pth file from Google Drive if not in directory; loads and returns trained model
@st.cache_resource
def load_model(file_id: str, file_name: str, model_name: nn.Module):
    # device allocation: CUDA -> MPS -> CPU
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # check if file in the current working directory
    if not os.path.exists(file_name):
        st.toast(f"Downloading {file_name} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)
    
    model = model_name()
    # load saved weights 
    model.load_state_dict(torch.load(file_name, map_location=device))
    model = model.to(device)
    model.eval() # puts model in evaluation mode (deactivates training behavior)
    return model, device

# transform for visual preprocessing 
visual_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1)
    ])

# transform pipeline for normalized tensor
def to_normalized_tensor(image):
    # transform exactly matches 'test_transform' from brain_mri_cnn_v5.ipynb file
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])   
    ])

    # transform and add batch dimension: [1, 256, 256] -> [1, 1, 256, 256]
    return transform(image).unsqueeze(0) 

# runs model inference; returns predictions
def run_inference(model, device, processed_image, classes):
    # turn image into normalized tensor
    input_tensor = to_normalized_tensor(processed_image).to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = classes[predicted_idx.item()]
        confidence_pct = confidence.item() * 100
        
    return predicted_class, confidence_pct, probabilities


# setup page UI
st.set_page_config(page_title="Brain MRI Tumor Classifier",layout="centered")

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

# get metadata for the chosen model
selected_config = MODEL_CONFIG[selected_model_name]

# main page content
st.title(f"Brain MRI Tumor Classifier")
st.header(f"**{selected_model_name}**")
st.write(f"**Model Overview:** {selected_config['description']}")
st.markdown("--")
st.write("Upload a brain MRI scan (JPG, JPEG, or PNG) below. The model will preprocess it, run inference, and predict the clinical diagnosis.")
st.bottom.info("⚠️ **Disclaimer:** This tool is an academic demonstration and is **NOT** intended for official medical diagnosis.")

# file uploader for image uploading
uploaded_file = st.file_uploader("Upload an MRI Scan...", type=["jpg", "jpeg", "png"])

# Classes matching your directory/labels map
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

if uploaded_file is not None:
    # render image visual transform
    original_image = Image.open(uploaded_file)
    processed_image = visual_transform(original_image)  
    
    st.write("##### Uploaded Scan (Processed)")
    st.image(processed_image, width=300)

    try:
        model, device = load_model(
            file_id=selected_config["file_id"],
            file_name=selected_config["file_name"],
            model_name=selected_config["model_name"]
        )

        st.toast("*Running image inference...*")
        pred, conf, probs = run_inference(model, device, processed_image, classes)
        
        if pred == "No Tumor":
            st.success(f"**Prediction: {pred}** (Confidence: {conf:.2f}%)")
        else:
            st.warning(f"**Tumor Detected: {pred}** (Confidence: {conf:.2f}%)")

        st.markdown("---")
        st.subheader("Diagnostic Results")
        st.write("##### Classification Confidence Breakdown:")
        for idx, name in enumerate(classes):
            prob = probs[idx].item() * 100
            st.progress(prob / 100, text=f"{name}: {prob:.2f}%")
                
    except Exception as e:
        st.error(f"Could not load selected model: {e}")