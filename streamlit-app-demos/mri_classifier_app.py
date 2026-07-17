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
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # input size: 128x128
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # input size: 64x64
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # input size: 32x32
        self.bn4 = nn.BatchNorm2d(num_features=128)

        self.dropout = nn.Dropout()
        self.full_conn1 = nn.Linear(in_features=128 * 16 * 16, out_features=512)  # input size: 16x16
        self.full_conn2 = nn.Linear(in_features=512, out_features=4)  # output size: 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # output size: 128x128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # output size: 64x64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # output size: 32x32
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # output size: 16x16

        x = torch.flatten(x, start_dim=1)  # flatten the tensor for fully connected layer
        x = F.relu(self.full_conn1(x))
        x = self.dropout(x)
        x = self.full_conn2(x)
        return x


# load model weights with caching for optimization
@st.cache_resource
def load_model():
    # device allocation: CUDA -> MPS -> CPU
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = SimpleCNN()
    # load saved weights from brain_mri_cnn_v5.pth file
    model.load_state_dict(torch.load('brain_mri_cnn_v5.pth', map_location=device))
    model = model.to(device)
    model.eval() # puts model in evaluation mode (deactivates training behavior)
    return model, device

# initialize model
try:
    model, device = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model weights: {e}. Please ensure correct file is in this folder.")


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
    
    tensor = transform(image)
    # add batch dimension: [1, 256, 256] -> [1, 1, 256, 256]
    tensor = tensor.unsqueeze(0) 
    return tensor


# set page config for web app interface
st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    layout="centered"
    )

# setup user interface
st.title("Brain MRI Tumor Classifier")
st.write("Upload a brain MRI scan (JPG, JPEG, or PNG) below. The model will preprocess it to grayscale, run inference, and predict the clinical diagnosis.")

# file uploader for image uploading
uploaded_file = st.file_uploader("Upload an MRI Scan...", type=["jpg", "jpeg", "png"])

# Classes matching your directory/labels map
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

if uploaded_file is not None and model_loaded:
    # open and render image
    original_image = Image.open(uploaded_file)

    processed_image = visual_transform(original_image)
    st.write("##### Uploaded Scan (Preprocessed)")
    st.image(processed_image, width=300)
    
    st.write("*Running image inference...*")
    
    # turn image into normalized tensor
    input_tensor = to_normalized_tensor(processed_image).to(device)
    
    # model predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = classes[predicted_idx.item()]
        confidence_pct = confidence.item() * 100

    # present diagnostic results
    st.markdown("---")
    st.subheader("Diagnostic Results")
    
    if predicted_class == "No Tumor":
        st.success(f"**Prediction: {predicted_class}** (Confidence: {confidence_pct:.2f}%)")
    else:
        st.warning(f"**Tumor Detected: {predicted_class}** (Confidence: {confidence_pct:.2f}%)")
        
    st.write("##### Classification Confidence Breakdown:")
    for idx, name in enumerate(classes):
        prob = probabilities[idx].item() * 100
        st.progress(prob / 100, text=f"{name}: {prob:.2f}%")