# Brain Tumor Detection Web Application

🌐 **Live Application:** [brain-tumor-detection-ai4all.streamlit.app](https://brain-tumor-detection-ai4all.streamlit.app/)

### Overview

This interactive Streamlit web application deploys custom Convolutional Neural Network (CNN) models designed to assist in brain MRI image classification. The application allows users to upload or drag-and-drop a brain MRI scan, inspect the visual preprocessing applied to the image, and receive real-time diagnostic predictions across four MRI classification categories:

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

The tool provides confidence score breakdowns for each classification category to give insight into model certainty.

### Core Features

* **Multi-Model Inference:** Allows users to switch between different custom CNN model architectures trained by individual team members.
* **Framework Flexibility:** Integrates both **PyTorch** and **TensorFlow/Keras** deep learning pipelines.
* **Preprocessing Visualization:** Displays original vs. preprocessed grayscale/channel transformations before running inference.
* **Diagnostic Breakdown:** Renders probability distribution bars for all four diagnosis classes.
