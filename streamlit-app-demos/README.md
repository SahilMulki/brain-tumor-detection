# DEMO Streamlit Web Applications

This folder contains interactive Streamlit web applications that deploy our custom CNN Models. The application allows clinical users to drag and drop a brain MRI scan, inspect the visual grayscale preprocessing, and view real-time tumor diagnostic results.  

This directory contains **Streamlit Python application files**, **trained model weights files**, and the **requirements text file** listing the necessary dependencies.

## Local Installation & Setup
To run this application locally on your machine, follow these steps:
### 1. Clone the Repository & Navigate to the Folder
```bash
git clone <your-repo-url>
cd <path-to-this-folder>
```

### 2. Install the Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run mri_classifier_app.py
```
