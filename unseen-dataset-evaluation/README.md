# External Validation & Generalization Testing

This folder contains the scripts and documentation for evaluating our trained Brain MRI classification models on completely unseen, out-of-distribution (external) datasets. 

## Evaluated Datasets

We performed external validation using the following two distinct, publicly available datasets:

### 1. Dataset 1: BRISC 2025 Dataset
* **Source:** [Kaggle - BRISC 2025 Dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025/)
* **Description:** A modern, curated brain image dataset used to test the model's adaptability to varying image contrasts and scan resolutions.
* **Usage:** Evaluated using the identical pre-trained weights to measure classification accuracy across the four core categories (*glioma, meningioma, pituitary, and no tumor*).

### 2. Dataset 2: Brain Tumor MRI Dataset for Deep Learning
* **Source:** [Kaggle - Brain Tumor MRI Dataset for Deep Learning](https://www.kaggle.com/datasets/alamshihab075/brain-tumor-mri-dataset-for-deep-learning)
* **Description:** An extensive collection of brain MRI scans presenting highly diverse tumor shapes, sizes, and orientations.
* **Usage:** Used as our second benchmark to confirm that our model's high performance is globally applicable and not restricted to our original training distribution.

## Key Metrics Tracked
For each external dataset, we do not rely solely on accuracy. We generate and analyze:
* **Precision:** To monitor and minimize False Positives.
* **Recall (Sensitivity):** To ensure we minimize False Negatives (missing a tumor).
* **F1-Score:** The harmonic mean to evaluate overall balance on the external data.
* **Accuracy:** Measures the overall percentage of correct predictions (both positive and negative cases) out of all evaluated scans.
* **Confusion Matrix Heatmaps:** Visually mapping exactly which classes are hardest for the model to generalize across different sources.
