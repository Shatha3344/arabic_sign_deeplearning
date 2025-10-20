#  Arabic Sign Deep Learning: Hybrid Spatio-Temporal Recognition Model

## ArSL Recognition System: Hybrid Spatio-Temporal Model

**Author:** Shatha3344
**Repository:** `Shatha3344/arabic_sign_deeplearning`

[](https://opensource.org/licenses/MIT)

A research and application project focused on developing a Deep Learning system for **Dynamic Arabic Sign Language (ArSL) Recognition**. The core contribution is a **Hybrid Architecture** leveraging **Temporal Transformers** for sequence modeling.

-----

## 1\. Introduction and Task Definition

Sign language recognition is a challenging spatio-temporal sequence processing task. This project specifically addresses **ArSL Sign Classification** by processing streams of **Keypoint Sequences** extracted from video inputs.

-----

## 2\. Architecture and Methodology

### 2.1. Core Architecture and Training Pipeline

The primary components defining our model's structure and the methodology used for training are presented below:

[**Figure 1: Model Architecture**](asset/image/architecture.png) | [**Figure 2: Training Workflow**](asset/image/alogthims.jpg)

  * **Model Architecture:** Figure 1 illustrates the hybrid `hybird_v3_toptransfprm` structure, which integrates feature extraction with **Temporal Transformer layers** for sequential analysis.
  * **Training Workflow:** Figure 2 displays the comprehensive procedural diagram, outlining the end-to-end training process from data preprocessing to final evaluation.

### 2.2. Feature Extraction Pipeline

Raw video data is converted into normalized, sequential keypoint data suitable for the deep learning model:

[**Figure 3: Keypoint Extraction Pipeline**](https://www.google.com/search?q=pipe.png)

  * **Description:** Figure 3 demonstrates the transformation process from video frames to structured **Keypoint Coordinates** using the **MediaPipe** library.
  * **Artifacts:** **`scaler.joblib`** is used for coordinate normalization, ensuring scale and position invariance.

-----

## 3\. Benchmarking and Evaluation

### 3.1. Performance Metrics and Learning Curves

The model's performance and learning behavior are benchmarked using standard classification metrics:

[**Figure 4: Training Curves**](https://www.google.com/search?q=train.png) | [**Figure 5: Performance Metrics**](https://www.google.com/search?q=acc.png)

  * **Training Curves:** The curves in Figure 4 show **Stable Convergence** between training and validation accuracy over epochs, confirming effective learning without significant overfitting.
  * **Performance Metrics:** Figure 5 summarizes the **Classification Accuracy** and other metrics obtained on the held-out test set, validating the model's effectiveness.

### 3.2. Validation and Deployment

The project includes an interactive web interface for real-time testing and practical validation:

[**Figure 6: Deployment Platform**](https://www.google.com/search?q=screen_platform.jpg) | [**Figure 7: Practical Validation Results**](https://www.google.com/search?q=result.png)

  * **Deployment Platform:** Figure 6 showcases the frontend web application (developed using HTML/CSS/SASS), demonstrating the system's readiness for real-time inference.
  * **Practical Validation:** Figure 7 displays a successful live prediction output, confirming the model's ability to generalize to unseen gestures.

-----

## 4\. File Structure and Resources

The repository is organized to facilitate access to code, models, and frontend resources:

```
arabic_sign_deeplearning/
├── arabic_sign_deeplearning/ 
│   ├── models/
│   │   └── hybird_v3_toptransfprm/
│   │       ├── hybrid_model_final_108.keras # Trained Model File
│   │       ├── label_encoder.joblib         # Label Decoder
│   │       └── scaler.joblib                # Data Scaler
│   ├── web/                   # Web Application (HTML/CSS/SASS)
│   │   ├── index.html         
│   │   └── style.scss         
│   └── (Other Python/Scripts for Service Hosting)
├── notebooks/
│   └── arabic-words-sign-language-detection (2).ipynb # Training and Analysis Code
├── acc.png                    # Accuracy Metrics (Figure 5)
├── architecture.png           # Model Architecture (Figure 1)
├── alogthims.jpg              # Training Methodology (Figure 2)
├── pipe.png                   # Feature Extraction (Figure 3)
├── train.png                  # Training Curves (Figure 4)
└── README.md
```

-----

## 5\. Getting Started

Follow these steps to run the project and interact with the model:

### 1\. Clone the Repository

```bash
git clone https://github.com/Shatha3344/arabic_sign_deeplearning.git
cd arabic_sign_deeplearning
```

### 2\. Environment Setup

The project requires a Python environment (Conda is recommended) and core deep learning packages:

```bash
# Create a new environment (optional)
conda create -n arsign python=3.9
conda activate arsign

# Install required packages
pip install tensorflow keras scikit-learn joblib mediapipe 
```

### 3\. Load the Model

Ensure that all binary files (artifacts) are present in the path `arabic_sign_deeplearning/models/hybird_v3_toptransfprm/`.

### 4\. Run the Application

#### A. Web Interface (Frontend)

1.  Navigate to the folder: `cd arabic_sign_deeplearning/web`
2.  Open the **`index.html`** file directly in your browser.

#### B. Local Service (Backend)

1.  **To run training/analysis:** Open **`notebooks/arabic-words-sign-language-detection (2).ipynb`**.
2.  **To run the service:**
    ```bash
    python arabic_sign_deeplearning/main_app.py 
    ```
    (Replace `main_app.py` with your service execution file name).
