# 🦴 AI-Based Musculoskeletal Abnormality Detection Using Explainable Deep Learning

**An Explainable AI Approach for Musculoskeletal X-ray Abnormality Classification**

A deep learning research project that analyzes musculoskeletal X-ray images and estimates whether an image is **Normal** or **Abnormal**. The application combines image classification with Grad-CAM visual explanations, model evaluation metrics, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Musculoskeletal X-rays are commonly used to examine bones and joints for potential abnormalities. Interpreting radiographs requires expertise, and deep learning offers a way to investigate automated image classification and visual explanation techniques.

This project implements an AI-assisted research prototype using the **MURA dataset** and an **EfficientNet-based image classification model**. Users can upload X-ray images, view predicted abnormality probabilities, and inspect Grad-CAM heatmaps highlighting image regions that influenced the model's prediction.

The application also provides a model performance dashboard with evaluation metrics, confusion matrices, ROC curves, and model comparison functionality when the corresponding evaluation results are available.

> **Disclaimer:** This is an educational research prototype, not a clinically validated medical device. Predictions and heatmaps must not be used for diagnosis or treatment decisions.

**Application Screenshots**
### 1. X-ray Image Upload
<img width="1600" height="758" alt="image" src="https://github.com/user-attachments/assets/f3b44a44-0ae7-473e-8c6b-db70306e8752" />

### 2. Model Performance Dashboard
<img width="1600" height="762" alt="image" src="https://github.com/user-attachments/assets/f365c211-f6bf-40ca-8a04-49b23c1af813" />


### 3. Prediction and Grad-CAM Explanation
<img width="1600" height="759" alt="image" src="https://github.com/user-attachments/assets/16cf22ae-355b-4047-8ca3-8ca7285abe81" />



## 🎯 Objectives

- Develop a deep learning pipeline for musculoskeletal X-ray classification.
- Explore EfficientNet-based transfer learning for medical image analysis.
- Generate visual explanations using Gradient-weighted Class Activation Mapping (Grad-CAM).
- Evaluate model predictions using standard classification metrics.
- Build an interactive interface for image review and performance analysis.
- Demonstrate how explainability can support the interpretation of AI model outputs.

## ✨ Key Features

### 1. AI-Based X-ray Classification
- Upload PNG or JPG radiographs.
- Estimate the probability of an abnormal classification.
- Display the predicted label and confidence in the predicted class.
- Adjust the decision threshold interactively.

### 2. Explainable AI with Grad-CAM
- Generate heatmaps to visualize image regions that influenced the model.
- Overlay visual explanations on uploaded radiographs.
- Adjust heatmap opacity.
- Download prediction reports as PNG or PDF when supported by the application.

Grad-CAM provides an approximate visualization of model attention. It does not identify or confirm a medical lesion.

### 3. Model Performance Dashboard
The dashboard can display:
- Accuracy
- Precision
- Recall (sensitivity)
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- Performance by body part
- Training history
- Patient-bootstrap confidence intervals, when available

Evaluation results appear after the corresponding evaluation workflow has been executed successfully.

### 4. Baseline CNN vs. EfficientNet
The project includes workflows for comparing a baseline CNN with EfficientNet. Comparison results depend on the availability of trained models and generated evaluation outputs.

### 5. Interactive Streamlit Interface
- Light and dark appearance options.
- Configurable classification threshold.
- Configurable Grad-CAM heatmap opacity.
- Model information and preprocessing details.
- Image review and explainability visualizations.
- Separate Prediction, Model Performance, and About & Limitations sections.

## 🧠 Models and Methodology

### EfficientNetB0

EfficientNetB0 is used as the transfer-learning architecture for image classification. The training workflow supports ImageNet weights and configurable training parameters.

### Baseline CNN

A baseline convolutional neural network provides a reference architecture for comparison with EfficientNet.

### Grad-CAM

Gradient-weighted Class Activation Mapping generates a heatmap using gradients associated with the model's output. It helps visualize which image regions contribute to a prediction.

### Workflow

1. Load the MURA image manifests and dataset.
2. Prepare image paths, labels, and data splits.
3. Preprocess and resize X-ray images.
4. Train the selected classification model.
5. Generate predictions for uploaded images.
6. Generate Grad-CAM visual explanations.
7. Evaluate model predictions using classification metrics.
8. Display predictions, visualizations, and evaluation results in Streamlit.

## 🗂️ Dataset

This project uses **MURA (Musculoskeletal Radiographs)**, a public dataset of musculoskeletal radiographs.

The dataset must be downloaded separately and placed in the expected local directory.

Expected structure:

```text
data/
└── MURA-v1.1/
    ├── train/
    ├── valid/
    ├── train_image_paths.csv
    ├── valid_image_paths.csv
    ├── train_labeled_studies.csv
    └── valid_labeled_studies.csv
```

The labeled-study CSV files are expected by some versions of the standard MURA dataset layout. Use the files required by the actual project configuration and data-loading implementation.

**Dataset access:** https://stanfordmlgroup.github.io/competitions/mura/

The dataset and its images are not included in this repository. Follow the dataset provider's access conditions and terms.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow / Keras |
| Model architecture | EfficientNetB0 and baseline CNN |
| Explainability | Grad-CAM |
| Data processing | NumPy, Pandas |
| Evaluation | scikit-learn |
| Visualization | Matplotlib and Plotly, where used |
| Web application | Streamlit |
| Image processing | OpenCV and Pillow |

## 📁 Project Structure

```text
mura_msk/
├── app/
│   ├── streamlit_app.py
│   └── styles.py
├── data/
│   └── MURA-v1.1/          # Download separately
├── notebooks/
│   └── colab_quickstart.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── make_dummy_data.py
│   └── smoke_test.sh
├── src/
│   ├── config.py
│   ├── data.py
│   ├── manifest.py
│   ├── models.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── compare.py
│   └── insights.py
├── outputs/                # Generated locally; not committed
├── requirements.txt
├── .gitignore
└── README.md
```

*The exact filenames may differ slightly depending on the current repository version.*

## ⚙️ Installation and Setup

### Prerequisites

- Python version compatible with the project's TensorFlow requirements.
- Git.
- A Windows, Linux, or macOS environment.
- The MURA dataset for training and evaluation.
- Sufficient memory and storage for image processing and model training.

### 1. Clone the repository

```bash
git clone https://github.com/Vidit72wanjari/AI-Based-Musculoskeletal-Abnormality-Detection-Using-Explainable-Deep-Learning.git
cd AI-Based-Musculoskeletal-Abnormality-Detection-Using-Explainable-Deep-Learning
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure the dataset

Download and extract the MURA dataset into `data/MURA-v1.1/`. Verify that the image folders and CSV manifests match the structure expected by `src/config.py` and `src/data.py`.

## 🚀 Training

Inspect the available options first:

```bash
python -m src.train --help
```

Example EfficientNet training command:

```bash
python -m src.train --model efficientnet --run_name efficientnet --epochs 5 --head_epochs 2 --batch_size 16 --weights imagenet
```

This example trains using the configured dataset and is not restricted to 15 images. Training time depends on the dataset size, hardware, and configuration.

To inspect baseline CNN options:

```bash
python -m src.train --help
```

Use the model argument supported by your implementation to train the baseline CNN.

## 📊 Evaluation

After a trained model has been saved, evaluate it using:

```bash
python -m src.evaluate --run_name efficientnet
```

The evaluation workflow generates metrics and visualizations in the corresponding output directory.

If your local version supports the `--max_images` option, a small demonstration can be run with:

```bash
python -m src.evaluate --run_name efficientnet --max_images 15 --n_boot 50
```

This evaluates up to 15 images from each evaluated split. Small samples are useful for demonstrating the pipeline but do not provide statistically reliable model performance estimates.

Do not compare demo metrics with full-dataset research results as if they were equivalent.

## 🖥️ Run the Streamlit Application

Make sure a trained model and its matching configuration file exist in the expected output directory.

For Windows PowerShell:

```powershell
$env:MURA_OUTPUT = "$PWD\outputs"
python -m streamlit run app/streamlit_app.py
```

Open the local URL shown in the terminal, normally:

```text
http://localhost:8501
```

### Expected model files

```text
outputs/
└── efficientnet/
    ├── best_model.keras
    ├── config.json
    ├── metrics.json
    ├── confusion_matrix.png
    └── roc_curve.png
```

Some files are created only after training or evaluation. Do not create empty model files manually.

## 📈 Evaluation Metrics

The project uses standard binary classification metrics:

- **Accuracy:** Proportion of correctly classified samples.
- **Precision:** Proportion of predicted abnormal samples that are actually abnormal.
- **Recall:** Proportion of actual abnormal samples correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks positive examples above negative examples.
- **Confusion matrix:** Summarizes true positives, true negatives, false positives, and false negatives.

Metric values depend on the trained model, selected dataset split, decision threshold, and evaluation sample. No generalization or clinical performance level is claimed here.

## 🔬 Current Demonstration

The current interface demonstrates:

- Uploading an X-ray image.
- Displaying a Normal or Abnormal model prediction.
- Showing the predicted abnormality probability.
- Generating a Grad-CAM visual explanation.
- Displaying evaluation metrics and a confusion matrix.
- Visualizing the ROC curve.
- Comparing model results when the required outputs are available.

The current small-sample evaluation is a software demonstration, not evidence of clinical effectiveness.

## ⚠️ Limitations

- The model has not been clinically validated.
- A predicted label is not a medical diagnosis.
- Grad-CAM visualizations are qualitative and may not correspond to actual lesions.
- Small evaluation samples can produce unstable or undefined metrics.
- Performance may vary across anatomical regions, imaging equipment, and patient populations.
- Model probabilities may be uncalibrated.
- Dataset quality, label noise, and distribution shift can affect predictions.

## 🔮 Future Improvements

- Train and evaluate on appropriately sized patient-independent datasets.
- Improve robustness and generalization across anatomical regions.
- Add model calibration and uncertainty estimation.
- Investigate class imbalance and data augmentation.
- Improve explainability evaluation and visualization.
- Add systematic error analysis and reproducible experiment tracking.
- Conduct independent validation before considering any clinical application.

## 👨‍💻 Author

**Ishika Paunikar**

GitHub: https://github.com/ishikaPaunikar

## 📜 Disclaimer

This project is intended for educational and research purposes only. It is not a substitute for professional medical interpretation, diagnosis, or treatment. Never use its predictions to make medical decisions.

## ⭐ Acknowledgements

Thanks to the Stanford ML Group for making the MURA dataset available to the research community.
