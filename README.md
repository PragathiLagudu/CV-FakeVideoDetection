# Forensics-Aware Deepfake Detection using Multi-Stream Feature Fusion

Deepfake detection framework based on **XceptionNet** enhanced with **frequency-domain** and **edge-based forensic features** for improved manipulation detection on the **FaceForensics++** dataset. The project combines RGB, FFT spectra, and edge maps into a unified 9-channel representation for robust deepfake classification. 

---

##  Project Overview

Traditional RGB-only deepfake detectors often fail to capture subtle forensic traces introduced during face manipulation. This project introduces a **Forensics-Aware XceptionNet** that integrates:

* **RGB Features** → facial semantic information
* **FFT Log-Magnitude Spectra** → GAN frequency artifacts
* **Sobel + Canny Edge Maps** → blending seam inconsistencies

The model is evaluated on the **FaceForensics++ (FF++)** benchmark and demonstrates improved performance over the standard RGB-only baseline. 

---

##  Features

*  Multi-stream forensic feature extraction
*  9-channel modified XceptionNet
*  FFT-based frequency analysis
*  Sobel-X, Sobel-Y, and Canny edge extraction
*  Grad-CAM visualization support
*  Comparison with RGB-only baseline
*  Compression robustness evaluation
*  Per-manipulation-method analysis

---

##  Proposed Architecture

### Input Streams

| Stream      | Purpose                                        |
| ----------- | ---------------------------------------------- |
| RGB         | Standard visual semantics                      |
| FFT Spectra | Detect GAN-generated frequency artifacts       |
| Edge Maps   | Detect blending boundaries and discontinuities |

These streams are concatenated into a:

```text
9-channel input tensor
(RGB + FFT + Edge)
```

which is fed into a modified XceptionNet backbone. 

---

## 📂 Repository Structure

```text
├── CV_Project_Code.ipynb      # Main training & evaluation notebook
├── Report.pdf                 # Detailed project report
└── README.md                  # Project documentation
```

---

##  Dataset

### FaceForensics++

The project uses the **FaceForensics++** dataset containing:

* Real videos
* Deepfakes
* Face2Face
* FaceSwap
* NeuralTextures

### Dataset Details

* Compression level: `c40`
* 4000 face crops used
* 80/10/10 train-validation-test split
* Face extraction using **MTCNN**



---

##  Model Details

### Baseline Model

* XceptionNet
* RGB-only input
* Pretrained on ImageNet

### Proposed Model

* Modified XceptionNet
* 9-channel forensic input
* Warm-start initialization

Parameter overhead:

```text
+1728 parameters only (+0.01%)
```



---

## 🏋️ Training Configuration

| Hyperparameter | Value            |
| -------------- | ---------------- |
| Input Size     | 224 × 224        |
| Batch Size     | 16               |
| Optimizer      | Adam             |
| Learning Rate  | 1e-4             |
| Weight Decay   | 1e-5             |
| Scheduler      | Cosine Annealing |
| Epochs         | 10               |

Training performed on:

```text
NVIDIA Tesla T4 GPU (Google Colab)
```



---

##  Results

### Baseline vs Proposed Model

| Metric    | Baseline | Proposed |
| --------- | -------- | -------- |
| Accuracy  | 86.50%   | 88.75%   |
| AUC-ROC   | 0.9511   | 0.9623   |
| F1 Score  | 0.8650   | 0.8866   |
| Precision | 0.8522   | 0.8800   |
| Recall    | 0.8782   | 0.8934   |

The proposed method consistently improves performance across all evaluation metrics. 

---

##  Key Findings

* FFT features help expose GAN upsampling artifacts
* Edge features improve blending boundary detection
* NeuralTextures remains the hardest manipulation type
* Multi-stream fusion improves robustness under compression



---

## Additional Analysis

The project also includes:

* Grad-CAM visualizations
* Compression robustness analysis
* Per-manipulation evaluation
* Error Level Analysis (ELA)
* Facial landmark inconsistency analysis



---

##  How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install torch torchvision timm opencv-python numpy matplotlib scikit-learn facenet-pytorch
```

### 3. Open Notebook

Run the notebook:

```bash
CV_Project_Code.ipynb
```

using:

* Jupyter Notebook
* Google Colab

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* timm
* NumPy
* Matplotlib
* Scikit-learn
* FaceNet / MTCNN

---

##  Future Improvements

* Train on the full FF++ dataset
* Add temporal modeling using LSTMs/3D CNNs
* GPU-accelerated FFT computation
* Cross-dataset generalization testing
* Learnable forensic preprocessing modules



---

## 📄 Report

Detailed explanation, methodology, experiments, and results are available in:

```text
Report.pdf
```

---


## ⭐ Acknowledgements

* FaceForensics++ Dataset
* timm library maintainers
* Google Colab GPU resources

