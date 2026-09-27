# Enhancing Cybersecurity with Machine Learning: Leveraging UNSW-NB15 Dataset

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4A90E2?logo=fastapi&logoColor=white)](https://lightgbm.readthedocs.io/)

A machine learning-based Network Intrusion Detection System (NIDS) that classifies and detects network traffic anomalies and cyber attacks using the benchmark **UNSW-NB15** dataset.

---

## 📌 Project Overview

Network attacks are growing increasingly sophisticated, rendering traditional signature-based intrusion detection systems insufficient. This project explores and benchmarks multiple machine learning algorithms for effective, multi-class cyber attack detection and categorization.

### Authors & Academic Details
- **Authors**: Patel Devarshi Alkeshkumar, Chauhan Yash Chetankumar
- **Degree**: B.Sc. IT (Cyber Security), Semester-VI
- **Institution**: Department of Computer Science, Ganpat University
- **Project Coordinator / Guide**: Dr. Kashyap C. Patel

---

## 🎯 Target Attack Categories

The system performs multi-class classification across 10 network flow categories:
1. **Normal**: Standard benign network activity
2. **Generic**: Attacks against cryptographic primitives or generic exploits
3. **Exploits**: Known vulnerabilities targeted through targeted payloads
4. **Fuzzers**: Flooding software with randomized data to discover buffer overflows
5. **DoS**: Denial of Service attacks aimed at exhausting host resources
6. **Reconnaissance**: Probing and port scans to gather network architecture data
7. **Analysis**: Web attacks including port scans, spam, and HTML script intrusions
8. **Backdoor**: Stealthy persistent access methods bypassing security controls
9. **Shellcode**: Exploitation injecting shell command code into host memory
10. **Worms**: Self-replicating malware propagating through network channels

---

## 🔬 Evaluated Machine Learning Models

We systematically benchmarked five prominent algorithms:
- **Random Forest Classifier**: Robust ensemble of decision trees handling non-linear feature interactions and mitigating overfitting.
- **LightGBM Classifier**: Gradient boosting framework optimized for speed and high-dimensional network flow data.
- **Decision Tree Classifier**: Interpretable hierarchical decision rules for low-latency packet classification.
- **K-Nearest Neighbors (KNN)**: Non-parametric distance-based classification for localized anomaly clustering.
- **Logistic Regression**: Linear baseline probabilistic model.

---

## 📂 Project Structure

```text
CyberProject/
├── src/                               # Production source code
│   ├── train_and_evaluate.py          # Unified model training & evaluation pipeline
│   └── predict.py                     # Real-time inference script for flow records
├── data/                              # Dataset directory
│   ├── UNSW_NB15_training-set.csv     # Training partition (82,332 records)
│   ├── UNSW_NB15_testing-set.csv      # Testing partition (175,341 records)
│   ├── NUSW-NB15_features.csv         # Feature dictionary and metadata
│   ├── UNSW-NB15_LIST_EVENTS.csv      # Attack event metadata
│   ├── train_test_network.csv         # Network flow records
│   └── README.md                      # Dataset documentation & source download links
├── results/
│   └── figures/                       # High-resolution evaluation plots
│       ├── Accuracy*.png              # Accuracy bar charts per model
│       ├── Classification*.png        # Precision, recall, and F1 heatmaps
│       ├── ConfusionMatrix*.png       # Confusion matrix heatmaps per model
│       ├── Model_Accuracy_Comparison.png
│       └── metrics_summary.csv        # Tabulated metric comparisons
├── docs/                              # Project documentation and academic reports
│   ├── CyberSecurityFinalReport.pdf   # Complete published project report
│   ├── GG Final Report (1).docx       # Project documentation source
│   └── Main Page Report.docx          # Academic submission details
├── experiments/                       # Original exploratory research scripts (cyber1.py to cyber11.py)
├── projects/                          # Additional related research projects
│   ├── ipo_prediction/                # IPO Subscription rate prediction (Random Forest & Ensembles)
│   └── sign_language_detection/       # Computer vision sign language detector using YOLOv8
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusion rules for large datasets and binaries
└── README.md                          # Repository documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Installation
Clone this repository and install the dependencies:
```bash
git clone <repository_url>
cd CyberProject
pip install -r requirements.txt
```

---

## 💻 Usage

### Train and Evaluate Models
To train all benchmark models and generate performance evaluation plots in `results/figures/`:
```bash
python src/train_and_evaluate.py
```

To run a specific model (e.g., Random Forest and LightGBM):
```bash
python src/train_and_evaluate.py --models RandomForest LightGBM
```

### Run Model Inference
To evaluate sample network traffic records and predict potential intrusion vectors:
```bash
python src/predict.py --model-path models/DecisionTree_model.pkl
```

Or pass a custom CSV file of captured flow records:
```bash
python src/predict.py --model-path models/RandomForest_model.pkl --input-csv path/to/flow_records.csv
```

---

## 📊 Evaluation Visualizations

Evaluation heatmaps and performance charts are stored in [`results/figures/`](results/figures/):
- **Confusion Matrices**: Detailed true positive vs. false positive breakdowns per attack class.
- **Classification Reports**: Precision, Recall, and F1-Score breakdown for rare and dominant attack types.
- **Model Accuracy Comparisons**: Head-to-head performance across all algorithms.

---

## 📁 Additional Sub-Projects

This repository also contains two research sub-projects in the [`projects/`](projects/) directory:
1. **IPO Subscription Prediction** ([`projects/ipo_prediction/`](projects/ipo_prediction/)): Predicts investor subscription metrics using financial indicators and Random Forest Regressors.
2. **Sign Language Recognition** ([`projects/sign_language_detection/`](projects/sign_language_detection/)): Real-time webcam sign language alphabet detection using YOLOv8 computer vision models.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
