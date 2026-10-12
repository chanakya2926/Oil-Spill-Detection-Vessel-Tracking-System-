# Multimodal Satellite Oil Spill Detection & Emergency Recovery System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end artificial intelligence and remote sensing system designed to detect, segment, and quantify marine oil spills from satellite imagery, followed by automated emergency vessel dispatch and cleanup tasking.

---

## 📌 Project Overview

Marine oil spills cause catastrophic damage to ocean ecosystems and marine economies. Rapid detection, thickness estimation, and response logistics are critical. 

This project implements a complete pipeline:
1. **Multisensor Satellite Ingestion**: Processes both **Sentinel-2 (Optical)** and **PALSAR (Synthetic Aperture Radar - SAR)** imagery.
2. **Dual-Branch Deep Learning (CNN + ViT)**: Combines a Convolutional Neural Network (CNN) branch for local spectral features with a Vision Transformer (ViT) branch for global spatial context and slick boundary continuity.
3. **Joint Edge-Preserving Loss**: Enhances boundary accuracy at tricky oil-water interfaces using combined BCE, Dice, and Laplacian edge filtering.
4. **Thickness & Volume Quantification**: Classifies slick thickness layers (Sheen, Rainbow, Slick, Emulsion) based on radiometric intensity and calculates total volume in cubic meters ($m^3$).
5. **Autonomous Emergency Tasking**: Integrates live **AIS (Automatic Identification System)** vessel telemetry to calculate real-time ETAs and assign the optimal responder ships via the Haversine distance formula.

---

## 🏗️ Architecture

```
Satellite Input (Optical / SAR)
             │
      ┌──────┴──────┐
      ▼             ▼
 Spectral CNN   Spatial ViT
  (Local Cues)   (Global Context)
      └──────┬──────┘
             ▼
      Feature Fusion Layer (1x1 Conv + Sigmoid)
             │
             ▼
      Predicted Binary Mask
             │
      ┌──────┴──────────────────────────┐
      ▼                                 ▼
Morphological Post-Processing     Radiometric Thickness Map
(Denoising & Hole Filling)       (Sheen, Rainbow, Slick, Emulsion)
      │                                 │
      ▼                                 ▼
 Final Segmentation Output         Estimated Volume (m³)
                                        │
                                        ▼
                             AIS Vessel Telemetry & ETA
                                        │
                                        ▼
                           Automated Dispatch Report
```

---

## 📊 Visual Results

### Segmentation & Boundary Overlay
![Segmentation Result](result.png)

### Confidence Heatmap Analysis
![AI Confidence Heatmap](heatmap_20839.png)

---

## 🚀 Key Features

- **Hybrid Model Architecture**: Custom PyTorch `DualBranchDetector` combining `SpectralCNNBranch` and `SpatialViTBranch`.
- **Multimodal Remote Sensing**: Robust evaluation on both SAR (all-weather radar) and optical satellite feeds.
- **Custom Loss Formulation**: Laplacian-weighted edge loss penalizing misclassifications at slick borders.
- **Volume Estimation Engine**: Maps pixel area ($10\text{m} \times 10\text{m}$ resolution) and thickness standards to compute physical spill volume.
- **Maritime Decision Support**: Automated responder filtering (tugs, pollution skimmers, patrol vessels) and route optimization based on speed-over-ground (SOG).

---

## 📂 Project Structure

```text
├── model.py                 # DualBranchDetector (Spectral CNN + Spatial ViT)
├── train.py                 # Model training loop with custom Edge Loss & Cosine Scheduler
├── data_loader.py           # Dataset loaders with augmentation (flips, rotations, normalization)
├── test_engine_final.py     # Evaluation metrics (mIoU, Pixel Accuracy, Volume calculations)
├── thickness.py             # Radiometric thickness mapping and volume estimation
├── vessel.py                # Live AIS API integration and vessel tasking system
├── vessel_tasking.py        # Offline vessel dispatch from AIS dataset
├── utils.py                 # Haversine distance calculations & severity classification
├── result.png               # Visual overlay and ground-truth comparison
└── heatmap_20839.png        # Confidence probability heatmap
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/oil-spill-detection-recovery.git
cd oil-spill-detection-recovery
```

### 2. Install dependencies
```bash
pip install torch torchvision opencv-python numpy pandas matplotlib requests
```

### 3. Run Inference & Testing
```bash
python test_engine_final.py
```

### 4. Run Thickness & Vessel Tasking
```bash
python thickness.py
```

---

## 📈 Evaluation Metrics

The system evaluates detection and segmentation using standard computer vision and operational metrics:
- **Mean Intersection over Union (mIoU)**
- **F1-Score / Dice Coefficient**
- **Pixel Accuracy & Precision / Recall**
- **Estimated Spill Volume ($m^3$)**
- **Vessel ETA (Hours)**

---

## 📜 License
This project is open-source under the MIT License.
