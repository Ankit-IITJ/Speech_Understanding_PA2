# Speech Enhancement & MFCC Analysis of Indian Languages

This repository contains the implementation and analysis for two tasks:  
1. **Speech Enhancement** for speaker verification and separation.  
2. **MFCC Feature Extraction** and language classification for Indian languages (Marathi, Punjabi, Hindi).  

---

## 📁 Project Structure
- **Speech Enhancement (Question-1.ipynb))**: Baseline model evaluation, fine-tuning with LoRA and ArcFace, multi-speaker dataset creation, and SepFormer-based separation.  
- **MFCC Analysis (Question-2.ipynb)**: Feature extraction, visualization, statistical analysis, and language classification using MFCCs.  

---

## 📋 Key Components

### 1. Speech Enhancement
- **Datasets**:  
  - **VoxCeleb1**: Primary evaluation for speaker verification.  
  - **VoxCeleb2**: Fine-tuning and multi-speaker separation.  
- **Models**:  
  - **HuBERT-Large**: Baseline speaker verification.  
  - **SepFormer-Whamr**: Speech separation and enhancement.  
- **Results**:  
  - **Baseline EER**: 27.27% | **TAR@1%FAR**: 0.222  
  - **SepFormer Metrics**: SDR (10.77–20.24 dB), SAR (10.77–20.24 dB), PESQ (3.35–4.31)  
  - **Rank-1 Accuracy**: 42.45% (fine-tuned model) → 60% (integrated pipeline).  

### 2. MFCC Analysis & Classification
- **Dataset**: [Kaggle’s 10 Indian Languages](https://www.kaggle.com/datasets/hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages).  
- **Features**: 13 MFCC coefficients extracted using `librosa`.  
- **Classifier**: Random Forest (accuracy: 95%).  
- **Visualizations**: MFCC spectrograms, mean/variance plots.  

---

## ⚙️ Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/Ankit-llTJ/Speech-Understanding-PA2.git
