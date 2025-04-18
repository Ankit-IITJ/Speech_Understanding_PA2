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
   git clone https://github.com/microsoft/UniSpeech.git
   git clone https://github.com/Ankit-llTJ/Speech-Understanding-PA2.git
2. Install dependencies:
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa -y
   sudo apt-get update -y
   sudo apt-get install python3.10 python3.10-dev python3.10-venv -y
   python3.10 -m venv venv
   source venv/bin/activate
   pip install notebook librosa scikit-learn speechbrain pandas matplotlib s3prl fairseq tqdm soundfile ipykernel tensorboardX ffmpeg torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

## 🚀 Usage

### 1. Speech Enhancement

#### Speaker Verification
```bash
# Compare two audio files using HuBERT-Large
python verification.py \
  --model_name Hubert_large \
  --wav1 /path/to/audio1.wav \
  --wav2 /path/to/audio2.wav \
  --checkpoint /path/to/HuBERT_Large_SV_fixed.th
```
#### Run Question-1.ipynb for below functionalities
- Fine tuning speaker verification model using LoRA (Low-Rank Adaptation) and ArcFace loss
- Create a multi-speaker scenario dataset by mixing/overlapping utterances from 2 different speakers of the VoxCeleb2 dataset 
- Use the pre-trained SepFormer model (speechbrain/sepformer-whamr) to perform speaker separation
- Audio enhamcement using sepformer model (speechbrain/sepformer-wham-enhancement)
- Speaker identification using pre-trained and finetuned speaker identification HuBert-large model on enhanced audio
- A novel pipeline/algorithmic approach to combine the speaker identification model along with the SepFromer model to perform speaker separation with the speaker identification model and speech enhancement with the SepFormer model

### 2. MFCC Analysis & Classification

#### Run Question-2.ipynb for belwo functionalities
- Extract MFCC Features and visualize MFCC spectrograms for a representative set of samples from at least 3 languages (Marathi,Punjabi,Hindi) of your choice.
- Compare the MFCC spectrograms across the different languages
- Perform a statistical analysis (e.g., compute the mean and variance of MFCC coefficients)
- Train Language Classifier (Random forest classifier)
- Classify Audio and show confusion matrix

## 📊 Results Highlights

### 🎙️ Speech Enhancement
- **✅ Improvement**:  
  - Rank-1 accuracy **increased by 17.55%** (from 42.45% → 60%) using the integrated pipeline combining separation and identification.  
- **⚠️ Challenges**:  
  - Limited gains from LoRA fine-tuning due to **1-epoch training constraint** (no change in EER/TAR metrics).  

### 🗣️ Language Classification
- **✅ Performance**:  
  - **95% overall accuracy** using MFCC features and a Random Forest classifier.  
- **🔍 Observations**:  
  - Minor misclassifications between **Marathi ↔ Hindi** (see confusion matrix below), likely due to overlapping phonetic features.  

## 📝 Future Work

- **Improve Fine-Tuning**:  
  Increase LoRA rank or training epochs to enhance model adaptation capabilities.

- **Integrated Training**:  
  Incorporate speaker identification loss (e.g., ArcFace) directly into speech separation training for better alignment of objectives.

- **Robust MFCC Classification**:  
  Develop strategies to mitigate the impact of regional accents and background noise on MFCC-based language classification.

---

## 🔗 References

- [Librosa Documentation](https://librosa.org/doc/latest/)  
- [SepFormer Models](https://huggingface.co/speechbrain/sepformer-whamr)  
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
