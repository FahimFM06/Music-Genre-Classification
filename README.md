# 🎵 Music Genre Classification using CRNN + BiGRU (PyTorch)

This project implements a **Music Genre Classification system** using a deep learning model based on  
**CRNN (Convolutional Recurrent Neural Network) with Bi-directional GRU**, built in **PyTorch** and deployed using **Streamlit**.

The system takes an audio file as input and predicts its music genre with confidence scores.

---

## 📌 Features

- 🎧 Supports audio formats: **WAV, MP3, OGG, FLAC, M4A**
- 🎼 Automatic audio preprocessing (resampling, segmentation, mel-spectrogram)
- 🧠 Deep Learning model: **CRNN + BiGRU**
- 📊 Segment-wise prediction and final aggregated prediction
- 🌐 Interactive **Streamlit Web App**
- ⚡ Works on **CPU and GPU**
- 🧩 Easy to extend with new genres or models

---

## 🏗️ Project Structure

Music-Genre-Classification/
│
├── app.py # Streamlit web application
├── model.py # CRNN + BiGRU model architecture
├── audio_utils.py # Audio preprocessing & feature extraction
├── best_crnn_bigru_pytorch.pt # Trained PyTorch model checkpoint
├── Code_file.ipynb # Training notebook
├── requirements.txt # Required Python libraries
└── README.md # Project documentation

yaml
Copy code

---

## 🧠 Model Architecture

The model combines:

### 🔹 CNN (Convolutional Neural Network)
- Extracts **time-frequency features** from log-mel spectrograms
- Uses multiple convolution + batch normalization + pooling layers

### 🔹 BiGRU (Bidirectional GRU)
- Captures **temporal dependencies** in music
- Processes information from both past and future frames

### 🔹 Fully Connected Layers
- Perform final genre classification

---

## 🎛️ Audio Preprocessing Pipeline

1. Audio is resampled to **22,050 Hz**
2. Converted to **mono**
3. Standardized to **30 seconds**
4. Split into **10 segments** of **3 seconds**
5. For each segment:
   - Log-Mel Spectrogram extraction  
     - `n_mels = 128`
     - `n_fft = 2048`
     - `hop_length = 512`
   - Normalization (zero mean, unit variance)
6. Model predicts each segment
7. Final prediction is obtained by **aggregating segment probabilities**

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/FahimFM06/Music-Genre-Classification.git
cd Music-Genre-Classification
2️⃣ Create a Virtual Environment (Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Streamlit App
bash
Copy code
streamlit run app.py
The app will open automatically in your browser.

🖥️ Streamlit Web App Overview
Upload an audio file

View:

Device type (CPU/GPU)

Input spectrogram size

Number of segments

Listen to uploaded audio

See:

Final predicted genre

Confidence score

Top-K genre probabilities

Segment-wise predictions

Waveform & mel-spectrogram visualizations

🎯 Supported Genres
The model is trained on 10 music genres:

perl
Copy code
blues
classical
country
disco
hiphop
jazz
metal
pop
reggae
rock
⚠️ Important
The genre order must match the order used during training.
If needed, create a classes.txt file (one genre per line) to explicitly define label order.

📦 Model Checkpoint
File: best_crnn_bigru_pytorch.pt

Framework: PyTorch

Architecture: CRNN + BiGRU

The model definition in model.py must exactly match the training architecture.

🛠️ Common Issues & Fixes
❌ State Dict Loading Error
If you see errors like:

java
Copy code
Missing key(s) / Unexpected key(s) / Size mismatch
✅ Solution
Ensure model.py exactly matches the architecture used in Code_file.ipynb

Clear Streamlit cache:

bash
Copy code
streamlit cache clear
📚 Technologies Used
Python

PyTorch

Streamlit

Librosa

NumPy

Matplotlib

Pandas
