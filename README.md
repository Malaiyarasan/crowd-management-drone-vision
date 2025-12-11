# Crowd Management from Drone Footage (Computer Vision + CNN)

This project uses **drone-based computer vision** to estimate crowd density from aerial images.  
The goal is to support **safer crowd management** by detecting areas where people are densely packed.

---

## 🔍 Problem

Crowd-related accidents often happen when a specific zone becomes **overcrowded** before anyone notices.  
Drones provide a top-down view, enabling early detection of:

- `low` – sparse / safe crowd  
- `medium` – moderate / manageable density  
- `high` – crowded / potentially unsafe  

This project builds a lightweight CNN to classify crowd density and generate a **heatmap** to highlight high-risk areas.

---

## 🧠 Approach

### 1. Data Setup (Demo)
A synthetic dataset is generated to simulate crowd levels:

data/crowd_demo/
├── low/
├── medium/
└── high/

Each folder contains artificially created crowd images.

### 2. Preprocessing
- Images resized to 224×224  
- Normalized to [0, 1]  
- Augmentation using `ImageDataGenerator`  

### 3. Model
A custom **CNN classifier** built using TensorFlow/Keras:

- Convolution blocks  
- MaxPooling layers  
- Dense layers  
- Softmax output for 3 classes  

### 4. Training
- 10 epochs  
- Categorical cross-entropy  
- Accuracy monitored on validation split  

### 5. Extensions Included
✔ Sliding-window **density heatmap**  
✔ Region-based risk scoring  
✔ Gradio demo for real-time testing

---

## 🧰 Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Gradio (interactive web UI)  
- NumPy  

---

## 📁 Project Structure

```text
crowd-management-drone-vision/
│
├── notebooks/
│   └── crowd_management_.ipynb     # Full training + demo notebook
│
├── model/
│   └── crowd_cnn_demo.h5           # Trained CNN model
│
├── data/
│   └── crowd_demo/                 # Synthetic sample dataset
│
├── src/                            # (optional scripts)
│
└── README.md
