# 🛩️ Crowd Management from Drone Footage (Computer Vision + CNN)

This project explores using **drone-based computer vision** to estimate
crowd density from aerial views. The goal is to support safer crowd
management by detecting zones that are too congested or unsafe.

---

## 🔍 Problem

Large gatherings (events, rallies, festivals) can become dangerous when
crowd density gets too high in certain areas.

Drones provide a **high-angle top-down view**, which is perfect for analysing crowd distribution.

This project uses a **CNN-based classifier** to label each frame as:

- **low** → safe / sparse crowd  
- **medium** → normal / manageable density  
- **high** → potentially unsafe / overcrowded  

---

## 🧠 Approach

### 1️⃣ Data Setup (Conceptual)
Images are organized as:

data/crowd/
├── low/
├── medium/
└── high/

Each folder contains labeled aerial crowd images.

---

### 2️⃣ Preprocessing

- Resize images (224 × 224)  
- Normalize pixel values  
- Batch loading using `ImageDataGenerator`

---

### 3️⃣ Model Architecture

A lightweight **Convolutional Neural Network (CNN)** designed to classify crowd levels.

Layers include:

- Conv2D  
- MaxPooling  
- Flatten  
- Dense classifier with softmax output  

---

### 4️⃣ Training & Evaluation

The model is trained on the 3 crowd categories and evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  

---

### 5️⃣ Extensions

You can expand this into:

- Drone live-feed real-time detection  
- Heatmap overlay per region  
- Warning system for unsafe zones  
- Crowd-flow movement tracking  
- Integration with government/public safety systems  

---

## 🧰 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- scikit-learn  
- Google Colab  
- Gradio (demo interface)

---

## 📁 Project Structure

crowd-management-drone-vision/
│
├── data/
│ └── crowd/ # dataset root (placeholder)
│
├── src/
│ └── train_crowd_cnn.py # CNN model training script
│
├── model/
│ └── crowd_cnn_demo.h5 # trained model (demo version)
│
└── notebooks/
└── crowd_management_.ipynb # full Colab notebook

---

## 🔗 Google Colab Notebook (Training + Full Code)

Click below to view/run the full project in Colab:

👉 **Open in Google Colab**  
https://colab.research.google.com/github/Malaiyarasan/crowd-management-drone-vision/blob/main/notebooks/crowd_management_.ipynb

This notebook includes:

- Dataset preparation  
- CNN training  
- Evaluation  
- Sample predictions  
- Exporting model `.h5`  
- Code for real-world drone video usage  

---

## 🚀 Live Demo (Gradio)

Try the model in your browser:

🔗 **Live Demo:**  
https://cd67d1e1d508e42c76.gradio.live

Upload an image → model predicts **Low / Medium / High** crowd density.

---

## 📦 Model Download

Download the trained demo model here:

🔗 `crowd_cnn_demo.h5`  
(Already included in `/model/` folder)

---

## 👤 Author

**Malaiyarasan M**  
AI Engineer – Applied Robotics & Computer Vision  

---

## ⭐ If this helped, give the repo a star! ⭐

