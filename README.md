# Crowd Management from Drone Footage (Computer Vision + CNN)

This project explores using **drone-based computer vision** to estimate
crowd density from aerial views. The goal is to support safer crowd
management by detecting zones that are too congested.

---

## 🔍 Problem

Large gatherings (events, rallies, festivals) can become dangerous when
crowd density gets too high in certain areas. Drones provide a top-down
view that can be used to analyse how packed the crowd is.

This project uses a **CNN-based classifier** to label frames as:

- `low`   – safe / sparse crowd  
- `medium` – normal / manageable crowd  
- `high`  – potentially unsafe density  

---

## 🧠 Approach

1. **Data Setup (Conceptual)**
   - Aerial or elevated images of crowds organised into folders:
     - `data/crowd/low/`
     - `data/crowd/medium/`
     - `data/crowd/high/`
   - Each folder contains example frames or images captured from a drone
     or similar elevated perspective.

2. **Preprocessing**
   - Resize images to a fixed resolution (e.g., 224×224).
   - Normalize pixel values to [0, 1].

3. **Model**
   - CNN classifier (either a small custom CNN or a transfer learning
     backbone like MobileNetV2).
   - Output layer: 3-class softmax (`low`, `medium`, `high`).

4. **Training & Evaluation**
   - Train on labeled images from each density category.
   - Evaluate using accuracy and classification report.

5. **Extension Ideas**
   - Divide frame into grids and estimate density per region.
   - Overlay heatmap for "risk zones".
   - Run inference on live video feed from a drone.

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- ImageDataGenerator (directory-based loading)
- NumPy
- scikit-learn (for metrics)

---

## 📁 Project Structure

```text
crowd-management-drone-vision/
│
├── data/
│   └── crowd/               # dataset root (placeholder)
│       ├── low/
│       ├── medium/
│       └── high/
│
├── src/
│   └── train_crowd_cnn.py   # CNN training script
│
└── README.md
