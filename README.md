# Fridge Spoilage Detection System

A comprehensive multi-approach machine learning system for detecting food spoilage in refrigerator images using object detection, supervised classification, and unsupervised autoencoder methods.

## 🎯 Project Overview

This project implements three different methodologies to detect and classify food spoilage:

1. **Object Detection (YOLOv8)** - Identifies and locates food items in fridge images
2. **Supervised Classification (ResNet18)** - Binary classification of fresh vs spoiled food
3. **Unsupervised Autoencoder** - Detects spoilage through reconstruction error analysis

## 📁 Project Structure

### **Core Application Files**
- **`app/main.py`** - FastAPI application with ResNet18 classifier endpoint (`/predict`)
- **`app/app.py`** - FastAPI application with autoencoder spoilage detection endpoint (`/check_spoilage`)
- **`app/infer_spoilage.py`** - Autoencoder inference module for spoilage detection

### **Datasets**
- **`data/detection_dataset/`** - Roboflow "Fridge objects" dataset (YOLOv8 format)
  - **Classes**: 10 food items (avocado, bacon, butter, cheese, eggs, lemon, milk, pepper, tomatoes, yogurt)
  - **Size**: 2,425 images total (2,184 train, 125 valid, 106 test)
  - **Format**: YOLOv8 with bounding box annotations
  - **Augmentation**: Horizontal/vertical flips, random crops, rotations
- **`data/train/`** - Training images for supervised classification (~6,490 images)
- **`data/valid/`** - Validation images for supervised classification (~1,850 images)
- **`data/test/`** - Test images for supervised classification (~946 images)

### **Trained Models**
- **`models/classifier.pt`** - ResNet18 model weights for fresh/spoiled classification
- **`models/classifier_traced.pt`** - Traced PyTorch model for production deployment
- **`models/autoencoder.pt`** - Autoencoder model for unsupervised spoilage detection
- **`yolov8n.pt`** - Pre-trained YOLOv8 nano model

### **Training Scripts**

#### **Supervised Learning**
- **`scripts/supervised/train_classifier.py`** - ResNet18 binary classification training
- **`scripts/supervised/evaluate_classifier.py`** - Model evaluation and metrics
- **`scripts/supervised/export_classifier.py`** - Model export for deployment

#### **Unsupervised Learning**
- **`scripts/unsupervised/train_autoencoder.py`** - Autoencoder training on fresh food images
- **`scripts/unsupervised/evaluate_autoencoder.py`** - Autoencoder evaluation
- **`scripts/unsupervised/export_model.py`** - Model export utilities

#### **Object Detection**
- **`scripts/detection/run_yolo8v.py`** - YOLOv8 inference and cropping utilities
- **`scripts/detection/run_ocr.py`** - OCR processing for text extraction
- **`scripts/detection/infer_all.py`** - Batch inference pipeline

### **Training Results & Outputs**
- **`runs/detect/`** - YOLOv8 training runs and results
  - **`baseline_yolov8n12/`** - Initial baseline training (low performance)
  - **`train6/`** - Best performing YOLOv8 model (~92% mAP50)
  - **`yolov8m_augmented_visdrone/`** - Augmented training with VisDrone dataset
- **`outputs/tsne/`** - t-SNE visualization plots for autoencoder features

### **Testing & Evaluation**
- **`tests/test_inference.py`** - Unit tests for inference functionality

## 🚀 Features

### **Multi-Modal Detection**
- **Object Detection**: Identifies specific food items in fridge images
- **Spoilage Classification**: Determines if detected food is fresh or spoiled
- **Reconstruction Analysis**: Uses autoencoder to detect anomalies

### **Web API Endpoints**
- **`POST /predict`** - ResNet18 classification endpoint
- **`POST /check_spoilage`** - Autoencoder-based spoilage detection

### **Model Performance**
- **YOLOv8**: 92% mAP50 on validation set (best run)
- **ResNet18**: Binary classification for fresh/spoiled detection
- **Autoencoder**: t-SNE visualization for feature analysis

## 🛠️ Installation & Setup

### **Quick Setup (Recommended)**
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd fridge_spoilage_detection_starter

# 2. Run the automated setup script
chmod +x setup_environment.sh
./setup_environment.sh

# 3. Activate the environment
conda activate fridge_detection

# 4. Test the installation
python scripts/detection/enhanced_detection.py --help
```
### **Manual Setup**
```bash
# 1. Create conda environment
conda create -n fridge_detection python=3.9 -y
conda activate fridge_detection

# 2. Install exact dependencies
pip install -r requirements.txt

# 3. Test installation
python -c "import torch, cv2, ultralytics; print(' All dependencies working!')"
```

### **Running the Application**
```bash
# Activate environment
conda activate fridge_detection

# Run detection API
python app/main.py  # For ResNet18 classifier
# OR
python app/app.py   # For autoencoder spoilage detection

# Run enhanced detection
python scripts/detection/enhanced_detection.py --model yolov8n.pt --image test.jpg --show

# Train a model
python scripts/detection/train_yolov8.py --model yolov8n.pt --data data/detection_dataset/data.yaml --epochs 5
```

## 📊 Dataset Information

### **Roboflow Dataset**
- **Source**: [Fridge objects dataset](https://universe.roboflow.com/fooddetection-essdj/fridge-objects/dataset/12)

### **Classification Dataset**
- **Format**: Fresh vs Spoiled binary classification
- **Split**: ~70% train, ~20% validation, ~10% test

## 🔧 Technical Stack
- **Deep Learning**: PyTorch, TorchVision
- **Object Detection**: YOLOv8 (Ultralytics)
- **Web Framework**: FastAPI, Uvicorn
- **Image Processing**: OpenCV, PIL, scikit-image
- **Visualization**: Matplotlib, Seaborn
- **Data Analysis**: scikit-learn, t-SNE

## 📈 Model Training Results

### **YOLOv8 Performance (Best Run - train6)**
- **mAP50**: 92.7%
- **mAP50-95**: 80.5%
- **Precision**: 91.3%
- **Recall**: 85.8%

### **Training Runs**
- Multiple YOLOv8 experiments with different configurations
- Data augmentation experiments
- Transfer learning from pre-trained models

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📝 Usage Examples

### **Object Detection**
```python
from scripts.detection.run_yolo8v import detect_items
items = detect_items("path/to/fridge_image.jpg", save_crops=True)
```

### **Spoilage Classification**
```python
from app.infer_spoilage import classify_image
label, loss = classify_image("path/to/food_image.jpg")
```

### **Web API**
```bash
# Start the server
uvicorn app.main:app --reload

# Test endpoint
curl -X POST "http://localhost:8000/predict" -F "file=@food_image.jpg"
```

## Contributing & Installation Guide

### 1. Fork and Clone
```bash
# Fork the repo on GitHub first, then clone your fork
git clone https://github.com/YOUR_USERNAME/fridgespoilagedetection.git
cd fridgespoilagedetection
```

### 2. Install Dependencies

**Option A: Automated Setup (Recommended)**
```bash
# One-command setup - installs everything
./setup_environment.sh
conda activate fridge_detection
```

**Required Dependencies:**
- Python 3.9
- PyTorch & TorchVision
- OpenCV (cv2)
- Ultralytics (YOLOv8)
- FastAPI & Uvicorn
- Pandas, NumPy, Scikit-learn
- Pillow, Matplotlib, Seaborn

### 3. Test Installation
```bash
# Verify everything works
python final_test.py
```

### 4. Create Your Branch
```bash
# Always start from latest main
git checkout main
git pull origin main

# Create your feature branch
git checkout -b feature/your-feature-name
```

### 5. Make Changes & Push
```bash
# Make your changes, then:
git add .
git commit -m "Add your feature description"
git push origin feature/your-feature-name
```

### 6. Create Pull Request
- Go to your fork on GitHub
- Click "Compare & pull request"
- Describe your changes
- Submit for review



