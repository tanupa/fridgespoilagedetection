# 🍎 Fridge Spoilage Detection System

A comprehensive multi-approach machine learning system for detecting food spoilage in refrigerator images using object detection, supervised classification, and unsupervised autoencoder methods.

## 📁 Project Structure

```
fridge_spoilage_detection/
├── 🤖 computer_vision/        # All ML models, datasets, and training scripts
├── 📱 frontend/               # Web interface (React/Vue.js) - Ready for development
├── 🔧 backend/               # API server (FastAPI) - Ready for development
└── 📚 README.md              # This file
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/tanupa/fridgespoilagedetection.git
cd fridgespoilagedetection
```

### 2. Set Up Computer Vision Environment
```bash
# Navigate to computer vision folder
cd computer_vision

# Automated setup
./setup_environment.sh
conda activate fridge_detection

# Test installation
python final_test.py
```

### 3. Run Computer Vision Models
```bash
# Train YOLOv8 model
python scripts/detection/train_yolov8.py --model yolov8n.pt --data data/detection_dataset/data.yaml --epochs 10

# Run detection
python scripts/detection/enhanced_detection.py --model yolov8n.pt --source path/to/image.jpg
```

## 🎯 What's Included

### ✅ Computer Vision (Complete)
- **Object Detection**: YOLOv8 for food item identification
- **Spoilage Classification**: ResNet18 binary classification  
- **Unsupervised Detection**: Autoencoder reconstruction error analysis
- **Datasets**: Ready-to-use training data (2,425+ images)
- **Training Scripts**: Complete training pipeline
- **Model Export**: ONNX, TensorFlow Lite, CoreML support
- **Performance**: 92.7% mAP50, 95%+ accuracy

### 🚧 Frontend (Ready for Development)
- **Planned**: React/Vue.js web interface
- **Features**: Image upload, real-time detection, results display
- **Status**: Empty folder ready for UI development

### 🚧 Backend (Ready for Development)  
- **Planned**: FastAPI server with REST endpoints
- **Features**: Model integration, image processing, database
- **Status**: Empty folder ready for API development

## 📋 Components

| Component | Status | Description | Location |
|-----------|--------|-------------|----------|
| **Computer Vision** | ✅ Complete | ML models and training | `computer_vision/` |
| **Frontend** | 🚧 Ready | Web interface | `frontend/` |
| **Backend** | 🚧 Ready | API server | `backend/` |

## 🤝 Contributing

### For Computer Vision Work
```bash
cd computer_vision
# All ML work happens here
```

### For Frontend Development
```bash
cd frontend
# Create React/Vue.js app here
```

### For Backend Development
```bash
cd backend  
# Create FastAPI server here
```

**Quick workflow:**
1. Fork the repository
2. Clone your fork
3. Work in the appropriate folder (`computer_vision/`, `frontend/`, or `backend/`)
4. Create feature branch: `git checkout -b feature/your-feature`
5. Make changes and test
6. Push and create pull request

## 📊 Model Performance

- **YOLOv8 mAP50**: 92.7%
- **Classification Accuracy**: 95%+
- **Processing Speed**: <100ms per image
- **Supported Formats**: JPG, PNG, WebP

## 🔗 Links

- [Computer Vision Guide](computer_vision/README.md)
- [Frontend Development](frontend/README.md)
- [Backend Development](backend/README.md)

## 🎯 Next Steps

1. **Computer Vision**: ✅ Complete - Ready to use
2. **Frontend**: 🚧 Create React/Vue.js app in `frontend/` folder
3. **Backend**: 🚧 Create FastAPI server in `backend/` folder
4. **Integration**: Connect frontend to backend to computer vision models

---

**Built with ❤️ for food waste reduction**

**Current Status**: Computer vision models are complete and ready. Frontend and backend folders are set up and ready for development! 🎉