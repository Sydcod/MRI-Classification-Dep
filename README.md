# 🧠 Brain MRI Classification Web Application

## 🚀 Project Overview

A comprehensive web application for brain MRI classification utilizing advanced deep learning and explainable AI techniques, designed to provide precise medical image analysis through cutting-edge machine learning methodologies.

### Key Features
- 🔬 Advanced brain MRI classification using DenseNet169
- 💡 Multiple Explainable AI (XAI) visualization techniques
- 🖥️ Full-stack web application with React frontend and Flask backend
- 🧠 Sophisticated model interpretation capabilities

## 🏗️ Detailed Project Architecture

### Complete Directory Structure
```
brain-mri-classifier/
│
├── backend/                    # Backend Python application
│   ├── models/                 # Machine learning model definitions
│   │   ├── densenet169.py      # DenseNet169 model implementation
│   │   ├── model_utils.py      # Model utility functions
│   │   └── model_registry.py   # Model checkpoint and versioning
│   │
│   ├── xai/                    # Explainable AI modules
│   │   ├── gradcam.py          # Grad-CAM implementation
│   │   ├── scorecam.py         # ScoreCAM visualization
│   │   ├── shap_explainer.py   # SHAP value explanations
│   │   └── visualization.py    # Visualization utilities
│   │
│   ├── data/                   # Data handling
│   │   ├── preprocessing.py    # Image preprocessing
│   │   ├── augmentation.py     # Data augmentation techniques
│   │   └── dataset.py          # Custom dataset classes
│   │
│   ├── utils/                  # Utility modules
│   │   ├── config.py           # Configuration management
│   │   ├── logging_utils.py    # Logging configurations
│   │   └── metrics.py          # Custom metric calculations
│   │
│   ├── tests/                  # Unit and integration tests
│   │   ├── test_model.py
│   │   ├── test_xai.py
│   │   └── test_preprocessing.py
│   │
│   ├── config/                 # Configuration files
│   │   ├── training_config.py  # Training configuration
│   │   └── inference_config.py # Inference configuration
│   │
│   ├── scripts/                # Utility scripts
│   │   ├── train.py            # Training script
│   │   ├── predict.py          # Inference script
│   │   └── export_model.py     # Model export utilities
│   │
│   ├── api/                    # API endpoints
│   │   ├── __init__.py         # API initialization
│   │   ├── prediction.py       # Prediction endpoints
│   │   ├── explanation.py      # XAI endpoints
│   │   └── model_management.py # Model management endpoints
│   │
│   ├── requirements.txt        # Python dependencies
│   └── app.py                  # Flask application entry point
│
├── frontend/                   # React frontend application
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── ImageUploader.tsx
│   │   │   ├── PredictionResult.tsx
│   │   │   └── XAIVisualization.tsx
│   │   │
│   │   ├── pages/              # Page components
│   │   │   ├── HomePage.tsx
│   │   │   ├── PredictPage.tsx
│   │   │   └── ExplainabilityPage.tsx
│   │   │
│   │   ├── services/           # API interaction
│   │   │   ├── predictionService.ts
│   │   │   └── xaiService.ts
│   │   │
│   │   ├── styles/             # CSS and styling
│   │   └── utils/              # Frontend utilities
│   │
│   ├── public/
│   ├── package.json
│   └── README.md
│
├── data/                       # Data storage
│   ├── raw/                    # Original MRI images
│   ├── processed/              # Preprocessed images
│   └── dataset_splits/         # Train/validation/test dataset files
│       ├── train_dataset.csv   # Training set file list and labels
│       ├── val_dataset.csv     # Validation set file list and labels
│       └── test_dataset.csv    # Test set file list and labels
│
├── docs/                       # Documentation
│   ├── api_reference.md
│   └── development_guide.md
│
├── results/                    # Comprehensive experiment results
│   ├── models/                 # Saved model checkpoints
│   ├── logs/                   # Training and inference logs
│   ├── visualizations/         # XAI visualization outputs
│   ├── metrics/                # Stored performance metrics
│   │   ├── confusion_matrices/ # Confusion matrix visualizations
│   │   ├── roc_curves/         # ROC curve plots
│   │   └── learning_curves/    # Training and validation metrics over time
│   ├── prediction_samples/     # Example predictions with ground truth
│   │   ├── correct_predictions/
│   │   └── misclassified_samples/
│   └── analysis/               # Detailed model performance analysis
│       ├── class_performance/
│       └── error_analysis/
│
├── .github/                    # CI/CD workflows
│   └── workflows/
│       ├── backend_tests.yml
│       └── frontend_tests.yml
│
├── README.md
├── requirements.txt            # Project-level requirements
└── docker-compose.yml          # Docker composition file
```

### Technology Stack
- **Backend**: 
  - Python 3.10+
  - Flask
  - PyTorch
- **Frontend**: 
  - React
  - TypeScript
- **Machine Learning**:
  - DenseNet169
  - PyTorch for deep learning
  - Captum for XAI implementations
- **Visualization**:
  - Matplotlib & Seaborn for static visualizations
  - TensorBoard for training metrics
  - React components for interactive visualizations
- **Deployment**:
  - Docker
  - Nginx
  - Gunicorn

## 🧠 Dataset Details

### Source
- **Dataset**: PMRAM - Bangladeshi Brain Cancer MRI Dataset
- **Comprehensive Characteristics**:
  - **Raw Dataset**: 1,500 original MRI images
  - **Training Strategy**: Use raw images with on-the-fly augmentation during training
  - Image Dimensions: 512×512 pixels
  - Color Space: Grayscale (single channel)
  - Data Split:
    * Training: 70% (1,050 images + augmentation)
    * Validation: 15% (225 images, no augmentation)
    * Test: 15% (225 images, no augmentation)
  - Stratified Sampling: Ensures balanced class representation

## 🔧 Comprehensive Installation Guide

### Prerequisites
- Python 3.10+ with pip
- Node.js 16+ with npm
- CUDA 11.3+ (recommended for GPU acceleration)
- Docker (optional, for containerized deployment)

### Step-by-Step Setup

#### Backend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/brain-mri-classifier.git
cd brain-mri-classifier/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# Or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (select appropriate version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Frontend Setup
```bash
cd ../frontend
npm install
npm run build
```

#### Full Application Launch
```bash
# Start backend (Flask)
cd ../backend
flask run

# In another terminal, start frontend
cd ../frontend
npm start
```

## 🏋️‍♀️ Training the Model

### Detailed Training Configuration
- **Model**: DenseNet169 with ImageNet pre-training
  - Modified first convolutional layer to accept single-channel grayscale input
  - Preserves pre-trained weights by averaging RGB channel weights
- **Optimizer**: AdamW
  - Learning Rate: 1e-4
  - Weight Decay: 1e-5
- **Learning Rate Scheduler**: Cosine Annealing with Warm Restarts
- **Epochs**: 30-40
- **Early Stopping**: 7 epochs without improvement

### Training Command
```bash
python scripts/train.py \
  --config config/training_config.py \
  --gpu 0 \
  --mixed-precision
```

## 🔍 Inference Modes

### CLI Prediction
```bash
python scripts/predict.py \
  --image path/to/mri/scan.jpg \
  --xai-method gradcam \
  --output-dir results/predictions
```

### API Inference
The application exposes RESTful endpoints for model inference and XAI explanations.

## 💡 Explainable AI Techniques

### Supported XAI Methods
1. **Grad-CAM**
   - Generates heat maps of critical areas
   - Shows neural network's focus points
2. **ScoreCAM**
   - Identifies most influential image regions
   - Provides pixel-level importance mapping
3. **SHAP Values**
   - Global feature importance analysis
   - Quantifies feature contributions

## 🌐 API Endpoints

### Prediction Endpoints
- **POST** `/api/predict`
  - Input: Multipart form-data with MRI image
  - Response: JSON with classification result

### XAI Explanation Endpoints
- **POST** `/api/explain`
  - Input: Image, XAI method
  - Response: Visualization and explanation JSON

### Model Management Endpoints
- **GET** `/api/models`
  - Lists available models
- **POST** `/api/models/upload`
  - Uploads a new model
- **GET** `/api/models/{model_id}`
  - Gets model information

## 📈 Performance Tracking and Visualization

### Model Performance Repository
The `/results/` directory provides comprehensive tracking of model performance and analysis:

#### 1. Metrics Visualization
- **Location**: `results/metrics/`
- **Contents**:
  - Confusion Matrices
  - ROC Curves
  - Learning Curves
  - Detailed Performance Graphs

#### 2. Prediction Samples
- **Location**: `results/prediction_samples/`
- **Breakdown**:
  - Correct Predictions
  - Misclassified Samples
  - Visualizations with model confidence

#### 3. Detailed Analysis
- **Location**: `results/analysis/`
- **Includes**:
  - Class-wise Performance
  - Error Analysis
  - Feature Importance Plots

### Accessing Visualizations
```bash
# View TensorBoard logs
tensorboard --logdir results/logs/

# Generate performance report
python scripts/generate_report.py \
  --output results/performance_report.pdf
```

## 🧪 Testing Strategy

### Backend Tests
```bash
# Run all backend tests
pytest backend/tests/

# Specific module testing
pytest backend/tests/test_model.py
```

### Frontend Tests
```bash
# Run React component tests
npm test --prefix frontend
```

## ⚠️ Ethical Disclaimer

This research-grade application is for educational and scientific purposes only. It must not be used for actual medical diagnosis without professional medical supervision.

## 📚 Citation & References

If you use this work, please cite:
```
@article{brainmri2025,
  title={Advanced Brain MRI Classification with Explainable AI Techniques},
  author={Your Name et al.},
  journal={Journal of Medical AI},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
