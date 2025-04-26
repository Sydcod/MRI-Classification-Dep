# 🧠 Brain MRI Classification Web Application

A comprehensive web application for brain MRI classification utilizing advanced deep learning and explainable AI techniques.

## 🚀 Features

- 🔬 Brain MRI classification using DenseNet169
- 💡 Explainable AI visualizations with Grad-CAM
- 🖥️ Interactive web interface for image upload and analysis
- 📊 Real-time classification with confidence scores
- 🧪 Advanced model training with focal loss and class balancing

## 🏗️ Project Structure

The project is divided into two main components:

- **Backend**: Python Flask API with PyTorch for machine learning
- **Frontend**: React application with TypeScript and Tailwind CSS

## 📋 Prerequisites

- Python 3.10+
- Node.js 16+
- CUDA-compatible GPU (recommended for training)

## 🚀 Getting Started

### Backend Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## 🧠 Model Training

### Local Training

To train a new model locally:

```bash
cd backend
python scripts/train.py --data_dir /path/to/mri/dataset --split_dataset --batch_size 16 --epochs 30 --mixed_precision
```

### Google Colab Training

The model was trained using Google Colab with an NVIDIA A100-SXM4-40GB GPU for accelerated performance. See `colab_training.ipynb` for the complete training notebook.

#### Training Results
- **Test Accuracy**: 96.90%
- **Test F1 Score**: 0.9686
- **Training Time**: ~11-13s per epoch
- **Convergence**: Early stopping at epoch 16 (best model at epoch 13)

Important parameters:
- `--data_dir`: Path to the dataset directory containing class subdirectories
- `--split_dataset`: Split the dataset into train/val/test sets
- `--batch_size`: Number of samples per batch (adjust based on GPU memory)
- `--mixed_precision`: Enable mixed precision training for faster computation
- `--freeze_layers`: Number of DenseNet blocks to freeze (0-4)

## 🔎 Explainable AI

The application includes the following XAI methods:

- **Grad-CAM**: Visualizes important regions in the image that influenced the model's decision.

## 📝 API Endpoints

- **POST /api/predict**: Upload an MRI image and get a classification prediction
- **POST /api/explain**: Generate XAI visualizations for a given MRI image
- **GET /api/methods**: Get available XAI methods

## 🔍 Dataset

The model is designed to work with brain MRI images in the following format:
- Grayscale images
- 512x512 pixel dimensions
- Organized in class directories

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
