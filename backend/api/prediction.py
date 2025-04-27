import os
import json
import torch
import torch.nn.functional as F
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import base64
from backend.models.densenet169 import DenseNet169MRI
from backend.data.preprocessing import preprocess_image_for_prediction
from flask_cors import cross_origin

# Create a Blueprint for prediction routes
prediction_bp = Blueprint('prediction', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Model registry to keep track of loaded models
model_registry = {}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path, num_classes=4, device='cpu'):
    """
    Load a model from a checkpoint file
    
    Args:
        model_path (str): Path to the model checkpoint
        num_classes (int): Number of output classes
        device (str): Device to load the model on
        
    Returns:
        DenseNet169MRI: Loaded model
    """
    # Initialize model
    model = DenseNet169MRI(num_classes=num_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    # Validate model structure
    try:
        # Create a small random input tensor
        dummy_input = torch.randn(1, 1, 512, 512, device=device)
        # Run a forward pass to ensure the model architecture is valid
        with torch.no_grad():
            outputs = model(dummy_input)
        # Check that the output dimension matches the expected number of classes
        if outputs.shape[1] != num_classes:
            raise ValueError(f"Model output dimension ({outputs.shape[1]}) doesn't match expected number of classes ({num_classes})")
    except Exception as e:
        raise ValueError(f"Model validation failed: {str(e)}")
    
    return model

def get_class_mapping(mapping_path):
    """
    Load class index to label mapping
    
    Args:
        mapping_path (str): Path to the class mapping JSON file
        
    Returns:
        dict: Class index to label mapping
    """
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    return class_mapping

def get_default_class_mapping():
    """
    Get the default class mapping for brain MRI classification
    
    Returns:
        dict: Class index to label mapping
    """
    # Try to load from config file first
    config_path = os.path.join('backend', 'config', 'default_class_mapping.json')
    if os.path.exists(config_path):
        try:
            return get_class_mapping(config_path)
        except:
            pass
    
    # Default mapping if file doesn't exist or has errors
    return {
        "0": "Glioma",
        "1": "Meningioma",
        "2": "Normal",
        "3": "Pituitary"
    }

@prediction_bp.route('/predict', methods=['POST'])
@cross_origin(origins='*', supports_credentials=True)
def predict():
    """
    Endpoint to predict the class of an MRI image
    
    Returns:
        JSON response with prediction results
    """
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        
        # Check if a valid file was uploaded
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Allowed formats: png, jpg, jpeg, tif, tiff'}), 400
        
        # Get model ID from request (default to 'default')
        model_id = request.form.get('model_id', 'default')
        
        # Get device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model if not already in registry
        if model_id not in model_registry:
            # In production, these paths would come from config
            model_path = os.path.join('results', 'models', f'{model_id}.pth')
            mapping_path = os.path.join('results', 'models', f'{model_id}_classes.json')
            
            # First, determine the class mapping - before loading the model
            if os.path.exists(mapping_path):
                class_mapping = get_class_mapping(mapping_path)
            else:
                class_mapping = get_default_class_mapping()
            
            # Make sure the results/models directory exists
            if not os.path.exists(os.path.dirname(model_path)):
                return jsonify({'error': 'Model directory not found. Please train a model first.'}), 404
            
            # Check if model file exists
            if not os.path.exists(model_path):
                return jsonify({'error': f'Model {model_id} not found'}), 404
                
            # Load model after determining the class mapping
            try:
                model = load_model(model_path, num_classes=len(class_mapping), device=device)
                model_registry[model_id] = {
                    'model': model,
                    'class_mapping': class_mapping,
                    'device': device
                }
            except Exception as e:
                return jsonify({'error': f'Error loading model: {str(e)}'}), 500
        
        # Get model and class mapping from registry
        model = model_registry[model_id]['model']
        class_mapping = model_registry[model_id]['class_mapping']
        device = model_registry[model_id]['device']
        
        # Preprocess image
        try:
            # Preprocess the image
            image_tensor = preprocess_image_for_prediction(file, device=device)
        except Exception as e:
            return jsonify({'error': f'Error preprocessing image: {str(e)}'}), 500
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        # Get class label
        class_label = class_mapping.get(str(predicted_class), f'Class {predicted_class}')
        
        # Prepare response
        response = {
            'prediction': class_label,
            'confidence': confidence,
            'probabilities': {
                class_mapping.get(str(i), f'Class {i}'): prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'model_id': model_id
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
