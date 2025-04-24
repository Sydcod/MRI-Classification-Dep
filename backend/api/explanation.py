import os
import json
import torch
import torch.nn.functional as F
from flask import Blueprint, request, jsonify
import numpy as np
from PIL import Image
import io
import base64

from backend.models.densenet169 import DenseNet169MRI
from backend.data.preprocessing import preprocess_image_for_prediction, preprocess_image_for_display
from backend.xai.gradcam import GradCAM
from backend.xai.scorecam import ScoreCAM
from backend.xai.shap_explainer import SHAPExplainer

# Create a Blueprint for explanation routes
explanation_bp = Blueprint('explanation', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# XAI methods registry
XAI_METHODS = {
    'gradcam': {
        'name': 'Grad-CAM',
        'description': 'Gradient-weighted Class Activation Mapping'
    },
    'scorecam': {
        'name': 'Score-CAM',
        'description': 'Score-weighted Class Activation Mapping (gradient-free)'
    },
    'shap': {
        'name': 'SHAP',
        'description': 'SHapley Additive exPlanations for feature importance'
    }
}

# Import from prediction blueprint
from .prediction import allowed_file, model_registry, load_model, get_class_mapping, get_default_class_mapping

def generate_explanation(model, image_tensor, method='gradcam', target_class=None, alpha=0.5, colormap='jet'):
    """
    Generate an explanation for the given image
    
    Args:
        model (nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        method (str): XAI method name
        target_class (int): Target class index
        alpha (float): Transparency of heatmap overlay
        colormap (str): Colormap to use for heatmap
        
    Returns:
        tuple: (overlay_image, heatmap_image) as PIL Images
    """
    # Convert tensor to numpy for visualization
    original_image = image_tensor.squeeze(0).cpu().numpy()
    # Denormalize and convert to uint8
    original_image = ((original_image * 0.5 + 0.5) * 255).astype(np.uint8)
    # Convert from CxHxW to HxW for grayscale
    original_image = original_image[0]
    
    # Convert grayscale to RGB for visualization
    original_image_rgb = np.stack([original_image] * 3, axis=2)
    
    if method == 'gradcam':
        # Get target layer for Grad-CAM
        target_layer = model.get_grad_cam_target_layer()
        
        # Initialize Grad-CAM
        gradcam = GradCAM(model, target_layer)
        
        # Generate visualization
        overlay, heatmap = gradcam.visualize(
            image_tensor, 
            original_image_rgb,
            target_class=target_class,
            alpha=alpha,
            colormap=colormap
        )
    
    elif method == 'scorecam':
        # Get target layer for Score-CAM
        target_layer = model.get_grad_cam_target_layer()  # Use the same target layer
        
        # Initialize Score-CAM
        scorecam = ScoreCAM(model, target_layer)
        
        # Generate visualization
        overlay, heatmap = scorecam.visualize(
            image_tensor, 
            original_image_rgb,
            target_class=target_class,
            alpha=alpha,
            colormap=colormap
        )
    
    elif method == 'shap':
        # Initialize SHAP Explainer
        # Use a custom colormap for SHAP
        explainer = SHAPExplainer(model, n_samples=10)
        
        # Generate visualization
        overlay, heatmap = explainer.visualize(
            image_tensor, 
            original_image_rgb,
            target_class=target_class,
            alpha=alpha,
            colormap='RdBu_r' if colormap == 'jet' else colormap  # Default to RdBu_r for SHAP
        )
    
    else:
        raise ValueError(f"Unsupported XAI method: {method}")
    
    # Convert numpy arrays to PIL Images
    overlay_image = Image.fromarray(overlay)
    heatmap_image = Image.fromarray(heatmap)
    
    return overlay_image, heatmap_image

def image_to_base64(image):
    """
    Convert a PIL Image to base64 string
    
    Args:
        image (PIL.Image): Image to convert
        
    Returns:
        str: Base64 encoded image string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@explanation_bp.route('/explain', methods=['POST'])
def explain():
    """
    Endpoint to generate and return XAI explanations for an MRI image
    
    Returns:
        JSON response with explanation images
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
        
        # Get parameters from request
        model_id = request.form.get('model_id', 'default')
        xai_method = request.form.get('xai_method', 'gradcam')
        
        # Check if XAI method is supported
        if xai_method not in XAI_METHODS:
            return jsonify({'error': f'Unsupported XAI method: {xai_method}. Supported methods: {list(XAI_METHODS.keys())}'}), 400
        
        # Parse optional parameters
        try:
            alpha = float(request.form.get('alpha', 0.5))
            if alpha < 0 or alpha > 1:
                return jsonify({'error': 'Alpha must be between 0 and 1'}), 400
        except ValueError:
            return jsonify({'error': 'Alpha must be a number between 0 and 1'}), 400
            
        colormap = request.form.get('colormap', 'jet')
        
        # Get device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model if not already in registry
        if model_id not in model_registry:
            # In production, these paths would come from config
            model_path = os.path.join('results', 'models', f'{model_id}.pth')
            mapping_path = os.path.join('results', 'models', f'{model_id}_classes.json')
            
            # Get class mapping
            if os.path.exists(mapping_path):
                class_mapping = get_class_mapping(mapping_path)
            else:
                class_mapping = get_default_class_mapping()
            
            # Check if model file exists
            if not os.path.exists(model_path):
                return jsonify({'error': f'Model {model_id} not found'}), 404
                
            # Load model
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
            # Save original image for visualization
            file.seek(0)
            original_image = Image.open(file).convert('L')
            original_image = original_image.resize((512, 512))
            
            # Reset file pointer and preprocess for model input
            file.seek(0)
            image_tensor = preprocess_image_for_prediction(file, device=device)
        except Exception as e:
            return jsonify({'error': f'Error preprocessing image: {str(e)}'}), 500
        
        # Make prediction to get target class
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Generate explanation
        try:
            overlay_image, heatmap_image = generate_explanation(
                model, 
                image_tensor, 
                method=xai_method,
                target_class=predicted_class,
                alpha=alpha,
                colormap=colormap
            )
        except Exception as e:
            return jsonify({'error': f'Error generating explanation: {str(e)}'}), 500
        
        # Convert original image to RGB for consistency
        original_rgb = original_image.convert('RGB')
        
        # Prepare response
        response = {
            'explanation_method': xai_method,
            'explanation_name': XAI_METHODS[xai_method]['name'],
            'explanation_description': XAI_METHODS[xai_method]['description'],
            'prediction': class_mapping.get(str(predicted_class), f'Class {predicted_class}'),
            'confidence': confidence,
            'original_image': image_to_base64(original_rgb),
            'heatmap_image': image_to_base64(heatmap_image),
            'overlay_image': image_to_base64(overlay_image),
            'parameters': {
                'alpha': alpha,
                'colormap': colormap
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Explanation error: {str(e)}'}), 500

@explanation_bp.route('/methods', methods=['GET'])
def get_methods():
    """
    Get available XAI methods
    
    Returns:
        JSON response with available XAI methods
    """
    return jsonify({
        'methods': {
            method_id: {
                'name': method['name'],
                'description': method['description']
            }
            for method_id, method in XAI_METHODS.items()
        }
    }) 