import os
import json
import base64
import requests
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv

# Create a Blueprint for AI interpretation routes
interpretation_bp = Blueprint('interpretation', __name__)

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

@interpretation_bp.route('/interpret', methods=['POST'])
def interpret_mri():
    """
    Endpoint to generate an AI interpretation of MRI images and classification results
    using OpenAI's Vision API
    
    Returns:
        JSON response with AI-generated interpretation
    """
    try:
        # Check if API key is available
        if not OPENAI_API_KEY:
            return jsonify({'error': 'OpenAI API key not configured'}), 500
            
        # Get data from request
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Required fields
        required_fields = ['original_image', 'heatmap_image', 'overlay_image', 'prediction', 'confidence']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract data
        original_image = data['original_image']
        heatmap_image = data['heatmap_image'] 
        overlay_image = data['overlay_image']
        prediction = data['prediction']
        confidence = data['confidence']
        
        # Prepare content for OpenAI Vision API
        prompt = generate_vision_prompt(prediction, confidence)
        
        # Prepare images for the request
        content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{original_image.split(',')[1] if ',' in original_image else original_image}",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": "This is the Grad-CAM visualization showing areas of interest that influenced the model's decision:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{overlay_image.split(',')[1] if ',' in overlay_image else overlay_image}",
                    "detail": "high"
                }
            }
        ]
        
        # Make request to OpenAI Vision API
        response = call_openai_vision_api(content)
        
        # Return response
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Interpretation error: {str(e)}'}), 500

def generate_vision_prompt(prediction, confidence):
    """
    Generate a prompt for the Vision API based on the prediction and confidence
    
    Args:
        prediction (str): The model's prediction
        confidence (float): The model's confidence score
        
    Returns:
        str: Prompt for the Vision API
    """
    confidence_percent = confidence * 100
    
    prompt = f"""
You are a specialized neuroradiologist. You're analyzing a brain MRI scan that has been classified by an AI system.

The AI system has classified this MRI as: {prediction} with {confidence_percent:.1f}% confidence.

Please provide a comprehensive medical interpretation of this brain MRI. Your analysis should include:

1. Confirmation or challenge of the AI's classification based on visible evidence
2. Detailed description of any abnormalities, lesions, or tumors visible in the scan
3. Typical characteristics of the identified condition and whether they're present in this scan
4. Potential differential diagnoses to consider
5. Suggested follow-up tests or imaging that might be needed
6. General treatment approaches for this type of condition (if applicable)

Please structure your response clearly with sections for each aspect of the analysis.
Use medical terminology but also provide plain-language explanations when appropriate.
"""
    return prompt

def call_openai_vision_api(content):
    """
    Call OpenAI Vision API with the provided content
    
    Args:
        content (list): List of content dictionaries for the API
        
    Returns:
        dict: Response from the API
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1500
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    # Check if the response was successful
    if response.status_code != 200:
        error_detail = response.json().get('error', {}).get('message', 'Unknown error')
        raise Exception(f"OpenAI API error (status {response.status_code}): {error_detail}")
    
    # Parse response
    response_data = response.json()
    
    # Extract assistant's message
    interpretation = response_data['choices'][0]['message']['content']
    
    return {
        'interpretation': interpretation
    }
