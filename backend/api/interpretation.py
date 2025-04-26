import os
import json
import base64
import requests
import re
from io import BytesIO
from PIL import Image
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
        
        # Process and resize images to ensure they're properly formatted and not too large
        original_image_processed = process_image_for_openai(original_image)
        overlay_image_processed = process_image_for_openai(overlay_image)
        
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
                    "url": original_image_processed,
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
                    "url": overlay_image_processed,
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

def process_image_for_openai(image_data):
    """
    Process an image for OpenAI Vision API by:
    1. Extracting base64 data from data URI if needed
    2. Resizing the image if it's too large
    3. Re-encoding it as a proper data URI
    
    Args:
        image_data (str): The image data as either data URI or base64
        
    Returns:
        str: Processed image as data URI ready for OpenAI API
    """
    try:
        # Extract base64 part if it's a data URI
        if ',' in image_data:
            # Get the MIME type from the data URI if available
            mime_match = re.match(r'data:(image/[^;]+);base64,', image_data)
            mime_type = mime_match.group(1) if mime_match else 'image/jpeg'
            
            # Extract the base64 part
            base64_data = image_data.split(',', 1)[1]
        else:
            # If it's just base64, assume JPEG
            mime_type = 'image/jpeg'
            base64_data = image_data
        
        # Decode the base64 data
        image_bytes = base64.b64decode(base64_data)
        
        # Open the image using PIL
        img = Image.open(BytesIO(image_bytes))
        
        # Check if image needs resizing (OpenAI recommends < 20MB)
        # We'll aim for much smaller to ensure reliability
        MAX_SIZE = (800, 800)  # Reasonable size for medical images
        if img.width > MAX_SIZE[0] or img.height > MAX_SIZE[1]:
            img.thumbnail(MAX_SIZE, Image.LANCZOS)
        
        # Save the potentially resized image to bytes
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)  # Use consistent format and quality
        
        # Get the new base64 encoded data
        new_base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Return as proper data URI
        return f"data:{mime_type};base64,{new_base64_data}"
    
    except Exception as e:
        print(f"Error processing image for OpenAI: {str(e)}")
        # Return the original if something goes wrong
        if ',' in image_data and image_data.startswith('data:'):
            return image_data
        else:
            return f"data:image/jpeg;base64,{image_data}"
