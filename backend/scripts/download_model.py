#!/usr/bin/env python
"""
Script to download model files from Hugging Face during application startup.
This is used in production deployments to fetch the model from Hugging Face
instead of storing it in the Git repository.
"""

import os
import sys
import json
import requests
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_downloader')

# Hugging Face repository details
HF_REPO = os.environ.get('HF_MODEL_REPO', 'phangrisani/MRI-Classification')
MODEL_FILENAME = 'default.pth'
CLASSES_FILENAME = 'default_classes.json'

# Maximum download retries
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def create_url(repo, filename):
    """Create the Hugging Face download URL for a file"""
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def download_file(url, destination, description):
    """Download a file from a URL to a local destination with retries"""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Downloading {description} from {url} (Attempt {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress reporting
            total_size = int(response.headers.get('content-length', 0))
            
            # Create parent directories if they don't exist
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress reporting
            with open(destination, 'wb') as f:
                downloaded = 0
                chunk_size = 8192
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = int(100 * downloaded / total_size)
                            if percent % 10 == 0:
                                logger.info(f"Download progress: {percent}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
            
            logger.info(f"Successfully downloaded {description} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {description}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to download {description} after {MAX_RETRIES} attempts")
                return False


def create_default_class_mapping(destination):
    """Create a default class mapping file if download fails"""
    try:
        # Default mapping for MRI-Classification
        class_mapping = {
            "0": "Glioma",
            "1": "Meningioma",
            "2": "Normal",
            "3": "Pituitary"
        }
        
        # Write the mapping to file
        with open(destination, 'w') as f:
            json.dump(class_mapping, f, indent=4)
            
        logger.info(f"Created default class mapping at {destination}")
        return True
    except Exception as e:
        logger.error(f"Error creating default class mapping: {str(e)}")
        return False


def is_production():
    """Determine if the application is running in production"""
    # Check multiple environment variables to determine if in production
    return (os.environ.get('FLASK_ENV') == 'production' or
            os.environ.get('RENDER') == 'true' or  # Render.com specific
            os.environ.get('RENDER_EXTERNAL_URL') is not None or
            os.environ.get('ENVIRONMENT') == 'production')


def main():
    """Main function to download model files"""
    # Only download in production, otherwise use local files
    if not is_production():
        logger.info("Development environment detected. Using local model files.")
        return True
    
    logger.info("Production environment detected. Checking for model files.")
    
    # Determine the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # Go up two levels from scripts dir
    
    # Models directory - use application structure
    models_dir = project_root / "results" / "models"
    
    # Ensure the models directory exists
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths to model files
    model_file = models_dir / MODEL_FILENAME
    classes_file = models_dir / CLASSES_FILENAME
    
    # Check and download model file if needed
    if not model_file.exists():
        logger.info(f"Model file not found at {model_file}. Downloading from Hugging Face.")
        success = download_file(
            create_url(HF_REPO, MODEL_FILENAME),
            model_file,
            "model file"
        )
        if not success:
            logger.error("Failed to download model file. Deployment cannot proceed.")
            return False
    else:
        logger.info(f"Using existing model file at {model_file}")
    
    # Check and download class mapping file if needed
    if not classes_file.exists():
        logger.info(f"Class mapping file not found at {classes_file}. Downloading from Hugging Face.")
        success = download_file(
            create_url(HF_REPO, CLASSES_FILENAME),
            classes_file,
            "class mapping file"
        )
        if not success:
            logger.warning("Creating default class mapping file instead.")
            success = create_default_class_mapping(classes_file)
            if not success:
                logger.error("Failed to create default class mapping. Deployment may have issues.")
                return False
    else:
        logger.info(f"Using existing class mapping file at {classes_file}")
    
    logger.info("Model preparation complete!")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("Model preparation failed. Exiting.")
        sys.exit(1)  # Exit with error code if download failed
    logger.info("Model downloader completed successfully.")
