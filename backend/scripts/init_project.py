#!/usr/bin/env python
"""
Initialize project directory structure and create necessary placeholder files.
This ensures all paths are consistent and ready for use.
"""

import os
import json
import argparse
import shutil

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Initialize project directory structure")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args()

def create_directory(path, overwrite=False):
    """Create a directory if it doesn't exist"""
    if os.path.exists(path):
        if overwrite and os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"Recreated directory: {path}")
        else:
            print(f"Directory already exists: {path}")
    else:
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content="", overwrite=False):
    """Create a file with the given content if it doesn't exist"""
    if os.path.exists(path):
        if overwrite:
            with open(path, 'w') as f:
                f.write(content)
            print(f"Overwrote file: {path}")
        else:
            print(f"File already exists: {path}")
    else:
        with open(path, 'w') as f:
            f.write(content)
        print(f"Created file: {path}")

def main():
    """Initialize project directory structure"""
    args = parse_args()
    
    # Get root directory (assuming this script is in backend/scripts)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Create main directories
    directories = [
        os.path.join(root_dir, 'data'),
        os.path.join(root_dir, 'results'),
        os.path.join(root_dir, 'results', 'models'),
        os.path.join(root_dir, 'results', 'logs'),
        os.path.join(root_dir, 'results', 'visualizations'),
        os.path.join(root_dir, 'backend', 'uploads'),
        os.path.join(root_dir, 'backend', 'config'),
    ]
    
    for directory in directories:
        create_directory(directory, args.overwrite)
    
    # Create placeholder files
    placeholders = [
        os.path.join(root_dir, 'data', '.gitkeep'),
        os.path.join(root_dir, 'results', 'models', '.gitkeep'),
        os.path.join(root_dir, 'results', 'logs', '.gitkeep'),
        os.path.join(root_dir, 'results', 'visualizations', '.gitkeep'),
    ]
    
    for placeholder in placeholders:
        create_file(placeholder, "", args.overwrite)
    
    # Create default class mapping file
    default_mapping = {
        "0": "Glioma",
        "1": "Meningioma",
        "2": "Normal",
        "3": "Pituitary"
    }
    
    config_file = os.path.join(root_dir, 'backend', 'config', 'default_class_mapping.json')
    create_file(config_file, json.dumps(default_mapping, indent=4), args.overwrite)
    
    print("\nProject directory structure initialized successfully!")
    print("You can now run the following commands:")
    print("1. To train the model:")
    print("   python backend/scripts/train.py --data_dir Raw --split_dataset --mixed_precision --gradient_clipping")
    print("2. To run the web application:")
    print("   python run.py")

if __name__ == "__main__":
    main() 