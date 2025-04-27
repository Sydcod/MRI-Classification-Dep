#!/bin/bash

# Install the requests package which is needed for model download
pip install requests

# Run the model download script
cd backend
python scripts/download_model.py
cd ..

# Run the application with gunicorn using the wsgi entry point
# This will properly set up the Python path
gunicorn --bind 0.0.0.0:$PORT wsgi:app
