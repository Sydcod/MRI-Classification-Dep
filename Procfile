web: FLASK_ENV=production python backend/scripts/download_model.py && gunicorn -w 4 -b 0.0.0.0:$PORT wsgi:app --timeout 120
