web: FLASK_ENV=production python backend/scripts/download_model.py && gunicorn -w 1 --log-level debug -b 0.0.0.0:$PORT wsgi:app --timeout 300 --preload
