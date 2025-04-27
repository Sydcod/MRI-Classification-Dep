import os
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
import torch

# Import API blueprints
from backend.api.prediction import prediction_bp
from backend.api.explanation import explanation_bp
from backend.api.interpretation import interpretation_bp

def create_app(config=None):
    """
    Create and configure the Flask application
    
    Args:
        config (dict, optional): Configuration dictionary
        
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    # Enable CORS - use a simple configuration that allows all origins
    CORS(app, 
         origins="*",
         supports_credentials=True,
         allow_headers=["*"],
         methods=["GET", "POST", "OPTIONS"])
    
    # Additional CORS handling middleware for robustness
    @app.after_request
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    # Handle OPTIONS requests explicitly
    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def options_handler(path):
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    # Default configuration
    app.config.update(
        DEBUG=os.environ.get('FLASK_DEBUG', 'True') == 'True',
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-for-development-only'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB max upload
        UPLOAD_FOLDER=os.path.join(os.path.dirname(__file__), 'uploads'),
        RESULTS_FOLDER=os.path.join(os.path.dirname(__file__), '..', 'results')
    )
    
    # Override with custom config if provided
    if config:
        app.config.update(config)
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(prediction_bp, url_prefix='/api')
    app.register_blueprint(explanation_bp, url_prefix='/api')
    app.register_blueprint(interpretation_bp, url_prefix='/api')
    
    # Root route for health check
    @app.route('/')
    @cross_origin()
    def index():
        """Root endpoint for API health check"""
        return jsonify({
            'status': 'ok',
            'message': 'Brain MRI Classification API',
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors"""
        return jsonify({'error': 'Method not allowed'}), 405
    
    @app.errorhandler(500)
    def server_error(error):
        """Handle 500 errors"""
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    # Create and run the application
    app = create_app()
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    app.run(host=host, port=port)
