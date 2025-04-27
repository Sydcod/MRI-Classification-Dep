import os
from flask import Flask, jsonify
from flask_cors import CORS
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
    
    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    
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
