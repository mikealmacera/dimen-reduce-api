from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import your blueprints
    from .tsvd import tsvd_bp
    from .pca import pca_bp

    # Register the blueprints with a URL prefix
    app.register_blueprint(tsvd_bp, url_prefix='/tsvd')
    app.register_blueprint(pca_bp, url_prefix='/pca')

    @app.route('/')
    def home():
        return 'Welcome to Flask!'

    return app