"""Flask application: creates the web server and starts the AI pipeline."""

import logging

from flask import Flask, send_from_directory

import config
from backend.routes.api import api
from backend.routes.challans import create_challan_blueprint
from backend.services.database import Database
from backend.services.pipeline import TrafficPipeline


def create_app(start_pipeline=True):
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # no log line for every request
    frontend = config.PROJECT_ROOT / "frontend"
    app = Flask(__name__, static_folder=str(frontend), static_url_path="")
    app.register_blueprint(api)

    database = Database(config.DB_PATH)
    pipeline = TrafficPipeline(database=database)
    app.extensions["database"] = database
    app.extensions["pipeline"] = pipeline
    app.register_blueprint(create_challan_blueprint(database, pipeline))

    @app.get("/")
    def index():
        return send_from_directory(frontend, "index.html")

    if start_pipeline:
        pipeline.start()  # models load in the background; the server is ready at once
    return app


# only for tools that import "app"; the real server is started by create_app() in the root app.py
app = create_app(start_pipeline=False)