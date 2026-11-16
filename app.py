"""Start the whole system:   python app.py   then open http://127.0.0.1:5000"""

import config
from backend.app import create_app

if __name__ == "__main__":
    app = create_app()
    print(f"Open http://{config.HOST}:{config.PORT} in your browser (Ctrl+C to stop)")
    # use_reloader=False: otherwise Flask would start the AI pipeline twice
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True, use_reloader=False)