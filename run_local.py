import os
import logging
from presentation_app.app import app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("PORT", "8052"))
    logging.getLogger("run_local").info(f"Starting local server on http://localhost:{port}")
    app.run(debug=True, port=port)
