"""
Main Application Entrypoint.

Allows running `streamlit run app.py` directly from the root project directory in VS Code.
"""

import sys
from pathlib import Path

# Automatically add project root, backend, and frontend directories to sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(FRONTEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import and execute main Streamlit application
from frontend.app import main

if __name__ == "__main__":
    main()
