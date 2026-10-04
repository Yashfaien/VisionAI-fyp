"""
VisionAI - Project Root Runner Script
Starts the Flask server from workspace root
"""

import sys
import os

# Add vision_ai_project directory to sys.path
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_ai_project")
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import Flask application from vision_ai_project package
from vision_ai_project.app import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n=======================================================")
    print(f"[*] Launching VisionAI Clinical Diagnostic Platform")
    print(f"[*] Access Dashboard at: http://127.0.0.1:{port}")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
