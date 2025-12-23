import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure the project root is in sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # Use a string for the app to enable reload (if needed) but here strict import
    # We import here to verify it works
    from backend.app.main import app
    
    uvicorn.run(app, host="0.0.0.0", port=8000)