import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
# This assumes the file is in backend/app/core/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    logger.info(f"Added project root to sys.path: {PROJECT_ROOT}")

MODEL_PATH = os.path.join(PROJECT_ROOT, 'saved_models', 'bbbp_model.pth')
