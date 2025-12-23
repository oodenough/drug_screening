import torch
import os
import logging
from typing import Dict, Any, Optional

# Ensure sys.path is set
from backend.app.core.config import MODEL_PATH

from features.molecular_features import MolecularFeaturizer
from models.drug_models import DrugPredictorMLP
from inference.predictor import DrugPredictor

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.predictor: Optional[DrugPredictor] = None
        self.device: str = "cpu"

    def get_device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """Loads the model and predictor."""
        try:
            self.device = self.get_device()
            logger.info(f"Loading model on {self.device}...")
            
            # Model configuration (must match training config)
            model = DrugPredictorMLP(input_dim=1024, hidden_dims=[512, 256, 128], output_dim=1)
            
            if os.path.exists(MODEL_PATH):
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                model.load_state_dict(state_dict)
                model = model.to(self.device)
                model.eval()
                logger.info(f"Model loaded from {MODEL_PATH}")
            else:
                logger.warning(f"Model file not found at {MODEL_PATH}. Using random weights.")
                model = model.to(self.device)
            
            featurizer = MolecularFeaturizer(fingerprint_size=1024, radius=2)
            self.predictor = DrugPredictor(model, featurizer, device=self.device)
            
            logger.info("ML components initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            self.predictor = None

ml_service = MLService()
