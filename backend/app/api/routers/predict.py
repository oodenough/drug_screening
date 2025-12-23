from fastapi import APIRouter, HTTPException
import logging

from backend.app.models.schemas import SinglePredictionRequest, SinglePredictionResponse
from backend.app.services.ml_service import ml_service
from backend.app.services.chemistry import check_lipinski

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(request: SinglePredictionRequest):
    if not ml_service.predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = ml_service.predictor.predict_with_properties(request.smiles)
        
        if result['prediction'] is None:
             return SinglePredictionResponse(
                smiles=request.smiles,
                prediction=None,
                prediction_label="Error",
                properties=None,
                lipinski_passed=False,
                status="failed",
                error="Invalid SMILES or prediction failed"
            )

        props = result['properties']
        passed_lipinski = check_lipinski(props)
        
        # Assuming binary classification with 0.5 threshold for label
        label = "High Probability" if result['prediction'] > 0.5 else "Low Probability"
        
        return SinglePredictionResponse(
            smiles=request.smiles,
            prediction=float(result['prediction']),
            prediction_label=label,
            properties=props,
            lipinski_passed=passed_lipinski
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
