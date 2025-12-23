from fastapi import APIRouter, HTTPException
import logging

# Ensure sys.path is set
from backend.app.core.config import PROJECT_ROOT 

from backend.app.models.schemas import BatchScreeningRequest, BatchScreeningResponse, ScreenedMolecule
from backend.app.services.ml_service import ml_service
from inference.predictor import DrugScreener

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/screen", response_model=BatchScreeningResponse)
async def screen_library(request: BatchScreeningRequest):
    if not ml_service.predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        screener = DrugScreener(ml_service.predictor)
        
        # 1. Screen (Predict & Sort)
        results_df = screener.screen_library(
            request.smiles_list, 
            top_k=request.top_k if not request.apply_lipinski else len(request.smiles_list),
            ascending=request.ascending
        )
        
        # 2. Filter (Optional)
        if request.apply_lipinski:
            results_df = screener.filter_by_lipinski(results_df)
            results_df = results_df.head(request.top_k)
        
        # 3. Format Response
        output_results = []
        for idx, row in results_df.iterrows():
            props = {k: v for k, v in row.to_dict().items() if k not in ['smiles', 'score', 'rank']}
            
            output_results.append(ScreenedMolecule(
                rank=idx + 1 if 'rank' not in row else row['rank'],
                smiles=row['smiles'],
                score=float(row['score']),
                properties=props
            ))
            
        return BatchScreeningResponse(
            total_input=len(request.smiles_list),
            total_screened=len(output_results),
            results=output_results
        )

    except Exception as e:
        logger.error(f"Screening error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
