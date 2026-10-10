from fastapi import APIRouter, HTTPException

from app.ml.sentiment_model import predict_sentiment
from app.schemas.sentiment import SentimentRequest, SentimentResponse

router = APIRouter(prefix="/api/v1/sentiment", tags=["Sentiment"])


@router.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest) -> SentimentResponse:
    try:
        result = predict_sentiment(request.text)
        return SentimentResponse(**result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Sentiment model artifacts are unavailable.",
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Sentiment prediction failed.",
        )
