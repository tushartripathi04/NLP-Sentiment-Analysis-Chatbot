
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.sentiment import router as sentiment_router

app = FastAPI(
    title="SentimentIQ API",
    description="NLP Sentiment Analysis and Conversational Assistant API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nlp-sentiment-analysis-chatbot.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

@app.get("/")
def root():
    return {
        "message": "Welcome to SentimentIQ API",
        "status": "running",
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(sentiment_router)



