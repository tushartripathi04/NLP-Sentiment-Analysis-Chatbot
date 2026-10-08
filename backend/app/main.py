from fastapi import FastAPI

app = FastAPI(
    title="SentimentIQ API",
    description="NLP Sentiment Analysis and Conversational Assistant API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Welcome to SentimentIQ API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }