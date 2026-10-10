from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to classify as negative, neutral, or positive.",
        examples=["I really enjoyed this product."],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank.")
        return value


class SentimentResponse(BaseModel):
    sentiment: Literal["negative", "neutral", "positive"]
    label: Literal[0, 1, 2]
    model: str
