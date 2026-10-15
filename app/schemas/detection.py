from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

class DetectionResult(BaseModel):
    modality: Literal["image", "video", "text"]
    label: Literal["real", "fake", "inconclusive"]
    # P(fake/AI) as output by the model. A model estimate, NOT proof.
    fake_probability: float = Field(ge=0, le=1)
    # Probability assigned to the chosen label = max(p, 1-p); null when inconclusive.
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    model: str
    processing_ms: int
    details: dict[str, Any] = {}
    disclaimer: str = "AI detection results are estimates, not definitive proof of authenticity or manipulation."

class TextRequest(BaseModel):
    text: str
