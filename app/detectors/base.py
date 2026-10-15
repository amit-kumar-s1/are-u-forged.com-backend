from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from app.core.config import get_settings

class DetectorUnavailable(Exception):
    """Model not configured / weights or dependencies missing."""

@dataclass
class RawPrediction:
    fake_probability: float
    model: str
    details: dict[str, Any] = field(default_factory=dict)

def to_label(p_fake: float):
    if abs(p_fake - 0.5) < get_settings().inconclusive_band:
        return "inconclusive", None
    return ("fake", p_fake) if p_fake > 0.5 else ("real", 1 - p_fake)

class Detector(ABC):
    """Adapter contract: one implementation per modality; swap models without touching the API."""
    modality: str
    @abstractmethod
    def predict(self, payload) -> RawPrediction: ...
