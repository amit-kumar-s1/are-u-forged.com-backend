
"""Three-model image deepfake detector.

Uses ResNet-18, EfficientNet-B4, and DINOv2.
"""

import io
import threading

from PIL import Image, UnidentifiedImageError

from app.detectors.base import Detector, DetectorUnavailable, RawPrediction
from app.detectors.image.ensemble import ensemble_predict
from app.detectors.image.model_loader import load_models
from app.detectors.image.predictor import predict
from app.detectors.image.preprocessing import preprocess_image


class ThreeModelImageDetector(Detector):
    modality = "image"

    def __init__(self):
        self._models = None
        self._device = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._models is not None:
            return

        with self._lock:
            if self._models is None:
                self._models, self._device = load_models()

    def predict(self, payload: bytes) -> RawPrediction:
        self._ensure_loaded()

        try:
            with Image.open(io.BytesIO(payload)) as source:
                image = source.convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(
                "The uploaded image could not be decoded."
            ) from exc

        tensors = preprocess_image(image)
        scores = predict(self._models, tensors, self._device)
        result = ensemble_predict(scores)

        return RawPrediction(
            fake_probability=result["fake_probability"],
            model="ResNet-18 + EfficientNet-B4 + DINOv2",
            details={
                "model_scores": result["model_scores"],
                "real_probability": result["real_probability"],
                "ensemble_threshold": result["threshold"],
                "ensemble_weights": result["weights"],
                "input_sizes": {
                    "resnet18": "224x224",
                    "efficientnet_b4": "380x380",
                    "dinov2": "518x518",
                },
            },
        )
