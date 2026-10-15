
from app.core.config import get_settings

# Only the three selected models; EfficientNet-B0 is removed.
MODEL_WEIGHTS = {
    "resnet18": 0.25,
    "efficientnet_b4": 0.25,
    "dinov2": 0.30,
}


def ensemble_predict(predictions: dict) -> dict:
    """Combine the fake probabilities from all three models."""

    missing = set(MODEL_WEIGHTS) - set(predictions)
    if missing:
        raise ValueError(
            f"Missing model predictions: {', '.join(sorted(missing))}"
        )

    for name in MODEL_WEIGHTS:
        probability = predictions[name]
        if not isinstance(probability, (int, float)):
            raise ValueError(f"Invalid probability from {name}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability from {name} must be between 0 and 1")

    # Normalize weights because the original weights sum to 0.80.
    total_weight = sum(MODEL_WEIGHTS.values())
    fake_probability = sum(
        MODEL_WEIGHTS[name] * predictions[name]
        for name in MODEL_WEIGHTS
    ) / total_weight

    fake_probability = max(0.0, min(1.0, fake_probability))
    threshold = get_settings().image_fake_threshold

    return {
        "label": "fake" if fake_probability >= threshold else "real",
        "fake_probability": round(fake_probability, 6),
        "real_probability": round(1.0 - fake_probability, 6),
        "confidence": round(
            max(fake_probability, 1.0 - fake_probability), 6
        ),
        "threshold": threshold,
        "weights": {
            name: round(weight / total_weight, 6)
            for name, weight in MODEL_WEIGHTS.items()
        },
        "model_scores": {
            name: round(float(predictions[name]), 6)
            for name in MODEL_WEIGHTS
        },
    }
