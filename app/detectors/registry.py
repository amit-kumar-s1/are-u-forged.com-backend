
from app.detectors.image.detector import ThreeModelImageDetector

# Keep your existing video and text imports here.

_DETECTORS = {
    "image": ThreeModelImageDetector(),
    # Keep your existing video and text entries here.
}

def get_detector(kind: str):
    detector = _DETECTORS.get(kind)
    if detector is None:
        raise ValueError(f"Unsupported detection type: {kind}")
    return detector
