
from app.detectors.image.detector import ThreeModelImageDetector
from app.detectors.video.detector import VideoDetector
from app.detectors.text.detector import PendingTextDetector

_DETECTORS = {
    "image": ThreeModelImageDetector(),
    "video": VideoDetector(),
    "text": PendingTextDetector(),
}


def get_detector(kind: str):
    detector = _DETECTORS.get(kind)

    if detector is None:
        raise ValueError(
            f"Unsupported detection type: {kind}"
        )

    return detector
