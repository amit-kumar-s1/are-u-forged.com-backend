from app.detectors.base import Detector, DetectorUnavailable

class PendingTextDetector(Detector):
    """Integration pending: implement predict(text) with your real model (docs/MODEL_INTEGRATION.md)."""
    modality = "text"
    def predict(self, payload):
        raise DetectorUnavailable("The text detection model is not connected yet.")
