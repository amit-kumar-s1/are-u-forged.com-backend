from app.detectors.base import Detector, DetectorUnavailable

class PendingVideoDetector(Detector):
    """Integration pending: implement predict(path) with your real model (docs/MODEL_INTEGRATION.md)."""
    modality = "video"
    def predict(self, payload):
        raise DetectorUnavailable("The video detection model is not connected yet.")
