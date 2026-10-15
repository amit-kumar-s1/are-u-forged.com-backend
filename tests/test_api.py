import io
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.api import routes
from app.detectors.base import Detector, RawPrediction

class Mock(Detector):
    modality = "image"
    def __init__(self, p): self.p = p
    def predict(self, payload): return RawPrediction(self.p, "mock-test-model")

c = TestClient(app)
def png():
    b = io.BytesIO(); Image.new("RGB", (8, 8)).save(b, "PNG"); return b.getvalue()
def post(): return c.post("/api/v1/detect/image", files={"file": ("a.png", png(), "image/png")})

def test_health(): assert c.get("/health").json() == {"status": "ok"} and c.get("/api/v1/health").status_code == 200
def test_image_missing_model_is_503(): assert post().status_code == 503 and post().json()["code"] == "model_not_configured"
def test_image_mock_fake_and_inconclusive():
    app.dependency_overrides[routes.image_detector] = lambda: Mock(0.9)
    r = post().json(); assert r["label"] == "fake" and r["confidence"] == 0.9
    app.dependency_overrides[routes.image_detector] = lambda: Mock(0.52)
    assert post().json()["label"] == "inconclusive"
    app.dependency_overrides.clear()
def test_bad_extension(): assert c.post("/api/v1/detect/image", files={"file": ("a.exe", b"x", "x")}).status_code == 415
def test_signature_mismatch(): assert c.post("/api/v1/detect/image", files={"file": ("a.png", b"notpng!!", "image/png")}).json()["code"] == "signature_mismatch"
def test_empty_text(): assert c.post("/api/v1/detect/text", json={"text": "  "}).json()["code"] == "text_too_short"
def test_text_pending_503(): assert c.post("/api/v1/detect/text", json={"text": "x" * 80}).status_code == 503
def test_video_pending_503():
    r = c.post("/api/v1/detect/video", files={"file": ("a.mp4", b"\x00\x00\x00\x18ftypmp42" + b"0" * 20, "video/mp4")})
    assert r.status_code == 503
