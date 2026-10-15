import asyncio, io, os, socket, threading
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.api import routes
from app.core import limits
from app.core.config import get_settings
from app.detectors.base import Detector, RawPrediction
from app.services import scanner as sc
from app.services.uploads import received_upload

c = TestClient(app)
S = get_settings()

class Mock(Detector):
    modality = "image"; calls = 0
    def predict(self, payload):
        Mock.calls += 1; return RawPrediction(0.9, "mock-test-model")
class Boom(Detector):
    modality = "image"
    def predict(self, payload): raise RuntimeError("secret internal detail")

def img(fmt="PNG", size=(8, 8)):
    b = io.BytesIO(); Image.new("RGB", size).save(b, fmt); return b.getvalue()
def post(data=None, name="a.png"): return c.post("/api/v1/detect/image", files={"file": (name, data or img(), "application/octet-stream")})
def leftovers(): return os.listdir(S.temp_upload_dir) if os.path.isdir(S.temp_upload_dir) else []
@pytest.fixture
def mock(): Mock.calls = 0; app.dependency_overrides[routes.image_detector] = lambda: Mock(); yield; app.dependency_overrides.clear()

# ---- rate limiting ----
def test_text_rate_limit(monkeypatch):
    monkeypatch.setattr(S, "rate_limit_text_per_min", 3)
    codes = [c.post("/api/v1/detect/text", json={"text": "x" * 80}).status_code for _ in range(4)]
    assert codes[:3] == [503, 503, 503] and codes[3] == 429  # 503 = model pending, i.e. request was allowed through
    r = c.post("/api/v1/detect/text", json={"text": "x" * 80}); assert r.json()["code"] == "rate_limited" and int(r.headers["retry-after"]) >= 1
def test_upload_rate_limit_applies_to_uploads(monkeypatch, mock):
    monkeypatch.setattr(S, "rate_limit_image_per_min", 2)
    assert [post().status_code for _ in range(3)] == [200, 200, 429]
    assert c.post("/api/v1/detect/text", json={"text": "x" * 80}).status_code != 429  # endpoints limited separately
def test_rate_limit_rejects_before_parsing_body(monkeypatch):
    monkeypatch.setattr(S, "rate_limit_image_per_min", 0)
    assert post().status_code == 429 and leftovers() == []
def test_xff_ignored_unless_trusted(monkeypatch):
    sc_ = lambda h: {"headers": [(b"x-forwarded-for", h)], "client": ("9.9.9.9", 1)}
    assert limits.client_ip(sc_(b"1.1.1.1")) == "9.9.9.9"
    monkeypatch.setattr(S, "trusted_proxy_count", 1); assert limits.client_ip(sc_(b"6.6.6.6, 1.1.1.1")) == "1.1.1.1"  # spoofed left entry ignored
    monkeypatch.setattr(S, "trusted_proxy_count", 2); assert limits.client_ip(sc_(b"6.6.6.6, 1.1.1.1")) == "6.6.6.6"
def test_redis_limiter_shared_between_simulated_workers():
    class Fake:  # stands in for Redis; two limiter objects = two workers sharing one store
        d = {}
        def incr(self, k): self.d[k] = self.d.get(k, 0) + 1; return self.d[k]
        def expire(self, k, t): pass
    store = Fake(); w1, w2 = limits.RedisLimiter(store), limits.RedisLimiter(store)
    assert [w1.hit("k", 3, 3600)[0], w2.hit("k", 3, 3600)[0], w1.hit("k", 3, 3600)[0], w2.hit("k", 3, 3600)[0]] == [True, True, True, False]
def test_redis_failure_falls_back_to_memory(monkeypatch):
    class Dead:
        def hit(self, *a): raise ConnectionError()
    limits._limiter = Dead(); monkeypatch.setattr(S, "rate_limit_text_per_min", 1)
    assert limits.check_rate("text", "ip")[0] and not limits.check_rate("text", "ip")[0]

# ---- security headers ----
def test_headers_on_api_and_errors():
    for r in (c.get("/health"), c.post("/api/v1/detect/text", json={"text": ""})):
        h = r.headers
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "strict-origin-when-cross-origin" and "camera=()" in h["permissions-policy"]
        assert "frame-ancestors 'none'" in h["content-security-policy"] and "strict-transport-security" not in h
def test_hsts_only_when_enabled_and_docs_still_work(monkeypatch):
    monkeypatch.setattr(S, "enable_hsts", True); assert "max-age" in c.get("/health").headers["strict-transport-security"]
    r = c.get("/docs"); assert r.status_code == 200 and "content-security-policy" not in r.headers and r.headers["x-content-type-options"] == "nosniff"
def test_429_carries_cors_and_headers(monkeypatch):
    monkeypatch.setattr(S, "rate_limit_text_per_min", 0)
    r = c.post("/api/v1/detect/text", json={"text": "x" * 80}, headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 429 and r.headers["access-control-allow-origin"] == "http://localhost:3000" and r.headers["x-frame-options"] == "DENY"
def test_unhandled_error_is_generic(mock):
    app.dependency_overrides[routes.image_detector] = lambda: Boom()
    r = post(); assert r.status_code == 500 and "secret" not in r.text and leftovers() == []

# ---- malware scanning (mock scanners; no real malware) ----
class FakeScanner:
    def __init__(self, result=None, exc=None): self.result, self.exc, self.seen = result, exc, []
    def scan(self, path):
        self.seen.append(os.path.exists(path))
        if self.exc: raise self.exc
        return self.result
def use(monkeypatch, scanner, mode="required"):
    monkeypatch.setattr(S, "malware_scan_mode", mode); monkeypatch.setattr(sc, "get_scanner", lambda: scanner)
def test_clean_scan_is_invoked_and_reported(monkeypatch, mock):
    f = FakeScanner(sc.ScanOutcome(False)); use(monkeypatch, f)
    r = post(); assert r.status_code == 200 and r.json()["details"]["malware_scan"] == "clean" and f.seen == [True] and leftovers() == []
def test_infected_rejected_before_inference(monkeypatch, mock):
    use(monkeypatch, FakeScanner(sc.ScanOutcome(True, "Test-Signature")))
    r = post(); assert r.status_code == 422 and r.json()["code"] == "malware_detected" and Mock.calls == 0 and leftovers() == []
def test_required_mode_fails_closed(monkeypatch, mock):
    use(monkeypatch, FakeScanner(exc=sc.ScannerError("down")), "required")
    r = post(); assert r.status_code == 503 and r.json()["code"] == "scanner_unavailable" and Mock.calls == 0 and leftovers() == []
def test_optional_mode_never_claims_clean(monkeypatch, mock):
    use(monkeypatch, FakeScanner(exc=sc.ScannerError("down")), "optional")
    r = post(); assert r.status_code == 200 and r.json()["details"]["malware_scan"] == "not_performed"
def test_off_mode_reports_not_performed(mock): assert post().json()["details"]["malware_scan"] == "not_performed"
def test_video_is_scanned_too(monkeypatch):
    f = FakeScanner(sc.ScanOutcome(True, "X")); use(monkeypatch, f)
    r = c.post("/api/v1/detect/video", files={"file": ("a.mp4", b"\x00\x00\x00\x18ftypmp42" + b"0" * 20, "video/mp4")})
    assert r.status_code == 422 and f.seen == [True] and leftovers() == []

def fake_clamd(reply, hang=False):
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1); got = {}
    def run():
        conn, _ = srv.accept(); buf = b""
        while not buf.endswith(b"\x00\x00\x00\x00"):
            d = conn.recv(65536)
            if not d: break
            buf += d
        got["buf"] = buf
        if not hang: conn.sendall(reply)
        else: threading.Event().wait(2)
        conn.close(); srv.close()
    threading.Thread(target=run, daemon=True).start(); return srv.getsockname()[1], got
def test_clamd_protocol_clean_infected_and_timeout(tmp_path):
    p = tmp_path / "f.bin"; p.write_bytes(b"hello")
    port, got = fake_clamd(b"stream: OK\0")
    assert not sc.ClamdScanner("127.0.0.1", port, "", 2).scan(str(p)).infected and got["buf"].startswith(b"zINSTREAM\0")
    port, _ = fake_clamd(b"stream: Eicar-Test-Signature FOUND\0")
    o = sc.ClamdScanner("127.0.0.1", port, "", 2).scan(str(p)); assert o.infected and o.signature == "Eicar-Test-Signature"
    port, _ = fake_clamd(b"", hang=True)
    with pytest.raises(sc.ScannerError): sc.ClamdScanner("127.0.0.1", port, "", 0.3).scan(str(p))
    port, _ = fake_clamd(b"INSTREAM size limit exceeded. ERROR\0")
    with pytest.raises(sc.ScannerError): sc.ClamdScanner("127.0.0.1", port, "", 2).scan(str(p))
def test_scanner_unreachable_is_error(tmp_path):
    p = tmp_path / "f"; p.write_bytes(b"x")
    with pytest.raises(sc.ScannerError): sc.ClamdScanner("127.0.0.1", 1, "", 0.5).scan(str(p))

# ---- image protection ----
def test_valid_formats_accepted(mock):
    for fmt, name in (("PNG", "a.png"), ("JPEG", "a.jpg"), ("WEBP", "a.webp")): assert post(img(fmt), name).status_code == 200
def test_malformed_and_truncated_rejected(mock):
    assert post(b"\x89PNG\r\n\x1a\n" + b"junk" * 50).json()["code"] == "invalid_image"
    assert post(img(size=(300, 300))[:-40]).json()["code"] == "invalid_image"
    assert Mock.calls == 0 and leftovers() == []
def test_extension_cannot_bypass_validation(mock):
    r = post(img("JPEG"), "a.png"); assert r.status_code == 415 and r.json()["code"] == "signature_mismatch"
    assert post(b"GIF89a" + b"0" * 30).status_code == 415
def test_dimension_and_pixel_limits(monkeypatch, mock):
    monkeypatch.setattr(S, "max_image_dimension", 16); r = post(img(size=(32, 8))); assert r.status_code == 413 and r.json()["code"] == "image_dimensions_exceeded"
    monkeypatch.setattr(S, "max_image_dimension", 10000); monkeypatch.setattr(S, "max_image_pixels", 100)
    assert post(img(size=(32, 32))).json()["code"] == "image_pixels_exceeded"
def test_huge_dimension_bomb_rejected_before_decode(mock):
    b = io.BytesIO(); Image.new("1", (12000, 12000)).save(b, "PNG")  # tiny file, 144M pixels
    assert len(b.getvalue()) < 1_000_000 and post(b.getvalue()).status_code == 413 and Mock.calls == 0
def test_pillow_bomb_protection_is_active(monkeypatch, mock):
    assert Image.MAX_IMAGE_PIXELS is not None
    monkeypatch.setattr(S, "max_image_pixels", 10**9); monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # Pillow's own guard, our check passes
    assert post(img(size=(32, 32))).status_code == 413
def test_images_are_normalised(monkeypatch):
    monkeypatch.setattr(S, "image_normalize_max_side", 64); seen = {}
    class Cap(Mock):
        def predict(self, payload): seen["im"] = Image.open(io.BytesIO(payload)).size; return super().predict(payload)
    app.dependency_overrides[routes.image_detector] = lambda: Cap(); post(img(size=(256, 128))); app.dependency_overrides.clear()
    assert seen["im"] == (64, 32)

# ---- upload / memory handling ----
def test_oversize_rejected_via_content_length(monkeypatch, mock):
    monkeypatch.setattr(S, "max_image_mb", 1); assert post(b"\x89PNG\r\n\x1a\n" + b"0" * 3_000_000).json()["code"] == "request_too_large"
def test_oversize_chunked_without_content_length_rejected_mid_stream(monkeypatch, mock):
    monkeypatch.setattr(S, "max_image_mb", 1); sent = []
    def gen():
        yield b'--B\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n\r\n\x89PNG\r\n\x1a\n'
        for _ in range(40): sent.append(1); yield b"0" * 100_000
        yield b"\r\n--B--\r\n"
    r = c.post("/api/v1/detect/image", content=gen(), headers={"content-type": "multipart/form-data; boundary=B"})
    assert r.status_code == 413 and r.json()["code"] == "request_too_large" and leftovers() == []
def test_lying_content_length_does_not_bypass(monkeypatch, mock):
    monkeypatch.setattr(S, "max_image_mb", 1)
    r = c.post("/api/v1/detect/image", content=b"0" * 3_000_000, headers={"content-type": "multipart/form-data; boundary=B", "content-length": "100"})
    assert r.status_code in (413, 422, 400)  # httpx may correct the header; either way it is never processed
    assert Mock.calls == 0
def test_text_body_limit(monkeypatch):
    monkeypatch.setattr(S, "max_text_chars", 100); assert c.post("/api/v1/detect/text", json={"text": "x" * 5000}).status_code in (413,)
def test_gate_bounds_concurrency():
    g = limits.Gate(2); assert [g.acquire_nowait() for _ in range(3)] == [True, True, False]; g.release(); assert g.acquire_nowait()
    assert asyncio.run(limits.Gate(0).acquire(0.1)) is False
def test_upload_concurrency_limit_sheds_load(monkeypatch, mock):
    monkeypatch.setattr(S, "max_concurrent_uploads", 1); limits.reset_state(); assert limits.get_gate("upload").acquire_nowait()
    r = post(); assert r.status_code == 503 and r.json()["code"] == "server_busy" and r.headers["retry-after"] == "5"
def test_inference_concurrency_limit_sheds_load(monkeypatch, mock):
    monkeypatch.setattr(S, "max_concurrent_image_inference", 1); monkeypatch.setattr(S, "queue_wait_timeout_s", 0.1)
    limits.reset_state(); assert limits.get_gate("image").acquire_nowait()
    assert post().status_code == 503 and leftovers() == []
def test_inference_timeout_returns_504(monkeypatch):
    import time
    class Slow(Mock):
        def predict(self, p): time.sleep(0.6); return super().predict(p)
    monkeypatch.setattr(S, "inference_timeout_image_s", 0.1); app.dependency_overrides[routes.image_detector] = lambda: Slow()
    r = post(); app.dependency_overrides.clear(); assert r.status_code == 504 and leftovers() == []
def test_temp_file_removed_on_cancellation_and_errors():
    class FU:
        filename = "a.png"
        def __init__(self): self.d = [b"\x89PNG\r\n\x1a\n" + b"0" * 10, b""]
        async def read(self, n): return self.d.pop(0)
        async def close(self): pass
    async def cancelled():
        async with received_upload(FU(), "image", 1) as (p, _, __): assert os.path.exists(p); raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError): asyncio.run(cancelled())
    async def too_big():
        class Big(FU):
            async def read(self, n): return b"0" * 2_000_000
        async with received_upload(Big(), "image", 1): pass
    with pytest.raises(HTTPException): asyncio.run(too_big())
    assert leftovers() == []
def test_temp_files_are_private_and_unpredictable():
    class FU:
        filename = "../../etc/passwd.png"
        def __init__(self): self.d = [b"\x89PNG\r\n\x1a\n" + b"0" * 10, b""]
        async def read(self, n): return self.d.pop(0)
        async def close(self): pass
    async def go():
        async with received_upload(FU(), "image", 1) as (p, _, __):
            assert os.path.dirname(p) == S.temp_upload_dir and "passwd" not in p and (os.stat(p).st_mode & 0o777) == 0o600
    asyncio.run(go())
