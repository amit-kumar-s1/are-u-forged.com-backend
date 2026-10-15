import asyncio, logging, time
from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool
from app.core import limits
from app.core.config import get_settings
from app.detectors.base import Detector, DetectorUnavailable, to_label
from app.detectors.registry import get_detector
from app.schemas.detection import DetectionResult, TextRequest
from app.services.images import normalize_image
from app.services.scanner import scan_upload
from app.services.uploads import received_upload
from app.utils.validation import err

log = logging.getLogger("aryouforged")
router = APIRouter(prefix="/api/v1")

def image_detector() -> Detector: return get_detector("image")
def video_detector() -> Detector: return get_detector("video")
def text_detector() -> Detector: return get_detector("text")

async def _run(det: Detector, payload, extra: dict) -> DetectionResult:
    s, kind = get_settings(), det.modality
    gate = limits.get_gate(kind)
    if not await gate.acquire(s.queue_wait_timeout_s):  # bounded wait, then shed load
        raise err(503, "server_busy", "The analysis service is busy. Please try again shortly.", {"Retry-After": "10"})
    t0 = time.perf_counter()
    fut = asyncio.ensure_future(run_in_threadpool(det.predict, payload))
    def done(f):  # slot is freed only when the worker thread really finishes (threads can't be killed)
        gate.release()
        if not f.cancelled(): f.exception()
    fut.add_done_callback(done)
    try:
        raw = await asyncio.wait_for(asyncio.shield(fut), getattr(s, f"inference_timeout_{kind}_s"))
    except asyncio.TimeoutError:
        raise err(504, "inference_timeout", "The analysis took too long and was stopped.")
    except DetectorUnavailable as e:
        raise err(503, "model_not_configured", str(e))
    except Exception:
        log.exception("inference failed (%s)", kind)  # never log user content
        raise err(500, "inference_failed", "The model failed while analysing this content.")
    label, conf = to_label(raw.fake_probability)
    return DetectionResult(modality=kind, label=label, fake_probability=raw.fake_probability, confidence=conf,
                           model=raw.model, processing_ms=int((time.perf_counter() - t0) * 1000), details={**raw.details, **extra})

@router.get("/health")
def health(): return {"status": "ok"}

@router.post("/detect/image", response_model=DetectionResult)
async def detect_image(file: UploadFile = File(...), det: Detector = Depends(image_detector)):
    async with received_upload(file, "image", get_settings().max_image_mb) as (path, ext, size):
        scan = await scan_upload(path)
        data, info = await run_in_threadpool(normalize_image, path, ext)
        return await _run(det, data, {"bytes": size, "format": ext.lstrip("."), "malware_scan": scan, **info})

@router.post("/detect/video", response_model=DetectionResult)
async def detect_video(file: UploadFile = File(...), det: Detector = Depends(video_detector)):
    async with received_upload(file, "video", get_settings().max_video_mb) as (path, ext, size):
        scan = await scan_upload(path)
        return await _run(det, path, {"bytes": size, "format": ext.lstrip("."), "malware_scan": scan})

@router.post("/detect/text", response_model=DetectionResult)
async def detect_text(body: TextRequest, det: Detector = Depends(text_detector)):
    s, n = get_settings(), len(body.text.strip())
    if n < s.min_text_chars: raise err(422, "text_too_short", f"Enter at least {s.min_text_chars} characters.")
    if n > s.max_text_chars: raise err(413, "text_too_long", f"Text exceeds {s.max_text_chars} characters.")
    return await _run(det, body.text, {"characters": n})
