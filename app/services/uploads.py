"""Chunked upload -> private temp file with a hard byte counter. Always deleted afterwards."""
import logging, os, tempfile
from contextlib import asynccontextmanager
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool
from app.core.config import get_settings
from app.utils.validation import SIGNATURES, check_signature, err

log = logging.getLogger("aryouforged.upload")
CHUNK = 1024 * 1024

def temp_dir() -> str:
    d = get_settings().temp_upload_dir or os.path.join(tempfile.gettempdir(), "ayf-uploads")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d

@asynccontextmanager
async def received_upload(file: UploadFile, kind: str, max_mb: int):
    name = (file.filename or "").lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in SIGNATURES[kind]:
        raise err(415, "unsupported_type", f"Unsupported file type. Allowed: {', '.join(SIGNATURES[kind])}.")
    limit = max_mb * 1048576
    fd, path = tempfile.mkstemp(suffix=ext, prefix="up-", dir=temp_dir())  # 0600, unpredictable; client filename never used
    try:
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > limit: raise err(413, "file_too_large", f"File exceeds the {max_mb} MB limit.")
                await run_in_threadpool(out.write, chunk)
        if size == 0: raise err(400, "empty_file", "The uploaded file is empty.")
        with open(path, "rb") as f: head = f.read(16)
        check_signature(head, ext, kind)
        yield path, ext, size
    finally:  # runs on success, validation failure, exceptions and task cancellation
        try: os.unlink(path)
        except FileNotFoundError: pass
        except OSError: log.error("failed to delete temp upload")
        try: await file.close()
        except Exception: pass
