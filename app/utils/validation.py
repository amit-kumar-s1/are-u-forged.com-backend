from fastapi import HTTPException

SIGNATURES = {
    "image": {".jpg": [b"\xff\xd8\xff"], ".jpeg": [b"\xff\xd8\xff"], ".png": [b"\x89PNG\r\n\x1a\n"], ".webp": [b"RIFF"]},
    "video": {".mp4": [b"ftyp"], ".mov": [b"ftyp", b"moov", b"wide", b"mdat"], ".webm": [b"\x1a\x45\xdf\xa3"]},
}

def err(status: int, code: str, message: str, headers: dict | None = None):
    return HTTPException(status_code=status, detail={"code": code, "message": message}, headers=headers)

def check_signature(head: bytes, ext: str, kind: str):
    sigs = SIGNATURES[kind][ext]
    ok = any(s in head for s in sigs) if kind == "video" else any(head.startswith(s) for s in sigs)
    if ext == ".webp": ok = ok and head[8:12] == b"WEBP"
    if not ok: raise err(415, "signature_mismatch", "File contents do not match its extension.")
