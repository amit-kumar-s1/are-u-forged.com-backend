"""ClamAV (clamd INSTREAM) client. Streams the file in chunks; nothing is sent to external services."""
import logging, socket, struct, time
from dataclasses import dataclass
from starlette.concurrency import run_in_threadpool
from app.core.config import get_settings
from app.utils.validation import err

log = logging.getLogger("aryouforged.scan")

class ScannerError(Exception): pass

@dataclass
class ScanOutcome:
    infected: bool
    signature: str | None = None

class ClamdScanner:
    def __init__(self, host, port, socket_path, timeout):
        self.host, self.port, self.socket_path, self.timeout = host, port, socket_path, timeout
    def _connect(self):
        if self.socket_path:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(self.timeout); s.connect(self.socket_path); return s
        return socket.create_connection((self.host, self.port), timeout=self.timeout)
    def scan(self, path: str) -> ScanOutcome:
        deadline, resp = time.monotonic() + self.timeout, b""
        try:
            with self._connect() as s, open(path, "rb") as f:
                s.settimeout(self.timeout)
                s.sendall(b"zINSTREAM\0")
                while chunk := f.read(64 * 1024):
                    if time.monotonic() > deadline: raise ScannerError("scan timed out")
                    s.sendall(struct.pack("!I", len(chunk)) + chunk)
                s.sendall(struct.pack("!I", 0))
                while not resp.endswith(b"\0") and b"\n" not in resp:
                    part = s.recv(4096)
                    if not part: break
                    resp += part
        except OSError as e:  # includes timeouts and refused connections
            raise ScannerError(f"scanner I/O error: {type(e).__name__}") from e
        text = resp.strip(b"\0\n ").decode("utf-8", "replace")
        if text.endswith("OK"): return ScanOutcome(False)
        if text.endswith("FOUND"): return ScanOutcome(True, text.split(":", 1)[-1].rsplit(" FOUND", 1)[0].strip())
        raise ScannerError("unexpected scanner response")  # includes "size limit exceeded ... ERROR"

def get_scanner() -> ClamdScanner:
    s = get_settings()
    return ClamdScanner(s.clamd_host, s.clamd_port, s.clamd_socket, s.scan_timeout_s)

async def scan_upload(path: str) -> str:
    """Returns "clean" or "not_performed". Never returns "clean" unless a scan actually completed."""
    mode = get_settings().malware_scan_mode
    if mode == "off": return "not_performed"
    try:
        out = await run_in_threadpool(get_scanner().scan, path)
    except ScannerError as e:
        log.error("malware scanner failure (mode=%s): %s", mode, e)
        if mode == "required": raise err(503, "scanner_unavailable", "Upload scanning is temporarily unavailable. Try again later.")
        return "not_performed"
    if out.infected:
        log.warning("malware detected: signature=%s", out.signature)
        raise err(422, "malware_detected", "This file was rejected by the malware scanner.")
    return "clean"
