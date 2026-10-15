"""Rate limiting (memory or Redis), trusted-proxy client IP, and non-blocking concurrency gates."""
import asyncio, collections, logging, threading, time
from app.core.config import get_settings

log = logging.getLogger("aryouforged.limits")

class Gate:
    """Counting gate that works across event loops (no asyncio.Semaphore binding)."""
    def __init__(self, limit: int): self.limit, self.n, self._l = limit, 0, threading.Lock()
    def acquire_nowait(self) -> bool:
        with self._l:
            if self.n < self.limit: self.n += 1; return True
            return False
    def release(self):
        with self._l: self.n = max(0, self.n - 1)
    async def acquire(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while not self.acquire_nowait():
            if time.monotonic() >= end: return False
            await asyncio.sleep(0.05)
        return True

class MemoryLimiter:
    """Sliding window, per process. Not shared between workers/instances."""
    def __init__(self): self.hits, self._l = collections.defaultdict(collections.deque), threading.Lock()
    def hit(self, key: str, limit: int, window: int):
        now = time.monotonic()
        with self._l:
            q = self.hits[key]
            while q and q[0] <= now - window: q.popleft()
            if len(q) >= limit: return False, (int(window - (now - q[0])) + 1 if q else window)
            q.append(now); return True, 0

class RedisLimiter:
    """Fixed window using INCR/EXPIRE; shared by every worker and instance using the same Redis."""
    def __init__(self, client): self.c = client
    def hit(self, key: str, limit: int, window: int):
        now = time.time(); k = f"ayf:rl:{key}:{int(now // window)}"
        n = self.c.incr(k)
        if n == 1: self.c.expire(k, window + 1)
        return n <= limit, max(1, int(window - now % window))

_limiter = None; _fallback = MemoryLimiter(); _gates: dict[str, Gate] = {}

def _primary():
    global _limiter
    if _limiter is None:
        s = get_settings()
        if s.rate_limit_backend == "redis":
            import redis
            _limiter = RedisLimiter(redis.Redis.from_url(s.redis_url, socket_timeout=1, socket_connect_timeout=1))
        else: _limiter = MemoryLimiter()
    return _limiter

def check_rate(kind: str, ip: str):
    s = get_settings(); limit = getattr(s, f"rate_limit_{kind}_per_min"); key = f"{kind}:{ip}"
    try: return _primary().hit(key, limit, s.rate_limit_window_s)
    except Exception:  # Redis down: degrade to per-process limiting rather than open or dead
        log.warning("rate-limit backend failed; using in-process fallback", exc_info=True)
        return _fallback.hit(key, limit, s.rate_limit_window_s)

def client_ip(scope) -> str:
    """X-Forwarded-For is honoured ONLY for the configured number of trusted proxies, read from the right."""
    n = get_settings().trusted_proxy_count
    if n > 0:
        xff = dict(scope["headers"]).get(b"x-forwarded-for")
        if xff:
            parts = [p.strip() for p in xff.decode("latin-1").split(",") if p.strip()]
            if len(parts) >= n: return parts[-n]
    c = scope.get("client"); return c[0] if c else "unknown"

def get_gate(name: str) -> Gate:
    if name not in _gates:
        s = get_settings()
        _gates[name] = Gate(s.max_concurrent_uploads if name == "upload" else getattr(s, f"max_concurrent_{name}_inference"))
    return _gates[name]

def reset_state():
    global _limiter; _limiter = None; _gates.clear(); _fallback.hits.clear()
