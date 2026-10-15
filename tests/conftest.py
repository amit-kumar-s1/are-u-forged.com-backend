import pytest
from app.core import limits
from app.core.config import get_settings

@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    s = get_settings()
    monkeypatch.setattr(s, "temp_upload_dir", str(tmp_path / "up"))
    monkeypatch.setattr(s, "malware_scan_mode", "off")
    limits.reset_state(); yield; limits.reset_state()
