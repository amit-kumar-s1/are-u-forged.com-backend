
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application / CORS
    cors_origins: str = "http://localhost:3000"
    docs_enabled: bool = False
    enable_hsts: bool = False

    # Upload size limits
    max_image_mb: int = Field(default=10, ge=1, le=50)
    max_video_mb: int = Field(default=100, ge=1, le=500)
    max_text_chars: int = Field(default=20_000, ge=50, le=100_000)
    min_text_chars: int = Field(default=50, ge=1)

    # Image safety
    max_image_pixels: int = Field(default=40_000_000, ge=1)
    max_image_dimension: int = Field(default=10_000, ge=1)
    image_normalize_max_side: int = Field(default=2048, ge=1)

    # Temporary uploads
    temp_upload_dir: str = ""

    # Rate limiting
    rate_limit_image_per_min: int = Field(default=20, ge=1)
    rate_limit_video_per_min: int = Field(default=5, ge=1)
    rate_limit_text_per_min: int = Field(default=30, ge=1)
    rate_limit_window_s: int = Field(default=60, ge=1)

    rate_limit_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = Field(default="", repr=False)

    # Trust forwarding headers only when deployment is verified
    trusted_proxy_count: int = Field(default=0, ge=0, le=10)

    # Concurrency and timeouts
    max_concurrent_uploads: int = Field(default=4, ge=1)
    max_concurrent_image_inference: int = Field(default=2, ge=1)
    max_concurrent_video_inference: int = Field(default=1, ge=1)
    max_concurrent_text_inference: int = Field(default=2, ge=1)

    queue_wait_timeout_s: float = Field(default=5.0, gt=0)
    inference_timeout_image_s: float = Field(default=30, gt=0)
    inference_timeout_video_s: float = Field(default=300, gt=0)
    inference_timeout_text_s: float = Field(default=30, gt=0)

    # Malware scanning
    malware_scan_mode: Literal["off", "optional", "required"] = "off"
    clamd_host: str = "127.0.0.1"
    clamd_port: int = Field(default=3310, ge=1, le=65535)
    clamd_socket: str = ""
    scan_timeout_s: float = Field(default=30, gt=0)

    # Model configuration
    inconclusive_band: float = Field(default=0.10, ge=0, le=1)

    hf_model_repo: str = (
        "not-human-s1/DeepTrust-AI-DeepFake-Image-Detection-Model"
    )

    image_fake_threshold: float = Field(default=0.40, ge=0, le=1)
    device: str = "cpu"
    image_model_path: str = ""
    image_model_name: str = "Image model"
    image_fake_class_index: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.min_text_chars > self.max_text_chars:
            raise ValueError(
                "min_text_chars cannot exceed max_text_chars"
            )

        if self.rate_limit_backend == "redis" and not self.redis_url:
            raise ValueError(
                "REDIS_URL is required when RATE_LIMIT_BACKEND=redis"
            )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
