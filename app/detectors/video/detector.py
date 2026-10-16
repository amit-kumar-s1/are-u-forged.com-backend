
import time

import torch

from app.detectors.base import (
    Detector,
    DetectorUnavailable,
    RawPrediction,
)
from app.detectors.video.config import video_cfg
from app.detectors.video.model_loader import get_video_model
from app.detectors.video.preprocessing import preprocess_video


class VideoDetector(Detector):
    modality = "video"

    def predict(self, payload) -> RawPrediction:
        started = time.perf_counter()

        try:
            model, device = get_video_model()
            frames = preprocess_video(str(payload)).to(device)

            with torch.inference_mode():
                logits = model(frames)
                fake_probability = float(
                    torch.sigmoid(logits).reshape(-1)[0].item()
                )

            fake_probability = max(
                0.0, min(1.0, fake_probability)
            )

            return RawPrediction(
                fake_probability=fake_probability,
                model="EfficientNetV2-S + Transformer",
                details={
                    "threshold": video_cfg.threshold,
                    "frames_used": int(frames.shape[1]),
                    "processing_time_seconds": round(
                        time.perf_counter() - started, 3
                    ),
                    "device": str(device),
                },
            )

        except DetectorUnavailable:
            raise
        except (ValueError, FileNotFoundError) as exc:
            raise DetectorUnavailable(
                "The video could not be processed or the video model is unavailable."
            ) from exc
