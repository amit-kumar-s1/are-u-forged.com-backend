
import logging
import os
from threading import Lock

import torch
from huggingface_hub import hf_hub_download

from app.detectors.base import DetectorUnavailable
from app.detectors.video.config import model_cfg, video_cfg
from app.detectors.video.model import DeepfakeDetector

logger = logging.getLogger(__name__)

_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_model = None
_load_lock = Lock()


def get_video_model():
    global _model

    if _model is not None:
        return _model, _device

    with _load_lock:
        if _model is not None:
            return _model, _device

        try:
            token = os.getenv("HF_TOKEN")

            checkpoint_path = hf_hub_download(
                repo_id=video_cfg.model_repo,
                filename=video_cfg.model_filename,
                token=token or None,
            )

            model = DeepfakeDetector(model_cfg)

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                else:
                    state_dict = checkpoint
            else:
                raise ValueError("Unexpected checkpoint format.")

            model.load_state_dict(state_dict, strict=True)
            model.to(_device)
            model.eval()

            _model = model
            logger.info("Video detection model loaded on %s", _device)

        except Exception as exc:
            logger.exception("Video model initialization failed")
            raise DetectorUnavailable(
                "Video detection model is not configured or could not be loaded."
            ) from exc

    return _model, _device
