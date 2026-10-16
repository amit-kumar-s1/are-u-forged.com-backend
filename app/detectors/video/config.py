
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    frames_per_video: int = 30
    face_size: int = 224
    face_conf_thresh: float = 0.85
    face_scale_factor: float = 1.3
    scrfd_input_size: tuple[int, int] = (640, 640)
    use_gpu: bool = False


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "efficientnet_v2_s"
    backbone_pretrained: bool = False
    backbone_out_dim: int = 1280

    transformer_d_model: int = 512
    transformer_nhead: int = 8
    transformer_layers: int = 4
    transformer_dim_ff: int = 2048
    transformer_dropout: float = 0.1

    attn_pool_hidden: int = 256
    classifier_dropout: float = 0.4
    num_classes: int = 1

    frames_per_clip: int = 30
    img_size: int = 224


@dataclass(frozen=True)
class VideoConfig:
    model_repo: str = os.getenv(
        "VIDEO_MODEL_REPO",
        "not-human-s1/DeepTrustAI-Video-Model",
    )
    model_filename: str = os.getenv(
        "VIDEO_MODEL_FILENAME",
        "best_model_video.pth",
    )
    threshold: float = float(
        os.getenv("VIDEO_THRESHOLD", "0.5")
    )
    scrfd_model_path: str = os.getenv(
        "VIDEO_SCRFD_MODEL_PATH",
        "app/detectors/video/scrfd_models/scrfd_10g_bnkps.onnx",
    )


preprocess_cfg = PreprocessConfig()
model_cfg = ModelConfig()
video_cfg = VideoConfig()
