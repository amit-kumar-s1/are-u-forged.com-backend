
import logging
from functools import lru_cache

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from insightface.model_zoo import SCRFD

from app.detectors.video.config import preprocess_cfg, video_cfg

logger = logging.getLogger(__name__)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@lru_cache(maxsize=1)
def get_face_detector():
    model_path = video_cfg.scrfd_model_path

    import os
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "SCRFD model file is missing: " + model_path
        )

    detector = SCRFD(model_file=model_path)
    detector.prepare(
        ctx_id=-1,
        input_size=preprocess_cfg.scrfd_input_size,
        det_thresh=0.5,
    )
    return detector


def sample_frames(video_path: str, count: int):
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            raise ValueError("Unable to open this video.")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("The video contains no readable frames.")

        indices = np.linspace(
            0, total - 1, count, dtype=int
        )

        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)

        if not frames:
            raise ValueError("No frames could be extracted.")

        return frames

    finally:
        cap.release()


def _detect_largest_face(frame):
    detector = get_face_detector()

    boxes, _ = detector.detect(
        frame,
        input_size=preprocess_cfg.scrfd_input_size,
        max_num=0,
        metric="default",
    )

    if boxes is None or len(boxes) == 0:
        return None

    valid = [
        box for box in boxes
        if float(box[4]) >= preprocess_cfg.face_conf_thresh
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda box: max(0, box[2] - box[0])
        * max(0, box[3] - box[1]),
    )


def _crop_face(frame, box):
    x1, y1, x2, y2 = map(float, box[:4])

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = (x2 - x1) * preprocess_cfg.face_scale_factor
    height = (y2 - y1) * preprocess_cfg.face_scale_factor

    h, w = frame.shape[:2]

    left = max(0, int(cx - width / 2))
    top = max(0, int(cy - height / 2))
    right = min(w, int(cx + width / 2))
    bottom = min(h, int(cy + height / 2))

    if right <= left or bottom <= top:
        return None

    crop = frame[top:bottom, left:right]
    return cv2.resize(
        crop,
        (preprocess_cfg.face_size, preprocess_cfg.face_size),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_video(video_path: str):
    frames = sample_frames(
        video_path,
        preprocess_cfg.frames_per_video,
    )

    processed = []

    for frame in frames:
        face = _detect_largest_face(frame)

        if face is None:
            crop = cv2.resize(
                frame,
                (preprocess_cfg.face_size, preprocess_cfg.face_size),
            )
        else:
            crop = _crop_face(frame, face)

            if crop is None:
                crop = cv2.resize(
                    frame,
                    (preprocess_cfg.face_size, preprocess_cfg.face_size),
                )

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        processed.append(_transform(Image.fromarray(crop)))

    return torch.stack(processed).unsqueeze(0)
