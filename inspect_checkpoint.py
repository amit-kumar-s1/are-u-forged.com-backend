
import os
import torch
from huggingface_hub import hf_hub_download

REPO = "not-human-s1/DeepTrust-AI-DeepFake-Image-Detection-Model"

FILES = [
    "deepfake_image.pth",
    "best_model_v2.pth",
    "Gravex200_best_model.pth",
    "best_model.pth",
]

token = os.getenv("HF_TOKEN")

for filename in FILES:
    print(f"\n{'=' * 60}")
    print(f"CHECKPOINT: {filename}")

    try:
        path = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            token=token,
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        state = checkpoint

        if isinstance(checkpoint, dict):
            for wrapper in ("model_state_dict", "state_dict"):
                if isinstance(checkpoint.get(wrapper), dict):
                    state = checkpoint[wrapper]
                    print("Wrapper:", wrapper)
                    break

            print("Top-level keys:", list(checkpoint.keys())[:10])

        if not isinstance(state, dict):
            print("Unexpected checkpoint type:", type(state))
            continue

        print("First parameter keys and shapes:")

        for key, value in list(state.items())[:12]:
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value)
            print(f"  {key}: {shape}")

        print("\nArchitecture clues:")
        for key in (
            "conv1.weight",
            "layer1.0.conv1.weight",
            "fc.weight",
            "features.0.0.weight",
            "classifier.1.weight",
        ):
            if key in state:
                print(f"  {key}: {tuple(state[key].shape)}")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
