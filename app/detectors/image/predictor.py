
import torch


def predict(models, tensors, device):
    x224 = tensors["224"].unsqueeze(0).to(device)
    x380 = tensors["380"].unsqueeze(0).to(device)
    x518 = tensors["518"].unsqueeze(0).to(device)

    with torch.inference_mode():
        # Preserve the prediction logic from the old backend.
        resnet_logits = models["resnet18"](x224)
        resnet_fake = 1.0 - torch.softmax(
            resnet_logits, dim=1
        )[:, 1]

        b4_logits = models["efficientnet_b4"](x380)
        b4_fake = torch.sigmoid(b4_logits).reshape(-1)

        dino_logits = models["dinov2"](x518)
        dino_fake = torch.sigmoid(dino_logits).reshape(-1)

    return {
        "resnet18": float(resnet_fake.item()),
        "efficientnet_b4": float(b4_fake.item()),
        "dinov2": float(dino_fake.item()),
    }
