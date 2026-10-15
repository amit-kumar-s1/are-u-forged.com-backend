
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


TRANSFORMS = {
    "224": build_transform(224),
    "380": build_transform(380),
    "518": build_transform(518),
}


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    return {
        size: transform(image)
        for size, transform in TRANSFORMS.items()
    }
