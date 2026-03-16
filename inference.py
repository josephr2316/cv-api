from functools import lru_cache
from io import BytesIO
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from settings import settings


# CIFAR-10 class names in canonical order
CIFAR10_CLASSES: list[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Apply the same kind of preprocessing used during evaluation:
    resize -> center crop -> to tensor -> normalize.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform(image).unsqueeze(0)  # add batch dimension


def _build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@lru_cache(maxsize=1)
def load_model() -> tuple[nn.Module, list[str], torch.device]:
    """
    Load the trained model and class names from disk, move to device, and cache it.
    """
    device = get_device()

    checkpoint: dict[str, Any] = torch.load(settings.model_path, map_location=device)
    classes: list[str] = checkpoint.get("classes", CIFAR10_CLASSES)

    model = _build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, device


def predict_image_bytes(image_bytes: bytes) -> dict[str, Any]:
    """
    Take raw image bytes (as received from an UploadFile), run inference,
    and return prediction details.
    """
    with Image.open(BytesIO(image_bytes)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        input_tensor = _preprocess_image(img)

    model, classes, device = load_model()
    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    prob_values = probs.cpu().tolist()
    top_index = int(torch.argmax(probs).item())
    top_class = classes[top_index]
    top_prob = float(prob_values[top_index])

    return {
        "class_index": top_index,
        "class_name": top_class,
        "probability": top_prob,
        "all_probabilities": prob_values,
        "classes": classes,
    }

