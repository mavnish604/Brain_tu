from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_ROOT = PROJECT_ROOT / "Dataset"
CLASS_DIR = DATASET_ROOT / "BT-MRI Dataset" / "BT-MRI Dataset" / "Testing"
DEFAULT_IMAGE_PATH = (
    DATASET_ROOT
    / "Challenging Datasets"
    / "Challenging Datasets"
    / "Blurred Dataset"
    / "Glioma"
    / "bilateral_glioma (1).jpg"
)
ORIGINAL_MODEL_PATH = MODELS_DIR / "densenet121_brain_tumor.pth"
MOBILE_MODEL_PATH = MODELS_DIR / "brain_tumor_final_universal.ptl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original DenseNet121 model and the Lite mobile model on one image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the image to evaluate.",
    )
    return parser.parse_args()


def get_class_names() -> list[str]:
    if not CLASS_DIR.exists():
        raise FileNotFoundError(f"Class directory not found: {CLASS_DIR}")
    return sorted(path.name for path in CLASS_DIR.iterdir() if path.is_dir())


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_original_model(model_path: Path) -> torch.nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 4)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_mobile_model(model_path: Path) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    return model


def prepare_image(image_path: Path) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    return prepare_pil_image(image)


def prepare_pil_image(image: Image.Image) -> torch.Tensor:
    return build_transform()(image.convert("RGB")).unsqueeze(0).float()


def prepare_image_bytes(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return prepare_pil_image(image)


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, class_names: list[str]) -> dict[str, object]:
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_index = int(torch.argmax(probabilities).item())
    top_probability = float(probabilities[top_index].item())

    return {
        "label": class_names[top_index],
        "confidence": top_probability,
        "probabilities": {
            class_name: float(probabilities[idx].item())
            for idx, class_name in enumerate(class_names)
        },
    }


def print_result(model_name: str, result: dict[str, object]) -> None:
    print(f"\n{model_name}")
    print(f"Predicted class: {result['label']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("Class probabilities:")
    for class_name, score in result["probabilities"].items():
        print(f"  {class_name:<12} {score * 100:>6.2f}%")


class BrainTumorInferenceService:
    def __init__(self) -> None:
        self.class_names = get_class_names()
        self.original_model = load_original_model(ORIGINAL_MODEL_PATH)
        self.mobile_model = load_mobile_model(MOBILE_MODEL_PATH)

    def predict_from_tensor(self, image_tensor: torch.Tensor) -> dict[str, dict[str, object]]:
        original_result = predict(self.original_model, image_tensor, self.class_names)
        mobile_result = predict(self.mobile_model, image_tensor, self.class_names)
        return {
            "original_model": original_result,
            "mobile_model": mobile_result,
            "models_agree": original_result["label"] == mobile_result["label"],
        }

    def predict_from_path(self, image_path: Path) -> dict[str, dict[str, object]]:
        image_tensor = prepare_image(image_path)
        return self.predict_from_tensor(image_tensor)

    def predict_from_bytes(self, image_bytes: bytes) -> dict[str, dict[str, object]]:
        image_tensor = prepare_image_bytes(image_bytes)
        return self.predict_from_tensor(image_tensor)


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()

    expected_label = image_path.parent.name
    service = BrainTumorInferenceService()
    results = service.predict_from_path(image_path)
    original_result = results["original_model"]
    mobile_result = results["mobile_model"]

    print(f"Image: {image_path}")
    print(f"Expected label from folder: {expected_label}")
    print_result("Original model (.pth)", original_result)
    print_result("Mobile model (.ptl)", mobile_result)

    print(f"\nDo both models agree? {'Yes' if results['models_agree'] else 'No'}")


if __name__ == "__main__":
    main()
