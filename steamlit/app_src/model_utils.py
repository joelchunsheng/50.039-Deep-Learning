from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app_src.config import ALLOWED_IMAGE_SUFFIXES, CLASS_LABELS, INPUT_SIZE


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def discover_sample_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [path for path in sorted(folder.iterdir()) if path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES][:6]


def read_uploaded_image(uploaded_file: Any) -> Image.Image | None:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def create_model_architecture(model_config: dict) -> torch.nn.Module:
    raise NotImplementedError(
        "State-dict loading is not configured yet. If your saved file is a state_dict checkpoint, "
        "define the model architecture inside `create_model_architecture()` and set "
        "`checkpoint_type` to `state_dict` in `app_src/config.py`."
    )


@st.cache_resource(show_spinner=False)
def load_model(model_config: dict) -> tuple[torch.nn.Module | None, torch.device, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_config["file_path"])

    if not model_path.exists():
        return None, device, f"Model file not found at `{model_path}`. Add your trained model to this path first."

    try:
        checkpoint_type = model_config.get("checkpoint_type", "full_model")
        if checkpoint_type == "full_model":
            model = torch.load(model_path, map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = create_model_architecture(model_config)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

        model.eval()
        model.to(device)
        return model, device, f"Loaded model from `{model_path}`."
    except NotImplementedError as exc:
        return None, device, str(exc)
    except Exception as exc:
        return None, device, f"Failed to load model from `{model_path}`. Details: {exc}"


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    model_config: dict,
) -> dict:
    transform = build_transform()

    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(image_tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)

        predicted_label = CLASS_LABELS[int(predicted_index.item())]
        melanoma_probability = float(probabilities[0][1].item())
        nevus_probability = float(probabilities[0][0].item())

        return {
            "status": "success",
            "model_name": model_config["display_name"],
            "predicted_index": int(predicted_index.item()),
            "predicted_label": predicted_label,
            "confidence": float(confidence.item()),
            "melanoma_probability": melanoma_probability,
            "nevus_probability": nevus_probability,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Inference failed. Please verify your model output shape and preprocessing logic. Details: {exc}",
        }
