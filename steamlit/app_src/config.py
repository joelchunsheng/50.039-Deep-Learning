from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"


PROJECT_INFO = {
    "title": "Anomaly Detection of Melanoma in Dermoscopic Images",
    "subtitle": "By Gay Jun Han Dylan (1007831), Lydia Rachel Robert (1008462), Yeo Chun Sheng Joel (1008112)",
    "sidebar_note": (
        "This starter app includes placeholder dataset statistics, evaluation metrics, and model loading logic. "
        "Replace the sample values below with your real experiment results and saved model files."
    ),
}


DATASET_STATS = {
    "total_images": 11527,
    "melanoma_count": 1284,
    "nevus_count": 10243,
    "melanoma_pct": 11.13,
    "nevus_pct": 88.87,
    "class_ratio": "1 : 2.33",
    "imbalance_note": "The dataset is moderately imbalanced toward nevus, so recall-focused metrics such as F2-score are especially important.",
    "additional_statistics": {
        "Train Samples": 7991,
        "Validation Samples": 2024,
        "Test Samples": 1512,
        "Image Input Size": "224 x 224",
        "Normalization": "ImageNet mean/std",
        "Augmentation Used": "Resize, random horizontal and vertical flip, random rotation, color jitter, random affine transformation, normalization",
    },
}


CNN_TABLE = [
    {
        "Notebook": "Model 1 - Residual CNN",
        "Key Changes": "BatchNorm + Residual + Weighted",
        "Val F2": 0.7676,
        "Test F2": 0.5814,
        "Recall": 0.8187,
        "Prec": 0.2692,
        "AUC": 0.8672,
    },
]

RESNET_TABLE = [
    {
        "Notebook": "Model 2 - ResNet18",
        "Key Changes": "A transfer learning model fine-tuned from ResNet18.",
        "Val F2": 0.6835,
        "Test F2": 0.6704,
        "Recall": 0.7778,
        "Prec": 0.4318,
        "AUC": 0.9133,
    },
    {
        "Notebook": "Model 3 - ResNet50",
        "Key Changes": "A transfer learning model fine-tuned from ResNet18.",
        "Val F2": 0.6459,
        "Test F2": 0.6184,
        "Recall": 0.7485,
        "Prec": 0.3647,
        "AUC": 0.8841,
    },
]

EFFICIENTNET_TABLE = [
    {
        "Notebook": "Model 4 - EfficientNet",
        "Key Changes": "A compact high-performing architecture with strong feature extraction.",
        "Val F2": 0.6873,
        "Test F2": 0.6952,
        "Recall": 0.8830,
        "Prec": 0.3756,
        "AUC": 0.9182,
    },
]

MOBILENET_TABLE = []

DENSENET_TABLE = [
    {
        "Notebook": "Model 5 - DenseNet",
        "Key Changes": "A compact high-performing architecture with strong feature extraction.",
        "Val F2": 0.6873,
        "Test F2": 0.6952,
        "Recall": 0.8830,
        "Prec": 0.3756,
        "AUC": 0.9182,
    },
]

VIT_TABLE = []


MODEL_RESULTS = [
    {
        "key": "Residual_cnn",
        "display_name": "Model 1 - CNN",
        "file_path": str(MODELS_DIR / "baseline_cnn" / "model.pth"),
        "checkpoint_type": "full_model",
        "summary": "BatchNorm + Residual + Weighted",
        "is_best": False,
        "metrics": {
            "val_f2": 0.7676,
            "precision": 0.2692,
            "recall": 0.8187,
            "f2_score": 0.5814,
            "roc_auc": 0.8672,
        },
        "confusion_matrix": [[920, 380], [31, 140]],
        "training_history": {
            "epoch": [1, 2, 3, 4, 5],
            "train_loss": [0.72, 0.53, 0.39, 0.29, 0.22],
            "val_loss": [0.77, 0.58, 0.48, 0.40, 0.34],
            "train_accuracy": [0.61, 0.72, 0.81, 0.86, 0.89],
            "val_accuracy": [0.58, 0.68, 0.74, 0.81, 0.86],
        },
    },
    {
        "key": "resnet18",
        "display_name": "Model 2 - ResNet18",
        "file_path": str(MODELS_DIR / "resnet18_finetuned" / "model.pth"),
        "checkpoint_type": "full_model",
        "summary": "A transfer learning model fine-tuned from ResNet18.",
        "is_best": False,
        "metrics": {
            "val_f2": 0.6835,
            "precision": 0.4318,
            "recall": 0.7778,
            "f2_score": 0.6704,
            "roc_auc": 0.9133,
        },
        "confusion_matrix": [[95, 5], [9, 71]],
        "training_history": {
            "epoch": [1, 2, 3, 4, 5],
            "train_loss": [0.58, 0.41, 0.29, 0.21, 0.16],
            "val_loss": [0.63, 0.46, 0.35, 0.24, 0.19],
            "train_accuracy": [0.69, 0.80, 0.87, 0.91, 0.94],
            "val_accuracy": [0.66, 0.79, 0.85, 0.90, 0.92],
        },
    },
    {
        "key": "resnet50",
        "display_name": "Model 3 - ResNet50",
        "file_path": str(MODELS_DIR / "resnet18_finetuned" / "model.pth"),
        "checkpoint_type": "full_model",
        "summary": "A transfer learning model fine-tuned from ResNet18.",
        "is_best": False,
        "metrics": {
            "val_f2": 0.6459,
            "precision": 0.3647,
            "recall": 0.7485,
            "f2_score": 0.6184,
            "roc_auc": 0.8841,
        },
        "confusion_matrix": [[95, 5], [9, 71]],
        "training_history": {
            "epoch": [1, 2, 3, 4, 5],
            "train_loss": [0.58, 0.41, 0.29, 0.21, 0.16],
            "val_loss": [0.63, 0.46, 0.35, 0.24, 0.19],
            "train_accuracy": [0.69, 0.80, 0.87, 0.91, 0.94],
            "val_accuracy": [0.66, 0.79, 0.85, 0.90, 0.92],
        },
    },
    {
        "key": "efficientnet",
        "display_name": "Model 4 - EfficientNet",
        "file_path": str(MODELS_DIR / "efficientnet_b0" / "model.pth"),
        "checkpoint_type": "full_model",
        "summary": "A compact high-performing architecture with strong feature extraction.",
        "is_best": False,
        "metrics": {
            "val_f2": 0.6873,
            "precision": 0.3756,
            "recall": 0.8830,
            "f2_score": 0.6952,
            "roc_auc": 0.9182,
        },
        "confusion_matrix": [[94, 6], [10, 70]],
        "training_history": {
            "epoch": [1, 2, 3, 4, 5],
            "train_loss": [0.61, 0.44, 0.32, 0.23, 0.18],
            "val_loss": [0.65, 0.49, 0.37, 0.27, 0.22],
            "train_accuracy": [0.67, 0.77, 0.85, 0.90, 0.93],
            "val_accuracy": [0.64, 0.76, 0.84, 0.88, 0.91],
        },
    },
    {
        "key": "densenet",
        "display_name": "Model 5 - DenseNet",
        "file_path": str(MODELS_DIR / "efficientnet_b0" / "model.pth"),
        "checkpoint_type": "full_model",
        "summary": "A compact high-performing architecture with strong feature extraction.",
        "is_best": False,
        "metrics": {
            "val_f2": 0.6873,
            "precision": 0.3756,
            "recall": 0.8830,
            "f2_score": 0.6952,
            "roc_auc": 0.9182,
        },
        "confusion_matrix": [[94, 6], [10, 70]],
        "training_history": {
            "epoch": [1, 2, 3, 4, 5],
            "train_loss": [0.61, 0.44, 0.32, 0.23, 0.18],
            "val_loss": [0.65, 0.49, 0.37, 0.27, 0.22],
            "train_accuracy": [0.67, 0.77, 0.85, 0.90, 0.93],
            "val_accuracy": [0.64, 0.76, 0.84, 0.88, 0.91],
        },
    },
    {
        "key": "mobileNet",
        "display_name": "Model 6 - MobileNet",
        "file_path": str(MODELS_DIR / "efficientnet_b0" / "model.pth"),
        "checkpoint_type": "full_model",
        "summary": "Ensemble (MBV3 + EffB0)",
        "is_best": True,
        "metrics": {
            "val_f2": 0.7023,
            "precision": 0.3555,
            "recall": 0.8772,
            "f2_score": 0.6781,
            "roc_auc": 0.9235,
        },
        "confusion_matrix": [[94, 6], [10, 70]],
        "training_history": {
            "epoch": [1, 2, 3, 4, 5],
            "train_loss": [0.61, 0.44, 0.32, 0.23, 0.18],
            "val_loss": [0.65, 0.49, 0.37, 0.27, 0.22],
            "train_accuracy": [0.67, 0.77, 0.85, 0.90, 0.93],
            "val_accuracy": [0.64, 0.76, 0.84, 0.88, 0.91],
        },
    },
]


CLASS_LABELS = {0: "Nevus", 1: "Melanoma"}
INPUT_SIZE = (224, 224)
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
