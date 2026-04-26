"""
vision_target.py

Loads a vision model wrapped in ART's PyTorchClassifier.
Supports Metal GPU (Apple Silicon), CUDA, and CPU automatically.

Supported model targets:
  - resnet50        : baseline
  - wide_resnet     : WideResNet-50-2
  - densenet        : DenseNet-161
  - efficientnet    : EfficientNet-B4
  - convnext        : ConvNeXt-Base
  - vit             : ViT-B/16
  - resnet50_at     : ResNet50 adversarially trained (RobustBench)
  - vit_at          : ViT adversarially trained (RobustBench)

Usage:
    from vision_target import load_vision_target
    art_classifier, x_test, y_test = load_vision_target('wide_resnet')
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

from art.estimators.classification import PyTorchClassifier


# ---------------------------------------------------------
# DEVICE — Metal (Apple Silicon) > CUDA > CPU
# ---------------------------------------------------------
def _get_device() -> tuple[torch.device, str]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), "cpu"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    else:
        return torch.device("cpu"), "cpu"

TORCH_DEVICE, ART_DEVICE = _get_device()
print(f"[VisionTarget] Device: {TORCH_DEVICE}  (ART sees: '{ART_DEVICE}')")


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
NUM_CLASSES   = 1000
INPUT_SHAPE   = (3, 224, 224)
BATCH_SIZE    = 32
IMAGENET_PATH = "../data"       

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BASELINE_MODELS = {'resnet50', 'vit'}


# ---------------------------------------------------------
# STEP 1: Model loader
# ---------------------------------------------------------
def load_pytorch_model(model_name: str) -> nn.Module:
    name = model_name.lower().strip()

    if name in ('resnet50_at', 'vit_at'):
        try:
            from robustbench.utils import load_model as rb_load
        except ImportError:
            raise ImportError(
                "RobustBench not installed. "
                "Run: pip install git+https://github.com/RobustBench/robustbench.git"
            )
        if name == 'resnet50_at':
            model = rb_load(
                model_name   = 'Engstrom2019Robustness',
                dataset      = 'imagenet',
                threat_model = 'Linf'
            )
        else:
            model = rb_load(
                model_name   = 'Singh2023Revisiting_ViT-B-ConvStem',
                dataset      = 'imagenet',
                threat_model = 'Linf'
            )

    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    elif name == 'wide_resnet':
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)

    elif name == 'densenet':
        model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)

    elif name == 'efficientnet':
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

    elif name == 'convnext':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

    elif name == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: resnet50, wide_resnet, densenet, efficientnet, "
            f"convnext, vit, resnet50_at, vit_at"
        )

    if name in BASELINE_MODELS:
        print(f"[VisionTarget] ⚠️  '{name}' is a baseline — consider wide_resnet, "
              f"densenet, efficientnet, or convnext for harder targets.")

    model.eval()
    model.to(TORCH_DEVICE)
    return model


# ---------------------------------------------------------
# STEP 2: Wrap in ART PyTorchClassifier
# ---------------------------------------------------------
def wrap_with_art(model: nn.Module) -> PyTorchClassifier:
    return PyTorchClassifier(
        model         = model,
        loss          = nn.CrossEntropyLoss(),
        input_shape   = INPUT_SHAPE,
        nb_classes    = NUM_CLASSES,
        clip_values   = (0.0, 1.0),
        device_type   = ART_DEVICE,
        preprocessing = (
            np.array(IMAGENET_MEAN),
            np.array(IMAGENET_STD)
        )
    )


# ---------------------------------------------------------
# STEP 3: Load ImageNet validation batch
# ---------------------------------------------------------
def load_imagenet_batch(batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a batch from the full ImageNet validation set.
    Returns:
        x_test : (N, 3, 224, 224) float32 in [0, 1]
        y_test : (N,) int64 in [0, 999]
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),   # [0, 1] — ART handles normalization internally
    ])

    dataset = ImageNet(root=IMAGENET_PATH, split='val', transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    images, labels = next(iter(loader))
    return images.numpy().astype(np.float32), labels.numpy().astype(np.int64)


# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------
def load_vision_target(
    model_name: str,
    batch_size: int = BATCH_SIZE
) -> tuple[PyTorchClassifier, np.ndarray, np.ndarray]:
    """
    Args:
        model_name : resnet50 | wide_resnet | densenet | efficientnet |
                     convnext | vit | resnet50_at | vit_at
        batch_size : number of ImageNet val images to load
    Returns:
        art_classifier, x_test, y_test
    """
    print(f"[VisionTarget] Loading model      : {model_name}")
    model = load_pytorch_model(model_name)

    print(f"[VisionTarget] Wrapping with ART  : PyTorchClassifier")
    art_classifier = wrap_with_art(model)

    print(f"[VisionTarget] Loading ImageNet   : {batch_size} images from {IMAGENET_PATH}")
    x_test, y_test = load_imagenet_batch(batch_size)

    print(f"[VisionTarget] Ready. x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    return art_classifier, x_test, y_test


# ---------------------------------------------------------
# SANITY CHECK — python vision_target.py
# ---------------------------------------------------------
if __name__ == "__main__":
    for target in ['resnet50', 'wide_resnet', 'densenet', 'efficientnet', 'convnext']:
        print(f"\n{'='*50}")
        classifier, x, y = load_vision_target(target, batch_size=32)
        preds     = np.argmax(classifier.predict(x), axis=1)
        clean_acc = np.mean(preds == y)
        print(f"[{target}] Clean accuracy: {clean_acc*100:.1f}%")