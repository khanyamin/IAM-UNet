from pathlib import Path
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn as nn



ROOT_DIR = Path(__file__).resolve().parents[1]

# Allow imports from the repository root
sys.path.insert(0, str(ROOT_DIR))

from net import InMambaAttentionUNet
from matrix import (
    prepare_binary_mask,
    calculate_dice_coefficient,
    calculate_iou,
    calculate_precision,
    calculate_recall,
)


CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "savemodel_seed_82.pth"
INPUT_PATH = ROOT_DIR / "quick_test" / "sample_input.png"
MASK_PATH = ROOT_DIR / "quick_test" / "sample_mask.png"
OUTPUT_PATH = ROOT_DIR / "quick_test" / "sample_prediction.png"

IMAGE_SIZE = 256
THRESHOLD = 0.5


def load_grayscale_image(
    image_path: Path,
    image_size: int,
    is_mask: bool = False,
) -> torch.Tensor:
    """
    Load a grayscale image and convert it to a tensor with shape:
    [1, 1, height, width].
    """

    if not image_path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("L")

        try:
            if is_mask:
                interpolation = Image.Resampling.NEAREST
            else:
                interpolation = Image.Resampling.BILINEAR
        except AttributeError:
            if is_mask:
                interpolation = Image.NEAREST
            else:
                interpolation = Image.BILINEAR

        image = image.resize(
            (image_size, image_size),
            interpolation,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

    image_tensor = torch.from_numpy(image_array)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    return image_tensor.float()


def extract_state_dict(checkpoint):
    """
    Support checkpoints saved directly with model.state_dict()
    and checkpoints containing a nested state dictionary.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint does not contain a valid dictionary."
        )

    possible_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ]

    for key in possible_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

    cleaned_state_dict = {}

    for key, value in checkpoint.items():
        # Remove prefix created by torch.nn.DataParallel
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Place the trained checkpoint at savemodel/net.pth."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # For older PyTorch versions
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    return extract_state_dict(checkpoint)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("IAM-UNet Quick Test")
    print("=" * 60)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "Warning: GPU was not detected. The Mamba selective-scan "
            "operation may require CUDA."
        )

    # -----------------------------------------------------
    # Initialize model using the same settings as train.py
    # -----------------------------------------------------

    model = InMambaAttentionUNet(
        in_channels=1,
        num_classes=1,
        base_ch=64,
        drop_path=0.1,
        d_state=16,
    ).to(device)

    # -----------------------------------------------------
    # Load trained checkpoint
    # -----------------------------------------------------

    state_dict = load_checkpoint(
        CHECKPOINT_PATH,
        device,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("Checkpoint loaded successfully.")

    # -----------------------------------------------------
    # Load one input image and mask
    # -----------------------------------------------------

    input_tensor = load_grayscale_image(
        INPUT_PATH,
        IMAGE_SIZE,
        is_mask=False,
    ).to(device)

    mask_tensor = load_grayscale_image(
        MASK_PATH,
        IMAGE_SIZE,
        is_mask=True,
    ).to(device)

    # -----------------------------------------------------
    # Run inference
    # -----------------------------------------------------

    model.eval()

    with torch.inference_mode():
        prediction = model(input_tensor)

    expected_shape = (
        1,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected model output shape: {tuple(prediction.shape)}. "
            f"Expected: {expected_shape}"
        )

    if not torch.isfinite(prediction).all():
        raise RuntimeError(
            "The prediction contains NaN or infinite values."
        )

    # The network already applies sigmoid in net.py
    prediction_np = prediction.detach().cpu().numpy()
    mask_np = mask_tensor.detach().cpu().numpy()

    # Use exactly the same mask preparation as test.py
    prediction_binary = prepare_binary_mask(
        prediction_np,
        threshold=THRESHOLD,
        invert=True,
    )

    ground_truth_binary = prepare_binary_mask(
        mask_np,
        threshold=THRESHOLD,
        invert=True,
    )

    predicted_mask = prediction_binary[0, 0]
    true_mask = ground_truth_binary[0, 0]

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    criterion = nn.MSELoss()
    mse_loss = criterion(
        prediction,
        mask_tensor,
    ).item()

    dice = calculate_dice_coefficient(
        predicted_mask,
        true_mask,
    )

    iou = calculate_iou(
        predicted_mask,
        true_mask,
    )

    precision = calculate_precision(
        predicted_mask,
        true_mask,
    )

    recall = calculate_recall(
        predicted_mask,
        true_mask,
    )

    # -----------------------------------------------------
    # Save predicted mask
    # Pore = black, matrix = white
    # -----------------------------------------------------

    output_image = np.where(
        predicted_mask,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(output_image).save(OUTPUT_PATH)

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("-" * 60)
    print(f"Input image   : {INPUT_PATH}")
    print(f"Ground truth : {MASK_PATH}")
    print(f"Input shape   : {tuple(input_tensor.shape)}")
    print(f"Output shape  : {tuple(prediction.shape)}")
    print(
        f"Output range  : "
        f"{prediction.min().item():.6f} to "
        f"{prediction.max().item():.6f}"
    )
    print("-" * 60)
    print(f"MSE Loss      : {mse_loss:.6f}")
    print(f"Dice          : {dice:.6f}")
    print(f"IoU           : {iou:.6f}")
    print(f"Precision     : {precision:.6f}")
    print(f"Recall        : {recall:.6f}")
    print("-" * 60)
    print(f"Prediction saved to: {OUTPUT_PATH}")
    print("IAM-UNet quick test completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("=" * 60)
        print("IAM-UNet quick test failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")
        print("=" * 60)
        raise
