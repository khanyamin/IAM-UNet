from pathlib import Path
import sys

import torch


# Allow Python to import net.py from the repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# EDIT THESE LINES ACCORDING TO YOUR MODEL
# ============================================================

from net import InMambaAttentionUNet

CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "model.pth"

MODEL_ARGUMENTS = {
    # Example:
    # "in_channels": 1,
    # "num_classes": 1,
}

INPUT_CHANNELS = 1
IMAGE_SIZE = 256

# ============================================================


def extract_state_dict(checkpoint):
    """
    Supports common PyTorch checkpoint formats.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary.")

    for key in [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # Checkpoint was saved directly using model.state_dict()
    return checkpoint


def remove_module_prefix(state_dict):
    """
    Removes the 'module.' prefix from checkpoints trained
    using DataParallel.
    """

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("IAM-UNet Quick Test")
    print("=" * 55)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Place the checkpoint at checkpoints/model.pth"
        )

    # Initialize the model
    model = InMambaAttentionUNet(
        **MODEL_ARGUMENTS
    ).to(device)

    # Load the checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_module_prefix(state_dict)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print("Checkpoint loaded successfully.")

    model.eval()

    # Artificial grayscale image:
    # [batch, channels, height, width]
    sample_input = torch.randn(
        1,
        INPUT_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    with torch.no_grad():
        output = model(sample_input)

    # Some models return a tuple or list
    if isinstance(output, (tuple, list)):
        output = output[0]

    # Some models return a dictionary
    if isinstance(output, dict):
        if "out" in output:
            output = output["out"]
        elif "logits" in output:
            output = output["logits"]
        else:
            output = next(iter(output.values()))

    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "The model output is not a PyTorch tensor."
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "The output contains NaN or infinite values."
        )

    print("Input shape:", tuple(sample_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Output minimum:", output.min().item())
    print("Output maximum:", output.max().item())
    print("=" * 55)
    print("Quick test completed successfully.")
    print("=" * 55)


if __name__ == "__main__":
    main()
