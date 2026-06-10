from pathlib import Path
import sys

import torch


# Add the repository root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Import the IAM-UNet model from net.py
from net import InMambaAttentionUNet


def main():
    # Use GPU when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("IAM-UNet Quick Test")
    print("=" * 50)
    print(f"Device: {device}")

    # Initialize IAM-UNet
    model = InMambaAttentionUNet(
        in_channels=1,
        num_classes=1,
        base_ch=64,
        drop_path=0.1,
        d_state=16
    ).to(device)

    model.eval()

    # Create one synthetic grayscale micro-CT image
    # Shape: batch, channel, height, width
    sample_input = torch.randn(
        1, 1, 256, 256,
        device=device
    )

    # Perform a forward pass
    with torch.no_grad():
        output = model(sample_input)

    # Verify the output
    expected_shape = (1, 1, 256, 256)

    assert tuple(output.shape) == expected_shape, (
        f"Unexpected output shape: {tuple(output.shape)}. "
        f"Expected: {expected_shape}"
    )

    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0

    print(f"Input shape:  {tuple(sample_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output minimum: {output.min().item():.6f}")
    print(f"Output maximum: {output.max().item():.6f}")
    print("IAM-UNet quick test completed successfully.")


if __name__ == "__main__":
    main()
