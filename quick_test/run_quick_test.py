from pathlib import Path
import sys

import torch

# Add the repository root directory to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from net import InMambaAttentionUNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("IAM-UNet Quick Test")
    print("=" * 50)
    print(f"Device: {device}")

    # Initialize the IAM-UNet model
    model = InMambaAttentionUNet(
        in_channels=1,
        num_classes=1,
        base_ch=64,
        drop_path=0.1,
        d_state=16
    ).to(device)

    model.eval()

    # Create one synthetic grayscale micro-CT image
    sample_input = torch.randn(1, 1, 256, 256, device=device)

    # Run one forward pass
    with torch.no_grad():
        output = model(sample_input)

    if isinstance(output, (tuple, list)):
        output = output[0]

    expected_shape = (1, 1, 256, 256)

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(output.shape)}. "
            f"Expected: {expected_shape}"
        )

    print(f"Input shape:  {tuple(sample_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print("IAM-UNet quick test completed successfully.")


if __name__ == "__main__":
    main()
