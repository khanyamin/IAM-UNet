## Quick Test

A quick test is provided to verify checkpoint loading and inference using one sandstone micro-CT image. The test uses the files included in the `quick_test` folder:

```text
quick_test/
├── run_quick_test.py
├── sample_input.png
└── sample_mask.png
```

The script generates a predicted segmentation mask and reports MSE loss, Dice, IoU, Precision, and Recall.

### 1. Install the dependencies

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The IAM-UNet implementation uses the Mamba selective-scan operation. Therefore, a CUDA-enabled Linux environment is recommended.

### 2. Download the trained checkpoint

The trained IAM-UNet checkpoint is publicly available on Kaggle:

https://www.kaggle.com/datasets/mdyaminkhan/iam-unet-checkpoints-and-datasets

Download `net.pth` and place it in the following location:

```text
checkpoints/net.pth
```

The repository structure should be:

```text
IAM-UNet/
├── checkpoints/
│   └── savemodel_seed_82.pth
├── quick_test/
│   ├── run_quick_test.py
│   ├── sample_input.png
│   └── sample_mask.png
├── matrix.py
├── net.py
└── requirements.txt
```

### 3. Run the quick test

From the root directory of the repository, execute:

```bash
python quick_test/run_quick_test.py
```


The predicted binary mask will be saved as:

```text
quick_test/sample_prediction.png
```

The sample input and ground-truth mask are included only for demonstrating the quick-test procedure. The complete dataset is available through the Kaggle link provided above.
