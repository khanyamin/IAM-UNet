import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from net import InMambaAttentionUNet
from matrix import (
    prepare_binary_mask,
    calculate_dice_coefficient,
    calculate_iou,
    calculate_precision,
    calculate_recall
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_input_dir = "datasets/test/input"
test_mask_dir = "datasets/test/mask"
save_dir = "savemodel"
pred_dir = os.path.join(save_dir, "pred")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

data_path = sorted(os.listdir(test_input_dir))
mask_path = sorted(os.listdir(test_mask_dir))

if len(data_path) != len(mask_path):
    raise ValueError("Number of test images and test masks does not match.")


def default_loader(path):
    img = Image.open(os.path.join(test_input_dir, path)).convert("L")
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape((1, img.shape[0], img.shape[1]))
    img_tensor = torch.tensor(img, dtype=torch.float32)
    return img_tensor


def mask_loader(path):
    mask = Image.open(os.path.join(test_mask_dir, path)).convert("L")
    mask = np.array(mask, dtype=np.float32) / 255.0
    mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    return mask_tensor


class TestSet(Dataset):
    def __init__(self, img_list, loader=default_loader, mask_loader=mask_loader):
        self.images = img_list
        self.loader = loader
        self.mask_loader = mask_loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        mask = self.mask_loader(fn)
        return img, mask, fn

    def __len__(self):
        return len(self.images)


G = InMambaAttentionUNet().to(device)
mod = torch.load(os.path.join(save_dir, "net.pth"), map_location=device)
G.load_state_dict(mod)
G.eval()

test_data = TestSet(data_path)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

criterion = nn.MSELoss()

test_loss = 0.0
dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []

all_preds_raw = []

for i, (data, mask, filename) in enumerate(testloader):
    with torch.no_grad():
        data = data.to(device)
        mask = mask.to(device)

        pred = G(data)
        loss = criterion(pred, mask)
        test_loss += loss.item()

        pred_np = pred.cpu().numpy()
        mask_np = mask.cpu().numpy()

        pred_bin = prepare_binary_mask(pred_np, threshold=0.5, invert=True)
        true_bin = prepare_binary_mask(mask_np, threshold=0.5, invert=True)

        p = pred_bin[0, 0]
        t = true_bin[0, 0]

        dice = calculate_dice_coefficient(p, t)
        iou = calculate_iou(p, t)
        precision = calculate_precision(p, t)
        recall = calculate_recall(p, t)

        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)

        save_img = np.where(p, 0, 255).astype(np.uint8)

        out_name = os.path.splitext(filename[0])[0] + ".png"
        out_path = os.path.join(pred_dir, out_name)
        Image.fromarray(save_img).save(out_path)

        all_preds_raw.append(save_img)

        print(f"[{i + 1}/{len(testloader)}] {filename[0]}")
        print(
            f"Loss: {loss.item():.6f}, Dice: {dice:.6f}, IoU: {iou:.6f}, "
            f"Precision: {precision:.6f}, Recall: {recall:.6f}"
        )

average_test_loss = test_loss / len(testloader)
average_dice = np.mean(dice_scores)
average_iou = np.mean(iou_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)

print("=" * 50)
print(f"Test Loss      : {average_test_loss:.6f}")
print(f"Test Dice      : {average_dice:.6f}")
print(f"Test IoU       : {average_iou:.6f}")
print(f"Test Precision : {average_precision:.6f}")
print(f"Test Recall    : {average_recall:.6f}")
print("=" * 50)

with open(os.path.join(save_dir, "test_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test Loss: {average_test_loss:.6f}\n")
    f.write(f"Test Dice: {average_dice:.6f}\n")
    f.write(f"Test IoU: {average_iou:.6f}\n")
    f.write(f"Test Precision: {average_precision:.6f}\n")
    f.write(f"Test Recall: {average_recall:.6f}\n")

with open(os.path.join(save_dir, "test_metrics_per_image.csv"), "w", encoding="utf-8") as f:
    f.write("filename,dice,iou,precision,recall\n")
    for idx, name in enumerate(data_path):
        f.write(
            f"{name},{dice_scores[idx]:.6f},{iou_scores[idx]:.6f},"
            f"{precision_scores[idx]:.6f},{recall_scores[idx]:.6f}\n"
        )

all_preds_raw = np.array(all_preds_raw, dtype=np.uint8)
all_preds_raw.tofile(os.path.join(save_dir, "pred.raw"))

print(f"All predicted images saved in: {pred_dir}")
print(f"All test results saved in: {save_dir}")
