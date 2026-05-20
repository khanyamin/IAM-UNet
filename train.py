import time
import warnings
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from net import *
from matrix import (
    prepare_binary_mask,
    calculate_dice_coefficient,
    calculate_iou,
    calculate_precision,
    calculate_recall
)

warnings.filterwarnings("ignore")

seed = 83
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "datasets/train/input"
label_dir = "datasets/train/mask"
save_dir = "savemodel"
os.makedirs(save_dir, exist_ok=True)

data_path = sorted(os.listdir(train_dir))
data_path1 = sorted(os.listdir(label_dir))

if len(data_path) != len(data_path1):
    raise ValueError("Number of train images and label images does not match.")

train_val_ratio = 0.8
train_size = int(train_val_ratio * len(data_path))

train_paths = data_path[:train_size]
val_paths = data_path[train_size:]


def default_loader(path):
    img = Image.open(os.path.join(train_dir, path)).convert("L")
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape((1, img.shape[0], img.shape[1]))
    img_tensor = torch.tensor(img, dtype=torch.float32)
    return img_tensor


def default_loader1(path):
    mask = Image.open(os.path.join(label_dir, path)).convert("L")
    mask = np.array(mask, dtype=np.float32) / 255.0
    mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    return mask_tensor


class TrainSet(Dataset):
    def __init__(self, paths, loader=default_loader, loader1=default_loader1):
        self.images = paths
        self.loader = loader
        self.loader1 = loader1

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.loader1(fn)
        return img, target

    def __len__(self):
        return len(self.images)


def compute_batch_metrics(pred, label, threshold=0.5):
    pred_np = pred.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()

    pred_bin = prepare_binary_mask(pred_np, threshold=threshold, invert=True)
    label_bin = prepare_binary_mask(label_np, threshold=threshold, invert=True)

    batch_dice = []
    batch_iou = []
    batch_precision = []
    batch_recall = []

    for b in range(pred_bin.shape[0]):
        p = pred_bin[b, 0]
        t = label_bin[b, 0]

        batch_dice.append(calculate_dice_coefficient(p, t))
        batch_iou.append(calculate_iou(p, t))
        batch_precision.append(calculate_precision(p, t))
        batch_recall.append(calculate_recall(p, t))

    return (
        np.mean(batch_dice),
        np.mean(batch_iou),
        np.mean(batch_precision),
        np.mean(batch_recall),
    )


train_data = TrainSet(train_paths)
val_data = TrainSet(val_paths)

batch_size = 8

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

net = InceptionMambaAttentionUNet().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))

# If you want optimizer regularization, comment the line above and remove # below
# optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-5)

mse = nn.MSELoss()
epochs = 100

# If you want manual L2 regularization, remove # below
# lambda_reg = 1e-5

Loss_list = []
Val_loss_list = []

Train_dice_list = []
Train_iou_list = []
Train_precision_list = []
Train_recall_list = []

Val_dice_list = []
Val_iou_list = []
Val_precision_list = []
Val_recall_list = []

for epoch in range(epochs):
    net.train()
    t1 = time.time()

    train_loss = 0.0
    train_dice = 0.0
    train_iou = 0.0
    train_precision = 0.0
    train_recall = 0.0

    for i, (data, label) in enumerate(trainloader, 1):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = net(data)
        loss = mse(pred, label)

        # If you want manual L2 regularization, remove # from this block
        # l2_reg = 0.0
        # for param in net.parameters():
        #     l2_reg += torch.norm(param, p=2)
        # loss = loss + lambda_reg * l2_reg

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        dice, iou, precision, recall = compute_batch_metrics(pred, label, threshold=0.5)
        train_dice += dice
        train_iou += iou
        train_precision += precision
        train_recall += recall

        print(
            f"\rEpoch [{epoch + 1}/{epochs}] Batch [{i}/{len(trainloader)}] Loss: {loss.item():.6f}",
            end=""
        )

    print()

    avg_train_loss = train_loss / len(trainloader)
    avg_train_dice = train_dice / len(trainloader)
    avg_train_iou = train_iou / len(trainloader)
    avg_train_precision = train_precision / len(trainloader)
    avg_train_recall = train_recall / len(trainloader)

    Loss_list.append(avg_train_loss)
    Train_dice_list.append(avg_train_dice)
    Train_iou_list.append(avg_train_iou)
    Train_precision_list.append(avg_train_precision)
    Train_recall_list.append(avg_train_recall)

    net.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    val_precision = 0.0
    val_recall = 0.0

    with torch.no_grad():
        for data_val, label_val in valloader:
            data_val = data_val.to(device)
            label_val = label_val.to(device)

            pred_val = net(data_val)
            loss_val = mse(pred_val, label_val)
            val_loss += loss_val.item()

            dice, iou, precision, recall = compute_batch_metrics(pred_val, label_val, threshold=0.5)
            val_dice += dice
            val_iou += iou
            val_precision += precision
            val_recall += recall

    avg_val_loss = val_loss / len(valloader)
    avg_val_dice = val_dice / len(valloader)
    avg_val_iou = val_iou / len(valloader)
    avg_val_precision = val_precision / len(valloader)
    avg_val_recall = val_recall / len(valloader)

    Val_loss_list.append(avg_val_loss)
    Val_dice_list.append(avg_val_dice)
    Val_iou_list.append(avg_val_iou)
    Val_precision_list.append(avg_val_precision)
    Val_recall_list.append(avg_val_recall)

    t2 = time.time()

    print("=" * 50)
    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"Epoch Time       : {t2 - t1:.4f} sec")

    print(f"Train Loss       : {avg_train_loss:.6f}")
    print(f"Train Dice       : {avg_train_dice:.6f}")
    print(f"Train IoU        : {avg_train_iou:.6f}")
    print(f"Train Precision  : {avg_train_precision:.6f}")
    print(f"Train Recall     : {avg_train_recall:.6f}")

    print(f"Val Loss         : {avg_val_loss:.6f}")
    print(f"Val Dice         : {avg_val_dice:.6f}")
    print(f"Val IoU          : {avg_val_iou:.6f}")
    print(f"Val Precision    : {avg_val_precision:.6f}")
    print(f"Val Recall       : {avg_val_recall:.6f}")
    print("=" * 50)

    if (epoch + 1) % 10 == 0:
        torch.save(net.state_dict(), os.path.join(save_dir, f"net_{epoch + 1}.pth"))

        np.savetxt(os.path.join(save_dir, "Train Loss.csv"), np.array(Loss_list), delimiter=",")
        np.savetxt(os.path.join(save_dir, "Val Loss.csv"), np.array(Val_loss_list), delimiter=",")

        np.savetxt(os.path.join(save_dir, "Train Dice.csv"), np.array(Train_dice_list), delimiter=",")
        np.savetxt(os.path.join(save_dir, "Train IoU.csv"), np.array(Train_iou_list), delimiter=",")
        np.savetxt(os.path.join(save_dir, "Train Precision.csv"), np.array(Train_precision_list), delimiter=",")
        np.savetxt(os.path.join(save_dir, "Train Recall.csv"), np.array(Train_recall_list), delimiter=",")

        np.savetxt(os.path.join(save_dir, "Val Dice.csv"), np.array(Val_dice_list), delimiter=",")
        np.savetxt(os.path.join(save_dir, "Val IoU.csv"), np.array(Val_iou_list), delimiter=",")
        np.savetxt(os.path.join(save_dir, "Val Precision.csv"), np.array(Val_precision_list), delimiter=",")
        np.savetxt(os.path.join(save_dir, "Val Recall.csv"), np.array(Val_recall_list), delimiter=",")

torch.save(net.state_dict(), os.path.join(save_dir, "net.pth"))

np.savetxt(os.path.join(save_dir, "Train Loss.csv"), np.array(Loss_list), delimiter=",")
np.savetxt(os.path.join(save_dir, "Val Loss.csv"), np.array(Val_loss_list), delimiter=",")

np.savetxt(os.path.join(save_dir, "Train Dice.csv"), np.array(Train_dice_list), delimiter=",")
np.savetxt(os.path.join(save_dir, "Train IoU.csv"), np.array(Train_iou_list), delimiter=",")
np.savetxt(os.path.join(save_dir, "Train Precision.csv"), np.array(Train_precision_list), delimiter=",")
np.savetxt(os.path.join(save_dir, "Train Recall.csv"), np.array(Train_recall_list), delimiter=",")

np.savetxt(os.path.join(save_dir, "Val Dice.csv"), np.array(Val_dice_list), delimiter=",")
np.savetxt(os.path.join(save_dir, "Val IoU.csv"), np.array(Val_iou_list), delimiter=",")
np.savetxt(os.path.join(save_dir, "Val Precision.csv"), np.array(Val_precision_list), delimiter=",")
np.savetxt(os.path.join(save_dir, "Val Recall.csv"), np.array(Val_recall_list), delimiter=",")

print("Training finished and all results saved in savemodel/")