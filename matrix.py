import os
import numpy as np
from PIL import Image


def prepare_binary_mask(img, threshold=0.5, invert=True):
    img = np.asarray(img)

    if img.dtype != np.bool_:
        img = img > threshold

    if invert:
        img = np.logical_not(img)

    return img


def calculate_dice_coefficient(pred_img, true_img):
    intersection = np.logical_and(pred_img, true_img).sum()
    dice = (2.0 * intersection) / (pred_img.sum() + true_img.sum() + 1e-7)
    return dice


def calculate_iou(pred_img, true_img):
    intersection = np.logical_and(pred_img, true_img).sum()
    union = np.logical_or(pred_img, true_img).sum()
    iou = intersection / (union + 1e-7)
    return iou


def calculate_precision(pred_img, true_img):
    true_positives = np.logical_and(pred_img, true_img).sum()
    false_positives = np.logical_and(pred_img, np.logical_not(true_img)).sum()
    precision = true_positives / (true_positives + false_positives + 1e-7)
    return precision


def calculate_recall(pred_img, true_img):
    true_positives = np.logical_and(pred_img, true_img).sum()
    false_negatives = np.logical_and(np.logical_not(pred_img), true_img).sum()
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    return recall


if __name__ == "__main__":
    pred_folder = "savemodel/pred"
    true_folder = "datasets/test/mask"
    save_folder = "savemodel"

    os.makedirs(save_folder, exist_ok=True)

    pred_files = sorted(os.listdir(pred_folder))
    true_files = sorted(os.listdir(true_folder))

    if len(pred_files) != len(true_files):
        raise ValueError("Number of predicted images and ground truth images does not match.")

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(len(pred_files)):
        pred_path = os.path.join(pred_folder, pred_files[i])
        true_path = os.path.join(true_folder, true_files[i])

        pred_img = np.array(Image.open(pred_path).convert("L")) / 255.0
        true_img = np.array(Image.open(true_path).convert("L")) / 255.0

        pred_img = prepare_binary_mask(pred_img, threshold=0.5, invert=True)
        true_img = prepare_binary_mask(true_img, threshold=0.5, invert=True)

        if pred_img.shape != true_img.shape:
            raise ValueError(f"Image size mismatch: {pred_files[i]} and {true_files[i]}")

        dice_scores.append(calculate_dice_coefficient(pred_img, true_img))
        iou_scores.append(calculate_iou(pred_img, true_img))
        precision_scores.append(calculate_precision(pred_img, true_img))
        recall_scores.append(calculate_recall(pred_img, true_img))

    average_dice = np.mean(dice_scores)
    average_iou = np.mean(iou_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)

    print(f"Average Dice: {average_dice:.6f}")
    print(f"Average IoU: {average_iou:.6f}")
    print(f"Average Precision: {average_precision:.6f}")
    print(f"Average Recall: {average_recall:.6f}")

    result_file = os.path.join(save_folder, "test_metrics.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Average Dice: {average_dice:.6f}\n")
        f.write(f"Average IoU: {average_iou:.6f}\n")
        f.write(f"Average Precision: {average_precision:.6f}\n")
        f.write(f"Average Recall: {average_recall:.6f}\n")

    print(f"Results saved to: {result_file}")