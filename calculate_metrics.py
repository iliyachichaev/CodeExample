import json
import os
import cv2
import numpy as np

TRAIN_IMG_PATH = 'dataset/train/img'
MASK_PATH = 'dataset/train/mask'
TRAIN_JSON_PATH = 'dataset/train/json'
SEGM_RESULT_PATH = 'dataset/train/result_segm'
CLASS_RESULT_PATH = 'dataset/train/result_class'
PNG_FILE_FILTER = '.png'

LABELS = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus', 'moraxella_catarrhalis', 'c_kefir', 'ent_cloacae']

label_metrics = np.zeros((len(LABELS), len(LABELS)), int)
seg_metrics = []

def set_metrics(filename, json_path):
    global label_metrics, seg_metrics
    
    # Get true mask
    true_mask_filename = filename[:-len(PNG_FILE_FILTER)] + '_mask.png'
    true_mask = cv2.imread(os.path.join(MASK_PATH, true_mask_filename), cv2.IMREAD_GRAYSCALE)
    
    # Get predicted mask
    result_mask_filename = filename[:-len(PNG_FILE_FILTER)] + '_result.png'
    result_mask = cv2.imread(os.path.join(SEGM_RESULT_PATH, result_mask_filename), cv2.IMREAD_GRAYSCALE)
    
    # Calculate segmentation metric (IOU)
    seg_metrics += [np.count_nonzero(np.logical_and(true_mask, result_mask)) /
                    np.count_nonzero(np.logical_or(true_mask, result_mask))]
    
    # Get true label
    json_filename = filename[0:3] + '.json'
    with open(os.path.join(json_path, json_filename), 'r') as f:
        layout = json.load(f)
    true_label = layout['shapes'][0]['label']
    
    # Get predicted label
    class_filename = filename[:-len(PNG_FILE_FILTER)] + '_class.txt'
    with open(os.path.join(CLASS_RESULT_PATH, class_filename), 'r') as f:
        result_label = f.readline()
    
    # Calculate classification metrics (precisions)
    label_metrics[LABELS.index(result_label)][LABELS.index(true_label)] += 1


def calculate_metrics():
    mean_iou = np.mean(seg_metrics)

    precisions = dict.fromkeys(LABELS, 0.)
    
    print(label_metrics)

    for label in LABELS:
        i = LABELS.index(label)
        precisions[label] = label_metrics[i][i] / np.sum(label_metrics[i, :])
    mean_precision = np.mean(list(precisions.values()))

    score = mean_iou + np.sum(list(precisions.values()))

    print(f'mean_iou: {mean_iou}')
    for k, v in precisions.items():
        print(f'precision_{k}: {v}')
    print(f'mean_precision: {mean_precision}\nscore: {score}')


def main(img_path, json_path):
    for filename in os.listdir(img_path):
        set_metrics(filename, json_path)
    calculate_metrics()


main(TRAIN_IMG_PATH, TRAIN_JSON_PATH)
