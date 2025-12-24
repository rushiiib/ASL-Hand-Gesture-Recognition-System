#Hand Gesture Recognition System for ASL Alphabet - group 16
#This file is used to split the captured dataset into train/val/test (70/15/15)
#it creates a new folder called split 
#for the paramters, make sure we are in the right directory (change the directory accordingly)

import os
import shutil
import random

# parameters 
ORIGINAL_DATA_DIR = "/Users/aarushibhatnagar/Desktop/School/fall 2025/cmpt 310/Project/images"  # original dataset folder
SPLIT_DIR = "/Users/aarushibhatnagar/Desktop/School/fall 2025/cmpt 310/Project/split"        # folder to save train/val/test
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# create folders
for split in ['train', 'val', 'test']:
    split_path = os.path.join(SPLIT_DIR, split)
    os.makedirs(split_path, exist_ok=True)

# Split data
labels = [d for d in os.listdir(ORIGINAL_DATA_DIR) if os.path.isdir(os.path.join(ORIGINAL_DATA_DIR, d))]
print("Classes found:", labels)

for label in labels:
    label_dir = os.path.join(ORIGINAL_DATA_DIR, label)
    images = os.listdir(label_dir)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }

    for split, split_images in splits.items():
        split_label_dir = os.path.join(SPLIT_DIR, split, label)
        os.makedirs(split_label_dir, exist_ok=True)
        for img_name in split_images:
            src = os.path.join(label_dir, img_name)
            dst = os.path.join(split_label_dir, img_name)
            shutil.copy(src, dst)

    print(f"{label}: {n_train} train, {n_val} val, {n_test} test images")

print("Dataset split completed successfully!")
