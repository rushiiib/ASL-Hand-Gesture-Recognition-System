#Hand Gesture Recognition System for ASL Alphabet - group 16
#this file prints the final test accuracy and a confusion matrix on test data

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

IMG_SIZE = 128   # must match train.py
MODEL_PATH = "converted_keras/keras_model.h5"
BASE_DIR = "/Users/aarushibhatnagar/Desktop/School/fall 2025/cmpt 310/Project/split"

# loading labels
LABELS_PATH = "converted_keras/labels.txt"
with open(LABELS_PATH, "r") as f:
    labels = [ln.strip() for ln in f if ln.strip()]

#load test data
def load_split(split):
    X, y = [], []
    split_dir = os.path.join(BASE_DIR, split)
    class_names = sorted(os.listdir(split_dir))

    for idx, cname in enumerate(class_names):
        folder = os.path.join(split_dir, cname)
        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            if img_name.startswith("."):
                continue

            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img / 255.0)
            y.append(idx)

    return np.array(X), np.array(y)

print("Loading test data...")
X_test, y_test = load_split("test")
print(f"Loaded {len(X_test)} test images.")

#trained model
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

print("Predicting on test set...")
pred_probs = model.predict(X_test)
y_pred = np.argmax(pred_probs, axis=1)

#accuracy
test_acc = accuracy_score(y_test, y_pred)
print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%\n")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

#confusion matrix
plt.figure(figsize=(14, 14))
disp.plot(cmap="Purples", xticks_rotation=45)
plt.title("ASL Confusion Matrix — Test Set")
plt.savefig("test_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved confusion matrix as test_confusion_matrix.png")