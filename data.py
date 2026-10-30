# data.py - load the CIFAKE images and turn them into numbers for the ANN
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
CACHE_FILE = "data/cifake_cache.npz"
CLASSES = ["FAKE", "REAL"]        # label 0 = FAKE (AI-generated), label 1 = REAL photo
IMG_SIZE = 32                     # every image is 32 x 32 pixels, 3 colours (RGB)


def load_folder(split):
    """Read all images of one split (train or test). Returns X (pixels) and y (labels)."""
    X, y = [], []
    for label, class_name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, split, class_name)
        for file in os.listdir(folder):
            img = Image.open(os.path.join(folder, file)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            X.append(np.array(img, dtype=np.uint8))
            y.append(label)
    return np.array(X), np.array(y)


def preprocess(X):
    """Same steps for training AND prediction:
    1) scale pixels from 0-255 to 0-1   2) flatten 32x32x3 image into 3072 numbers."""
    X = X.astype("float32") / 255.0
    return X.reshape(len(X), -1)


def get_data():
    """Returns train, validation and test sets, ready for the ANN."""
    if os.path.exists(CACHE_FILE):                     # fast path: images already read once
        d = np.load(CACHE_FILE)
        X_train, y_train, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    else:
        print("Reading images (only the first time, takes a few minutes)...")
        X_train, y_train = load_folder("train")
        X_test, y_test = load_folder("test")
        np.savez_compressed(CACHE_FILE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # keep 10% of the training images aside for validation (test set is never touched in training)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    return (preprocess(X_train), y_train,
            preprocess(X_val), y_val,
            preprocess(X_test), y_test)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    print("Train:", X_train.shape, " Validation:", X_val.shape, " Test:", X_test.shape)
    print("Train images per class (FAKE, REAL):", np.bincount(y_train))
