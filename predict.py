# predict.py - check any image: is it REAL or AI-generated?
# usage:  python predict.py path/to/image.jpg
import sys
from typing import Any, cast
import numpy as np
from PIL import Image
import keras

from data import preprocess, IMG_SIZE

keras = cast(Any, keras)
model = keras.models.load_model("results/realorai_ann.keras")

for path in sys.argv[1:]:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))   # same size as training
    x = preprocess(np.array(img)[None, ...])                             # same preprocessing as training
    p_real = float(model.predict(x, verbose=0)[0, 0])
    label = "REAL" if p_real >= 0.5 else "FAKE (AI-generated)"
    confidence = p_real if p_real >= 0.5 else 1 - p_real
    print(f"{path}: {label}  (confidence {confidence * 100:.1f}%)")
