from typing import Any, cast

import numpy as np
import streamlit as st
from PIL import Image
import keras

from data import preprocess, IMG_SIZE

keras = cast(Any, keras)


@st.cache_resource
def load_model():
    return keras.models.load_model("results/realorai_ann.keras")


model = load_model()

st.title("Real vs AI-Generated Image Detector")
st.caption("ANN trained on CIFAKE (32x32 images). Upload an image to check it.")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    original = Image.open(file).convert("RGB")
    st.image(original, use_container_width=True)

    img = original.resize((IMG_SIZE, IMG_SIZE))
    x = preprocess(np.array(img)[None, ...])
    p_real = float(model.predict(x, verbose=0)[0, 0])

    if p_real >= 0.5:
        st.success(f"REAL (confidence {p_real * 100:.1f}%)")
    else:
        st.error(f"FAKE / AI-generated (confidence {(1 - p_real) * 100:.1f}%)")