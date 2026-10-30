# train.py - train the ANN and save the model + training curves
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras

from data import get_data
from model import build_model

os.makedirs("results", exist_ok=True)

X_train, y_train, X_val, y_val, X_test, y_test = get_data()
model = build_model()
model.summary()

callbacks = [
    # stop when validation loss stops improving, and go back to the best weights
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    # lower the learning rate when learning gets stuck
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=128,
    callbacks=callbacks,
)

model.save("results/realorai_ann.keras")
with open("results/history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

# training curves: accuracy and loss for train vs validation
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=120)
print("Saved model and curves in results/")
