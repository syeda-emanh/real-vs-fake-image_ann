# evaluate.py - test the trained model on the 20,000 unseen test images
import json
from typing import Any, cast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

from data import get_data, CLASSES

keras = cast(Any, keras)
_, _, _, _, X_test, y_test = get_data()
model = keras.models.load_model("results/realorai_ann.keras")

probs = model.predict(X_test, verbose=0).ravel()
y_pred = (probs >= 0.5).astype(int)          # probability >= 0.5 -> REAL, otherwise FAKE

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="macro"),
    "recall": recall_score(y_test, y_pred, average="macro"),
    "f1": f1_score(y_test, y_pred, average="macro"),
}
print("\nTest results:")
for k, v in metrics.items():
    print(f"  {k:9s}: {v:.4f}")
report = str(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
print()
print(report)

cm = confusion_matrix(y_test, y_pred)
metrics["confusion_matrix"] = cm.tolist()
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (test set)")
plt.savefig("results/confusion_matrix.png", dpi=120, bbox_inches="tight")
print("Saved results/metrics.json and results/confusion_matrix.png")
