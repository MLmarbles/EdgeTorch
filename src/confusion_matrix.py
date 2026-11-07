from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import SpeechCommandsDataset

dataset = SpeechCommandsDataset("../data/mini_speech_commands")

val_lines = Path("data/splits/val.txt").read_text().splitlines()

val_keys = set()

for line in val_lines:
    if line.strip():
        path = Path(line)

        folder = path.parent.name
        filename = path.name

        val_keys.add((folder, filename))

indices = []

for i, (path, label) in enumerate(dataset.samples):
    if (path.parent.name, path.name) in val_keys:
        indices.append(i)

label_names = {
    label: path.parent.name
    for path, label in dataset.samples
}

labels = sorted(label_names)
class_names = [label_names[label] for label in labels]

models = {
    "FP32": "models/speech_cnn.onnx",
    "INT8": "models/speech_cnn_int8_1000.onnx",
}

for model_name, model_path in models.items():
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    y_true = []
    y_pred = []

    for i in indices:
        spectrogram, label = dataset[i]

        x = spectrogram.unsqueeze(0).numpy().astype(np.float32)

        output = session.run(
            None, {input_name: x}
        )[0]

        prediction = int(np.argmax(output))

        y_true.append(label)
        y_pred.append(prediction)

    cm = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    accuracy = np.trace(cm) / np.sum(cm)

    print(f"\n{model_name} accuracy: {accuracy:.2%}")

    for i, name in enumerate(class_names):
        correct = cm[i, i]
        total = cm[i].sum()

        print(f"{name}: {correct}/{total} correct")

    fig, ax = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    output_path = f"../results/confusion_{model_name.lower()}.png"

    plt.savefig(output_path, dpi=150)
    plt.close()
