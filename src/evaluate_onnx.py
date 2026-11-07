
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dataset import SpeechCommandsDataset

dataset = SpeechCommandsDataset("../data/mini_speech_commands")

val_paths = set(Path("data/splits/val.txt").read_text().splitlines())

indices = []

for i, (path, label) in enumerate(dataset.samples):
    if str(path) in val_paths:
        indices.append(i)

models = {
    "ONNX FP32": "models/speech_cnn.onnx",
    "ONNX INT8": "models/speech_cnn_int8_1000.onnx",
}

for name, model_path in models.items():
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    correct = 0

    for i in indices:
        spectrogram, label = dataset[i]

        x = spectrogram.unsqueeze(0).numpy().astype(np.float32)

        output = session.run(None, {input_name: x})[0]
        prediction = int(np.argmax(output))

        if prediction == label:
            correct += 1

    accuracy = 100 * correct / len(indices)

    print(f"{name}: {accuracy:.2f}%")
    print(f"Samples evaluated: {len(indices)}")
