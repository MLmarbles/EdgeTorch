import random
from pathlib import Path

import numpy as np

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from dataset import SpeechCommandsDataset

class SpeechCalibrationReader(CalibrationDataReader):
    def __init__(self):
        dataset = SpeechCommandsDataset("../data/mini_speech_commands")
        train_paths = set(Path("data/splits/train.txt").read_text().splitlines())

        train_indices = []

        for i, (path, label) in enumerate(dataset.samples):
            if str(path) in train_paths:
                train_indices.append(i)

        rng = random.Random(67)
        indices = rng.sample(
            train_indices, min(1000, len(train_indices))
        )

        self.dataset = dataset
        self.indices = iter(indices)

    def get_next(self):
        index = next(self.indices, None)

        if index is None:
            return None

        spectrogram, _ = self.dataset[index]

        x = spectrogram.unsqueeze(0).numpy()
        x = x.astype(np.float32)

        return {"spectrogram": x}

reader = SpeechCalibrationReader()

quantize_static(
    model_input="models/speech_cnn.onnx",
    model_output="models/speech_cnn_int8_1000.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=["Conv", "Gemm", "MatMul"],
    calibrate_method=CalibrationMethod.MinMax,
)

print("quantization complete")