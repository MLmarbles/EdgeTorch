import time
import statistics

import torch
import numpy as np
import onnxruntime as ort

from dataset import SpeechCommandsDataset
from model import SpeechCNN

torch.set_num_threads(1)

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "models/speech_cnn_int8_1000.onnx",
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

dataset = SpeechCommandsDataset("../data/mini_speech_commands")
spectrogram, label = dataset[0]
x = spectrogram.unsqueeze(0)
x_numpy = x.numpy()
input_name = session.get_inputs()[0].name

model = SpeechCNN()
model.load_state_dict(
    torch.load(
        "models/best_model_sgd_dropout.pt",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

with torch.inference_mode():
    pytorch_output = model(x).numpy()

onnx_output = session.run(
    None, {input_name: x_numpy}
)[0]

difference = np.max(
    np.abs(pytorch_output - onnx_output)
)

print(f"Max output difference: {difference:.8f}")

times = []

for _ in range(50):
    session.run(None, {input_name: x_numpy})

for _ in range(500):
    start = time.perf_counter()

    session.run(None, {input_name: x_numpy})

    elapsed = time.perf_counter() - start
    times.append(elapsed * 1000)

print(f"Mean latency: {statistics.mean(times):.3f} ms")
print(f"Median latency: {statistics.median(times):.3f} ms")
print(f"Min latency: {min(times):.3f} ms")
print(f"Max latency: {max(times):.3f} ms")
