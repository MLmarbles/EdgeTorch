import torch
import onnx

from model import SpeechCNN

model = SpeechCNN()

model.load_state_dict(
    torch.load(
        "models/best_model_sgd_dropout.pt",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

dummy_input = torch.randn(1, 1, 40, 101)

torch.onnx.export(
    model,
    (dummy_input,),
    "models/speech_cnn.onnx",
    input_names=["spectrogram"],
    output_names=["logits"],
    opset_version=18,
    dynamo=True,
)

onnx_model = onnx.load("models/speech_cnn.onnx")
onnx.checker.check_model(onnx_model)
