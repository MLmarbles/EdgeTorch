import torch
from torch.utils.data import DataLoader, Subset

from dataset import SpeechCommandsDataset
from model import SpeechCNN

dataset = SpeechCommandsDataset("../data/mini_speech_commands")

path_to_idx = {
    str(path): i
    for i, (path, label) in enumerate(dataset.samples)
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SpeechCNN().to(device)

def make_loader(split_name, shuffle):
    with open(f"data/splits/{split_name}.txt") as f:
        indices = [
            path_to_idx[line.strip()]
            for line in f if line.strip()
        ]

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=64,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

print(device)

test_loader = make_loader("test", shuffle=False)

model.load_state_dict(
    torch.load(
        "models/best_model_sgd_dropout.pt",
        map_location=device,
        weights_only=True
    )
)

model.eval()

correct = 0
total = 0

with torch.inference_mode():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        predictions = model(x)
        predicted_labels = predictions.argmax(dim=1)

        correct += (predicted_labels == y).sum().item()
        total += y.size(0)

print(f"Test accuracy: {correct / total:.1%}")
