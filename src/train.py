import torch
from torch.utils.data import DataLoader, Subset

from dataset import SpeechCommandsDataset
from model import SpeechCNN

from pathlib import Path

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

train_loader = make_loader("train", shuffle=True)
val_loader = make_loader("val", shuffle=False)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

Path("models").mkdir(exist_ok=True)
best_val_accuracy = 0.0

for epoch in range(1, 31):
    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        predictions = model(x)
        loss = loss_fn(predictions, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        train_correct += (predictions.argmax(1) == y).sum().item()
        train_total += y.size(0)

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            predictions = model(x)

            val_correct += (predictions.argmax(1) == y).sum().item()
            val_total += y.size(0)

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total
    average_loss = train_loss / train_total


    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {average_loss:.4f} | "
        f"Train: {train_accuracy:.1%} | "
        f"Val: {val_accuracy:.1%}",
        flush=True,
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "models/best_model_sgd_dropout.pt")

print(f"Best validation accuracy: {best_val_accuracy:.1%}")
