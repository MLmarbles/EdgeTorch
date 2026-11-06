import torch
from torch.utils.data import DataLoader, Subset

from dataset import SpeechCommandsDataset
from model import SpeechCNN

dataset = SpeechCommandsDataset("../data/mini_speech_commands")

path_to_idx = {
    str(path): i
    for i, (path, label) in enumerate(dataset.samples)
}

with open("data/splits/train.txt") as f:
    train_indices = [
        path_to_idx[line.strip()]
        for line in f
    ]

train_dataset = Subset(dataset, train_indices)

loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SpeechCNN().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x, y = next(iter(loader))
x, y = x.to(device), y.to(device)

model.train()

optimizer.zero_grad()
predictions = model(x)
loss = loss_fn(predictions, y)
loss.backward()
optimizer.step()

print("Device:", device)
print("Training samples:", len(train_dataset))
print("Loss:", loss.item())