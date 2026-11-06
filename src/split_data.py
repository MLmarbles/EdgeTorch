from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from dataset import SpeechCommandsDataset


dataset = SpeechCommandsDataset(
    "../data/mini_speech_commands"
)

speaker_ids = [
    path.stem.split("_nohash_")[0]
    for path, label in dataset.samples
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, remaining_idx = next(
    splitter.split(dataset.samples, groups=speaker_ids)
)

remaining_speakers = [
    speaker_ids[i] for i in remaining_idx
]

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=42,
)

val_relative, test_relative = next(
    splitter2.split(
        remaining_idx,
        groups=remaining_speakers,
    )
)

val_idx = remaining_idx[val_relative]
test_idx = remaining_idx[test_relative]


def get_speakers(indices):
    return {speaker_ids[i] for i in indices}


train_speakers = get_speakers(train_idx)
val_speakers = get_speakers(val_idx)
test_speakers = get_speakers(test_idx)

print("Training recordings:", len(train_idx))
print("Validation recordings:", len(val_idx))
print("Test recordings:", len(test_idx))

print("Training speakers:", len(train_speakers))
print("Validation speakers:", len(val_speakers))
print("Test speakers:", len(test_speakers))

assert train_speakers.isdisjoint(val_speakers)
assert train_speakers.isdisjoint(test_speakers)
assert val_speakers.isdisjoint(test_speakers)

Path("data/splits").mkdir(parents=True, exist_ok=True)

for name, indices in [
    ("train", train_idx),
    ("val", val_idx),
    ("test", test_idx),
]:
    with open(f"data/splits/{name}.txt", "w") as f:
        for i in indices:
            wav_path, label = dataset.samples[i]
            f.write(f"{wav_path}\n")
