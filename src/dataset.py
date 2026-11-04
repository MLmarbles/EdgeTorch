from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.classes = sorted(
            p.name for p in self.root_dir.iterdir()
            if p.is_dir()
        )

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.classes)
        }

        self.samples = []

        for class_name in self.classes:
            folder = self.root_dir / class_name

            for wav_path in sorted(folder.glob("*.wav")):
                self.samples.append(
                    (wav_path, self.class_to_idx[class_name])
                )

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=40,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]

        audio, sample_rate = sf.read(
            wav_path, dtype="float32"
        )

        if sample_rate != 16000:
            raise ValueError(
                f"sample rate doesnt match 16000: {sample_rate}"
            )

        waveform = torch.from_numpy(audio)

        if waveform.ndim != 1:
            raise ValueError("expected mono audio")

        waveform = waveform[:16000]
        waveform = F.pad(
            waveform, (0, 16000 - waveform.shape[0])
        )

        waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6)

        return log_mel, label


if __name__ == "__main__":
    dataset = SpeechCommandsDataset(
        "data/mini_speech_commands"
    )

    print("Classes:", dataset.classes)
    print("Class mapping:", dataset.class_to_idx)
    print("Total samples:", len(dataset))

    spectrogram, label = dataset[0]

    print("One spectrogram:", spectrogram.shape)
    print("One label:", label)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(loader))

    print("Batch X shape:", batch_x.shape)
    print("Batch Y shape:", batch_y.shape)
