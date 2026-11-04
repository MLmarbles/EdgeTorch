from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt

audio_path = next(Path("data/mini_speech_commands/yes").glob("*.wav"))

audio, sample_rate = sf.read(audio_path, dtype="float32")

waveform = torch.from_numpy(audio).T
if waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)

plt.figure(figsize=(10, 3))
plt.plot(torch.arange(waveform.shape[1]) / sample_rate, waveform[0])
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Spoken command: YES")
plt.tight_layout()
plt.savefig("results/waveform.png")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=400,
    hop_length=160,
    n_mels=40,
)

mel_spec = mel_transform(waveform)
log_mel_spec = torch.log(mel_spec + 1e-6)

plt.figure(figsize=(10, 4))
plt.imshow(log_mel_spec[0].numpy(), origin="lower", aspect="auto")
plt.xlabel("Time frames")
plt.ylabel("Mel frequency bins")
plt.title("Log-mel spectrogram: YES")
plt.colorbar(label="Log energy")
plt.tight_layout()
plt.savefig("results/spectrogram.png")
